# -*- coding: utf-8 -*-
"""Candidate D, oracle test: does a SPECTRAL eviction criterion beat an
attention-score one at equal budget, with cost ignored?

SETUP. Prefill a prompt with the full cache, then drop a fraction of the prompt's
KV entries and score the continuation. Eviction is simulated exactly by a 4D
additive attention mask: prompt queries attend normally (so the retained K/V are
the ones the model actually computed), and every continuation query can attend
only to the kept prompt positions plus the continuation so far.

THE BASELINE THAT MATTERS IS NOT H2O. StructKV (2604.06746) already evicts on
global in-degree centrality aggregated across depth, with the same motivating
story -- keep tokens that are structurally important even when not locally
salient -- and GraphKV (EMNLP 2025) already formulates the cache as a graph. So
beating accumulated attention score proves nothing new. The criterion has to beat
in-degree centrality, which is why both are here.

Accumulated attention score summed over queries IS in-degree on the attention
graph; the two differ only in aggregation, so both variants are run: `attn_h2o`
sums over queries within layer then averages layers, `inC_struct` builds the
depth-aggregated graph first and takes weighted in-degree on it.

EFFECTIVE RESISTANCE USES THE COMBINATORIAL LAPLACIAN, deliberately departing
from the package's rw default. Spielman-Srivastava is stated for L = D - W and
R_ij = (e_i-e_j)^T L^+ (e_i-e_j); the node aggregate of edge leverage is
score_i = sum_j w_ij R_ij, which is the quantity SS samples by. Using the
normalised Laplacian here would not be the theorem's quantity.

PROTECTED SET. Every criterion keeps the same first `SINK` and last `RECENT`
positions, so the comparison is only about which middle tokens are chosen.
"""
import argparse
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "meta-llama/Llama-3.2-1B"
WIKI = (os.path.expanduser("~") + "/.cache/huggingface/hub/datasets--wikitext/"
        "snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/"
        "wikitext-103-raw-v1/train-00000-of-00002.parquet")
WIKI2 = (os.path.expanduser("~") + "/.cache/huggingface/hub/datasets--wikitext/"
         "snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/"
         "wikitext-2-raw-v1/train-00000-of-00001.parquet")
PROMPT_LEN, CONT_LEN = 512, 128
SINK, RECENT = 4, 32
BUDGETS = [0.50, 0.25, 0.10]
NEG = -1e30


# ---------------------------------------------------------------- criteria
def graph_from_attn(atts, upto):
    """Head-uniform, symmetrised, depth-averaged attention graph over the first
    `upto` positions. Matches the aggregation the rest of the project uses,
    except that it is summed across layers because the keep-set is global."""
    W = None
    for A in atts:
        a = A[0, :, :upto, :upto].to(torch.float32).mean(0)
        a = 0.5 * (a + a.T)
        W = a if W is None else W + a
    return (W / len(atts)).cpu().numpy()


def score_attn_h2o(atts, upto):
    """Accumulated attention received, per layer, then averaged over layers."""
    s = None
    for A in atts:
        a = A[0, :, :upto, :upto].to(torch.float32).mean(0)
        v = a.sum(0)
        s = v if s is None else s + v
    return (s / len(atts)).cpu().numpy()


def score_indegree(W):
    """Weighted in-degree on the depth-aggregated graph (StructKV-like)."""
    return W.sum(axis=1)


def _pinv_L(W):
    d = W.sum(axis=1)
    L = np.diag(d) - W
    return np.linalg.pinv(L, hermitian=True)


def score_eff_resistance(W):
    """score_i = sum_j w_ij R_ij, the node aggregate of SS edge leverage."""
    P = _pinv_L(W)
    dg = np.diag(P)
    R = dg[:, None] + dg[None, :] - 2.0 * P
    np.fill_diagonal(R, 0.0)
    return (W * R).sum(axis=1)


def score_fiedler(W):
    """|v2| on the rw-normalised Laplacian: distance from the partition axis."""
    d = np.maximum(W.sum(axis=1), 1e-12)
    Dm = np.diag(d ** -0.5)
    L = np.eye(len(W)) - Dm @ W @ Dm
    L = 0.5 * (L + L.T)
    ev, U = np.linalg.eigh(L)
    return np.abs(U[:, 1])


def score_lplus_diag(W):
    """Small L^+_ii = central. Negated so that larger = keep."""
    return -np.diag(_pinv_L(W))


def score_key_norm(pk, upto):
    """Mean L2 norm of the cached key vector per position, over heads and
    layers. Devoto et al. (EMNLP 2024) report that LOW key norm marks tokens
    that receive high attention, so the criterion is run in BOTH directions --
    getting the sign backwards would be an unfair test of someone else's
    method, and which direction helps query-agnostically is exactly what is in
    question."""
    s = None
    for lay in pk.layers:
        k = lay.keys[0, :, :upto, :].to(torch.float32)      # [H, upto, d]
        v = k.norm(dim=-1).mean(0)                          # [upto]
        s = v if s is None else s + v
    return (s / len(pk.layers)).cpu().numpy()


CRITERIA = ["attn_h2o", "inC_struct", "eff_res", "fiedler", "lplus",
            "key_norm_low", "key_norm_high", "stratified", "random"]


def keepset(name, scores, n, k, rng):
    """Protected head and tail, remainder chosen by criterion."""
    prot = list(range(min(SINK, n))) + list(range(max(0, n - RECENT), n))
    prot = sorted(set(prot))
    free = [i for i in range(n) if i not in set(prot)]
    need = max(0, k - len(prot))
    if need >= len(free):
        return sorted(set(prot) | set(free))
    if name == "random":
        pick = list(rng.choice(free, size=need, replace=False))
    elif name == "stratified":
        # Deterministic uniform coverage: the control that separates "random
        # helps" from "spreading retained tokens across the context helps".
        # No randomness at all, so any advantage is positional coverage.
        pos = np.linspace(0, len(free) - 1, need)
        pick = [free[int(round(p))] for p in pos]
    else:
        order = sorted(free, key=lambda i: -float(scores[i]))
        pick = order[:need]
    return sorted(set(prot) | set(pick))


# ------------------------------------------------------------------ masking
def build_mask(keep, n_prompt, total, device, dtype):
    """Causal, plus: continuation queries see only kept prompt positions."""
    m = torch.full((total, total), NEG, device=device, dtype=dtype)
    m = torch.triu(m, diagonal=1)                    # causal
    blocked = torch.ones(n_prompt, dtype=torch.bool, device=device)
    blocked[torch.tensor(keep, device=device)] = False
    m[n_prompt:, :n_prompt] = torch.where(
        blocked[None, :], torch.full_like(m[n_prompt:, :n_prompt], NEG),
        m[n_prompt:, :n_prompt])
    return m[None, None]


@torch.no_grad()
def cont_nll(model, ids, keep, n_prompt, target_slice):
    total = ids.shape[1]
    mask = build_mask(keep, n_prompt, total, ids.device, model.dtype)
    out = model(input_ids=ids, attention_mask=mask)
    lg = out.logits[0, :-1].float()
    tg = ids[0, 1:]
    lp = torch.log_softmax(lg, -1).gather(-1, tg[:, None])[:, 0]
    s0, s1 = target_slice
    return float(-lp[s0 - 1:s1 - 1].mean().item())


# --------------------------------------------------------------------- data
def load_docs(tok, n_docs, need_tokens):
    path = WIKI if os.path.exists(WIKI) else WIKI2
    txt = pd.read_parquet(path)["text"].tolist()
    buf, docs = [], []
    for t in txt:
        t = str(t).strip()
        if len(t) < 100 or t.startswith("="):
            continue
        buf.append(t)
        if len(" ".join(buf)) > need_tokens * 6:
            ids = tok(" ".join(buf), return_tensors="pt",
                      add_special_tokens=True)["input_ids"]
            if ids.shape[1] >= need_tokens:
                docs.append(ids[:, :need_tokens])
            buf = []
            if len(docs) >= n_docs:
                break
    return docs


def make_needle(tok, n_docs, n_prompt, rng):
    """Needle-in-haystack. The needle attracts almost no attention during
    prefill -- nothing refers to it until the question -- which is precisely
    the case a score-based criterion should mishandle and a structural one
    should catch. If spectral eviction has an advantage anywhere, here."""
    path = WIKI if os.path.exists(WIKI) else WIKI2
    txt = [str(t).strip() for t in pd.read_parquet(path)["text"].tolist()
           if len(str(t).strip()) > 200 and not str(t).strip().startswith("=")]
    items = []
    for i in range(n_docs):
        code = int(rng.integers(1000, 9999))
        needle = f" The secret passcode is {code}. "
        hay = " ".join(txt[(i * 7) % len(txt): (i * 7) % len(txt) + 40])
        ids_h = tok(hay, add_special_tokens=False)["input_ids"]
        body = tok.decode(ids_h[:n_prompt])
        cut = int(len(body) * float(rng.uniform(0.15, 0.5)))
        cut = body.find(" ", cut)
        doc = body[:cut] + needle + body[cut:]
        q = (f"\n\nQuestion: What is the secret passcode?\n"
             f"Answer: The secret passcode is {code}.")
        full = tok(doc, return_tensors="pt", add_special_tokens=True)
        qq = tok(q, return_tensors="pt", add_special_tokens=False)
        p = full["input_ids"][:, :n_prompt]
        ids = torch.cat([p, qq["input_ids"]], dim=1)
        ans = tok(f" {code}.", add_special_tokens=False)["input_ids"]
        # Which prompt positions actually carry the needle. Retention of these
        # is the mechanism the whole hypothesis rests on, so it is measured
        # directly rather than inferred from the NLL.
        toks = [tok.decode([t]) for t in p[0].tolist()]
        span = [j for j, s in enumerate(toks)
                if s.strip() and s.strip() in str(code)] or []
        joined = "".join(toks)
        if str(code) in joined and not span:
            span = [j for j, s in enumerate(toks) if s.strip().isdigit()]
        items.append((ids, p.shape[1], len(ans), span))
    return items


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=20)
    ap.add_argument("--out", default="oracle.csv")
    a = ap.parse_args()

    dev = "cuda"
    rng = np.random.default_rng(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager").to(dev).eval()

    rows = []
    tasks = [("ppl", load_docs(tok, a.docs, PROMPT_LEN + CONT_LEN)),
             ("needle", make_needle(tok, a.docs, PROMPT_LEN, rng))]

    for task, data in tasks:
        for di, item in enumerate(data):
            span = []
            if task == "ppl":
                ids = item.to(dev)
                n_p, tgt = PROMPT_LEN, (PROMPT_LEN, PROMPT_LEN + CONT_LEN)
            else:
                ids, n_p, n_ans, span = item
                ids = ids.to(dev)
                tgt = (ids.shape[1] - n_ans, ids.shape[1])

            with torch.no_grad():
                pre = model(input_ids=ids[:, :n_p], output_attentions=True,
                            use_cache=True)
            atts = pre.attentions
            W = graph_from_attn(atts, n_p)
            kn = score_key_norm(pre.past_key_values, n_p)
            sc = {"attn_h2o": score_attn_h2o(atts, n_p),
                  "inC_struct": score_indegree(W),
                  "eff_res": score_eff_resistance(W),
                  "fiedler": score_fiedler(W),
                  "lplus": score_lplus_diag(W),
                  "key_norm_low": -kn, "key_norm_high": kn,
                  "stratified": None, "random": None}
            del pre, atts
            torch.cuda.empty_cache()

            full = cont_nll(model, ids, list(range(n_p)), n_p, tgt)
            rows.append(dict(task=task, doc=di, criterion="full", budget=1.0,
                             nll=full, delta=0.0))
            for bud in BUDGETS:
                k = int(round(bud * n_p))
                for c in CRITERIA:
                    ks = keepset(c, sc[c], n_p, k, rng)
                    v = cont_nll(model, ids, ks, n_p, tgt)
                    kept = (float(len(set(ks) & set(span)) / len(span))
                            if span else np.nan)
                    rows.append(dict(task=task, doc=di, criterion=c,
                                     budget=bud, nll=v, delta=v - full,
                                     needle_kept=kept))
            if (di + 1) % 5 == 0:
                print(f"  {task} {di+1}/{len(data)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} rows={len(df)}")


if __name__ == "__main__":
    main()
