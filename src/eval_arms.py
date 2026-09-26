# -*- coding: utf-8 -*-
"""Measure every arm on ONE shared latent list -- the six-level crossed design.

All arms were trained from seed 0, so W0 is identical and latent i denotes the
same initial direction in each. The live sets are intersected, one uniform sample
is drawn from the intersection, and every arm is measured on exactly that sample.
Anything else makes the design nested and the interaction inestimable.

Each arm carries its own k, so the TopK gate is taken from that arm's full-width
pre-activation -- gating a 240-column subset would behave as if the dictionary
were 240 wide.

POSITION IS A SECOND DESIGN FACTOR, and --pos_mode chooses how it is handled.

  per_arm (default, and what the first six-arm run did): each arm's positions
    are its OWN top-n activating positions for that latent. This mirrors
    published practice -- everyone measures a latent where it fires hardest --
    but it makes position NESTED within (latent, arm). Arms then disagree about
    where to measure: measured Jaccard overlap of the position sets is 0.18 and
    only 17% of arm pairs pick the same single top position (E12,
    src/position_structure.py). Under this mode the latent x arm interaction
    mixes "the latent behaves differently under this fit" with "the arms chose
    different tokens", and the two are not separable after the fact.

  shared: positions are chosen ONCE per latent, arm-symmetrically, and every arm
    is measured at exactly those positions. Candidates are ranked by the MINIMUM
    gated activation across arms, so the latent demonstrably fires in every arm
    at every selected position and no arm is privileged as the reference. This
    makes the design fully crossed -- latent x arm x position -- so position
    becomes an estimable factor and the interaction is free of selection.

Run both. The contrast between them IS the estimate of how much of the
latent x arm interaction was position selection rather than the latent.
"""
import argparse
import glob
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
import torch

from train_saes import DEV, TopKSAE
from eval_saes import WIKI, kl_batch
from transformers import AutoModelForCausalLM, AutoTokenizer
import model_configs


def load_arms(arm_dir, cfg):
    """Each checkpoint carries its own d_model/width when present (arms
    trained after model_configs was introduced); older Gemma-2-2B checkpoints
    predate that field, so fall back to the requested base's config -- they
    were only ever trained on gemma2-2b anyway."""
    arms = {}
    for p in sorted(glob.glob(f"{arm_dir}/arm_*.pt")):
        d = torch.load(p, map_location=DEV)
        d_model = int(d.get("d_model", cfg["d_model"]))
        width = int(d.get("width", cfg["width"]))
        m = TopKSAE(d_model, width, int(d["k"]), 0).to(DEV)
        m.load_state_dict(d["state"])
        arms[d["tag"]] = m.eval()
    return arms


@torch.no_grad()
def gated(m, X, sel):
    """Activation of the selected latents under this arm's full-width TopK."""
    x = X - m.b_dec
    pre_sel = x @ m.W_enc[:, sel] + m.b_enc[sel]
    full = x @ m.W_enc + m.b_enc
    kth = torch.topk(full, m.k, dim=-1).values[:, -1:]
    del full
    return torch.relu(pre_sel) * (pre_sel >= kth).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seq", type=int, default=96)
    ap.add_argument("--n_feat", type=int, default=240)
    ap.add_argument("--n_pos", type=int, default=6)
    ap.add_argument("--out", default="results/eval_arms.csv")
    ap.add_argument("--base", default="gemma2-2b",
                    choices=list(model_configs.MODELS),
                    help="which base model the arms in --arm_dir were fit on")
    ap.add_argument("--arm_dir", default=None,
                    help="defaults to the base's own arm directory")
    ap.add_argument("--min_arms", type=int, default=0,
                    help="shared mode only: a position counts as common if at "
                         "least this many arms fire there (0 = all arms). "
                         "Relaxing it raises retention, which E19 shows is the "
                         "moderator of the position-crossing gain.")
    ap.add_argument("--pos_mode", default="per_arm",
                    choices=("per_arm", "shared", "union"),
                    help="per_arm: each arm uses its own top-n positions "
                         "(published practice; position nested within arm). "
                         "shared: one arm-symmetric position set per latent, "
                         "measured in every arm (fully crossed).")
    a = ap.parse_args()
    cfg = model_configs.get(a.base)
    arm_dir = a.arm_dir or cfg["dir"]

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16).to(DEV).eval()
    arms = load_arms(arm_dir, cfg)
    print(f"arms: {list(arms)}  (k = "
          f"{ {t: m.k for t, m in arms.items()} })", flush=True)
    if len(arms) < 3:
        raise SystemExit(f"need at least 3 arms in {arm_dir}; "
                         f"run src/chain6.sh (or the --base {a.base} "
                         f"equivalent) first")

    txt = [str(t).strip() for t in pd.read_parquet(WIKI)["text"].tolist()
           if len(str(t).strip()) > 400 and not str(t).strip().startswith("=")]
    seqs, buf = [], ""
    for t in txt:
        buf += " " + t
        i = tok(buf, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if i.shape[1] >= cfg["seq"]:
            seqs.append(i[:, :cfg["seq"]])
            buf = ""
            if len(seqs) >= a.n_seq:
                break
    ids_all = torch.cat(seqs, 0)
    S, T = ids_all.shape
    R = []
    with torch.no_grad():
        for i in range(0, S, 8):
            R.append(model(input_ids=ids_all[i:i + 8].to(DEV),
                           output_hidden_states=True)
                     .hidden_states[cfg["layer"] + 1][:, 1:, :].float().cpu())
    R = torch.cat(R)
    print(f"resid {tuple(R.shape)}", flush=True)

    live = None
    for tag, m in arms.items():
        cnt = torch.zeros(cfg["width"], device=DEV)
        with torch.no_grad():
            for i in range(0, S, 8):
                A = m.encode(R[i:i + 8].to(DEV).reshape(-1, cfg["d_model"]))
                cnt += (A > 0).float().sum(0)
                del A
        al = set(np.where((cnt / (S * (T - 1))).cpu().numpy() > 1e-5)[0].tolist())
        live = al if live is None else (live & al)
        print(f"  {tag}: {len(al)} live", flush=True)
    live = np.array(sorted(live))
    # Draw from the FULL id space under a fixed seed, then intersect with live
    # -- not `choice(live)`. Sampling from the live set makes the drawn ids a
    # function of the live-set intersection, so a rerun whose intersection
    # shifts by one latent draws an almost entirely different sample: the first
    # retrain shared only 8 of 240 latents with the original run, which made
    # the two incomparable at the latent level and turned a reproducibility
    # test into a between-sample comparison. Drawing from the fixed id space
    # makes the sample stable across runs and across arm sets.
    order = np.random.default_rng(0).permutation(cfg["width"])
    live_set = set(live.tolist())
    fids = np.sort(np.array([i for i in order if i in live_set][:a.n_feat]))
    print(f"shared intersection {len(live)}; sampled {len(fids)} "
          f"(stable draw from the fixed id space)", flush=True)

    sel = torch.tensor(fids, dtype=torch.long, device=DEV)

    # Gated activations for every arm, kept on CPU: needed up front because the
    # shared position set is a function of ALL arms at once.
    ACTS = {}
    for tag, m in arms.items():
        chunks = []
        with torch.no_grad():
            for i in range(0, S, 8):
                chunks.append(gated(m, R[i:i + 8].to(DEV)
                                    .reshape(-1, cfg["d_model"]), sel).cpu())
        ACTS[tag] = torch.cat(chunks).reshape(S, T - 1, len(fids))
        print(f"  {tag} activations computed", flush=True)
        torch.cuda.empty_cache()

    # Arm-symmetric position choice: rank candidates by the MINIMUM activation
    # across arms, so every selected position is one where the latent fires in
    # every arm, and no arm serves as the reference.
    shared_pos = {}
    if a.pos_mode == "shared":
        J = len(arms)
        k = J if a.min_arms <= 0 else min(a.min_arms, J)
        if k == J:
            # Incremental elementwise minimum. torch.stack materialised a
            # SECOND full copy of every arm's activations (arms x S x T-1 x F)
            # purely to take a min over the first axis. At n_seq 1536 on
            # gemma2-2b that copy is ~4.5 GB on top of a ~4.5 GB ACTS and a
            # ~7.2 GB residual, which pushed the process into swap: it ran
            # 3h38m without emitting a line, at 0% GPU and two cores busy,
            # before being killed. The peak, not the steady state, is what
            # breaks -- once pages are evicted the measurement loop faults on
            # every access. Same result, one buffer.
            mn = None
            for t in arms:
                mn = ACTS[t].clone() if mn is None else torch.minimum(mn, ACTS[t])
        else:
            stacked = torch.stack([ACTS[t] for t in arms])   # arms x S x T-1 x F
            # k-th largest across arms: a position qualifies when at least k
            # arms fire there. CAVEAT for k < J: the arms that do NOT fire are
            # still measured at that position and contribute kl ~ 0, which is a
            # real observation ("this latent has no effect here under this
            # fit") but sits at the log floor and will dominate a log-scale
            # variance decomposition. Downstream analysis must decide whether
            # to floor or drop those; the primary analysis uses k = J.
            mn = stacked.sort(dim=0, descending=True).values[k - 1]
            del stacked
        for j, fid in enumerate(fids):
            col = mn[:, :, j].reshape(-1)
            n = min(a.n_pos, int((col > 0).sum()))
            if n == 0:
                continue
            top = torch.topk(col, n)
            shared_pos[int(fid)] = [(int(ix) // (T - 1), int(ix) % (T - 1) + 1)
                                    for ix in top.indices]

    elif a.pos_mode == "union":
        # The minimum-gated rule above ranks EVERY position by its weakest arm,
        # which pushes selection toward tokens where all six arms fire hard --
        # and therefore AWAY from the tokens they most disagree about, which are
        # the tokens the whole paper is about. The controlled estimate can then
        # be unbiased between arms while being unrepresentative of the latent,
        # and part of the interaction collapse could be that selection rather
        # than the position control. This mode tests it.
        #
        # Candidates are the UNION of each arm's own top-n positions, so every
        # arm's argmax is eligible however much the others dislike it. The
        # all-arms-fire requirement is kept, or the design stops being crossed.
        # Ranking within the union is by MEAN across arms: max would privilege
        # whichever arm fires hardest and reintroduce an arm as reference, min
        # would rebuild the rule being tested.
        mn, mean = None, None
        for t in arms:
            mn = ACTS[t].clone() if mn is None else torch.minimum(mn, ACTS[t])
            mean = ACTS[t].clone() if mean is None else mean + ACTS[t]
        mean = mean / len(arms)
        n_cand = []
        for j, fid in enumerate(fids):
            cand = set()
            for t in arms:
                col_t = ACTS[t][:, :, j].reshape(-1)
                nt = min(a.n_pos, int((col_t > 0).sum()))
                if nt:
                    cand.update(int(i) for i in torch.topk(col_t, nt).indices)
            colmin = mn[:, :, j].reshape(-1)
            cand = [c for c in cand if float(colmin[c]) > 0]
            if not cand:
                continue
            n_cand.append(len(cand))
            colmean = mean[:, :, j].reshape(-1)
            cand.sort(key=lambda c: -float(colmean[c]))
            shared_pos[int(fid)] = [(c // (T - 1), c % (T - 1) + 1)
                                    for c in cand[:a.n_pos]]
        del mn, mean
        cov = 100.0 * len(shared_pos) / len(fids)
        print(f"  union position sets for {len(shared_pos)}/{len(fids)} latents "
              f"({cov:.1f}%); median {np.median(n_cand) if n_cand else 0:.0f} "
              f"eligible candidates per latent", flush=True)

    if a.pos_mode == "shared":
        cov = 100.0 * len(shared_pos) / len(fids)
        print(f"  shared position sets for {len(shared_pos)}/{len(fids)} latents "
              f"({cov:.1f}%); latents firing in all arms at >=1 common position",
              flush=True)

    import time
    rows = []
    for tag, m in arms.items():
        ACT = ACTS[tag]
        t0 = time.time()
        for j, fid in enumerate(fids):
            # Heartbeat. A stall inside this loop is otherwise invisible until
            # the arm finishes: the gemma2 shared run at n_seq 1536 sat here for
            # 3h38m emitting nothing, and a log whose last line is old looks
            # exactly like one that is still working. Progress is judged on the
            # timestamp of the most recent line, so there has to be one.
            if j and j % 50 == 0:
                print(f"    {tag}: {j}/{len(fids)} latents, "
                      f"{time.time() - t0:.0f}s elapsed", flush=True)
            if a.pos_mode in ("shared", "union"):
                if int(fid) not in shared_pos:
                    continue
                # same (sequence, position) pairs in every arm; the activation
                # is this arm's own value there
                best = [(float(ACT[s_i, p - 1, j]), s_i, p)
                        for s_i, p in shared_pos[int(fid)]]
            else:
                col = ACT[:, :, j].reshape(-1)
                n = min(a.n_pos, int((col > 0).sum()))
                if n == 0:
                    continue
                top = torch.topk(col, n)
                best = [(float(v), int(ix) // (T - 1), int(ix) % (T - 1) + 1)
                        for v, ix in zip(top.values, top.indices)]
            dvec = m.W_dec.data[int(fid)]
            ib = torch.stack([ids_all[i] for _, i, _ in best])
            kls = kl_batch(model, ib, [p for _, _, p in best],
                           [v * dvec for v, _, _ in best])
            for kl, (v, s_i, p) in zip(kls, best):
                pn = float((v * dvec).norm().item())
                rows.append(dict(arm=tag, fid=int(fid), seq=s_i, pos=p,
                                 rel_pos=p / (T - 1), act=v, kl=kl, pnorm=pn,
                                 kl_per_norm=kl / max(pn, 1e-6)))
        print(f"  {tag} measured, rows={len(rows)}", flush=True)
        torch.cuda.empty_cache()
    del ACTS

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    ov = df.groupby("fid").arm.nunique()
    print(f"wrote {a.out} rows={len(df)}  latents in all "
          f"{len(arms)} arms={int((ov == len(arms)).sum())}")


if __name__ == "__main__":
    main()
