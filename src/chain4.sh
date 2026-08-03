P=/c/Users/valno/anaconda3/envs/gemma_spectral/python.exe
cd /c/Users/valno/Dev/sae-artifacts
for ARM in trained frozen; do
  echo "### TRAIN $ARM with checkpoints  $(date +%H:%M:%S)"
  $P -u src/train_saes.py --tokens 12000000 --arm $ARM --ckpt_every 3000000 --out data/saes_$ARM.pt
done
echo "### MERGE per-milestone  $(date +%H:%M:%S)"
$P - <<'PY'
import torch, glob, os
for m in (3,6,9,12):
    d={}
    ok=True
    for arm in ("trained","frozen"):
        p=f"data/ck_{arm}_{m}M.pt"
        if not os.path.exists(p): ok=False; break
        d[arm]=torch.load(p,map_location="cpu")[arm]
    if ok:
        torch.save(d,f"data/ck_{m}M.pt"); print("merged",m,"M")
PY
for M in 3 6 9 12; do
  if [ -f data/ck_${M}M.pt ]; then
    echo "### EVAL @ ${M}M tokens  $(date +%H:%M:%S)"
    $P -u src/eval_saes.py --ckpt data/ck_${M}M.pt --n_seq 96 --per_bin 20 --n_pos 4 --uniform --out results/curve_${M}M.csv
  fi
done
echo "### CURVE DONE $(date +%H:%M:%S)"
