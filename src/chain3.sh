P="${PYTHON:-python}"
cd "$(dirname "$0")/.."
echo "### E6a  Gemma Scope uniform vs untrained tied   $(date +%H:%M:%S)"
$P -u src/sae_rare.py --n_seq 96 --per_bin 40 --n_pos 6 --uniform --out results/sae_rare_uniform.csv
echo "### E6a DONE $(date +%H:%M:%S)  gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "### E6b+c  trained vs soft-frozen vs untrained, uniform, same list   $(date +%H:%M:%S)"
$P -u src/eval_saes.py --ckpt data/saes.pt --n_seq 96 --per_bin 40 --n_pos 6 --uniform --with_random --out results/eval_saes_uniform.csv
echo "### E6 ALL DONE $(date +%H:%M:%S)"
