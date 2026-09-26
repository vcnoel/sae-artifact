P="${PYTHON:-python}"
cd "$(dirname "$0")/.."
# gate on the checkpoint series finishing; no second process on the GPU
while ! grep -q "CURVE DONE" logs/chain4.log 2>/dev/null; do
  if ! pgrep -f "src/" >/dev/null; then echo "### upstream died"; tail -4 logs/chain4.log; exit 1; fi
  sleep 60
done
echo "### E10 FOUR-DEPTH TEST  $(date +%H:%M:%S)"
$P -u src/multi_dict.py --n_seq 64 --n_feat 300 --n_pos 4 \
   --grid "5:16k:68,12:16k:82,19:16k:73,24:16k:73" \
   --out results/depth4.csv
echo "### E10 DONE $(date +%H:%M:%S)"
