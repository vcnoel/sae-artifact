P=/c/Users/valno/anaconda3/envs/gemma_spectral/python.exe
cd /c/Users/valno/Dev/sae-artifacts
echo "waiting for chain6_g3 to finish..."
while ! grep -q "SIX ARMS DONE" logs/chain6_g3.log 2>/dev/null; do
    if ! ps aux | grep -q "[t]rain_arms.py --base gemma3-1b"; then
        echo "### upstream train_arms died without SIX ARMS DONE marker"
        tail -50 logs/chain6_g3.log > logs/chain6_g3_died.log
        exit 1
    fi
    sleep 30
done
echo "### training done, starting eval $(date +%H:%M:%S)"
$P -u src/eval_arms.py --base gemma3-1b --n_seq 96 --n_feat 240 --n_pos 6 \
    --out results/eval_arms_g3.csv
echo "### eval done, starting decomposition $(date +%H:%M:%S)"
$P -u src/variance_decomp.py --crossed results/eval_arms_g3.csv
echo "### EVAL+DECOMP DONE (gemma3-1b) $(date +%H:%M:%S)"
