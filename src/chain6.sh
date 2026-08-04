P=/c/Users/valno/anaconda3/envs/gemma_spectral/python.exe
cd /c/Users/valno/Dev/sae-artifacts
run(){ echo "### $1  $(date +%H:%M:%S)"; $P -u src/train_arms.py --tag "$@" ; }
run free                       --tokens 12000000
run tau080  --tau 0.80         --tokens 12000000
run tau090  --tau 0.90         --tokens 12000000
run lr1e4   --lr 1e-4          --tokens 12000000
run k41     --k 41             --tokens 12000000
run order1  --order 1          --tokens 12000000
echo "### SIX ARMS DONE $(date +%H:%M:%S)"
