#!/bin/bash
P=/home/saksham/experiments/franka_cube_lift/.env/bin/python
OUT=../results/sweep.tsv
mkdir -p ../results
echo -e "demos\tepochs\tseed\tval_mse\tsuccess" > $OUT
for cfg in "state_demos 340" "clean1000 100" "clean1000 340" "clean3000 100"; do
  set -- $cfg; D=$1; E=$2
  for S in 0 1 2; do
    R=$($P train_bc.py --demos ../demos/$D.npz --epochs $E --seed $S \
        --eval-episodes 50 --out /tmp/bc_sweep.pt 2>&1)
    V=$(echo "$R" | grep -oP "epoch\s+$E\s+train\s+\S+\s+val\s+\K\S+")
    A=$(echo "$R" | grep -oP "BC policy success: \K[0-9.]+")
    echo -e "$D\t$E\t$S\t$V\t$A" | tee -a $OUT
  done
done
echo "SWEEP DONE"
