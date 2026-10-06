#!/bin/bash
# Submit run10 as 7 packed jobs (see train_throw_run10_pack.pbs for why):
# one per (budget, seed) at ent_coef=0.003, plus one ent_coef=0.01 control
# at the nominal budget, seed 0. Run from ~/cricket-project.
set -euo pipefail

ANGLES="${ANGLES:-230:240:250:260:265:270:275:280:285:290:300}"
SCALES="${SCALES:-0.75 1.0 1.5}"
SEEDS="${SEEDS:-0 1}"

for scale in $SCALES; do
    for seed in $SEEDS; do
        echo "scale=${scale} seed=${seed}: $(qsub -v SCALE=${scale},SEED=${seed},ENT=0.003,ANGLES=${ANGLES}${STEPS:+,STEPS=$STEPS} hpc/train_throw_run10_pack.pbs)"
    done
done
echo "control ent=0.01: $(qsub -v SCALE=1.0,SEED=0,ENT=0.01,ANGLES=${ANGLES}${STEPS:+,STEPS=$STEPS} hpc/train_throw_run10_pack.pbs)"
