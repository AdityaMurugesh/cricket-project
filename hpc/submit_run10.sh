#!/bin/bash
# Submit the run10 grid: release angle x actuator budget x seed, plus an
# ent_coef=0.01 control column at the nominal budget. Run from ~/cricket-project.
#   bash hpc/submit_run10.sh            # submit
#   DRY=1 bash hpc/submit_run10.sh      # print the qsub lines only
set -euo pipefail

ANGLES="${ANGLES:-230 240 250 260 265 270 275 280 285 290 300}"
SCALES="${SCALES:-0.75 1.0 1.5}"
SEEDS="${SEEDS:-0 1}"
CONTROL_ENT="${CONTROL_ENT:-0.01}"

submit() {  # ang scale seed ent
    local s100 e tag
    s100=$(awk -v s="$2" 'BEGIN { printf "%03d", s * 100 }')
    e=$(echo "$4" | tr -d '.')
    tag="a$1_s${s100}_e${e}_${3}"
    local cmd="qsub -v ANG=$1,SCALE=$2,SEED=$3,ENT=$4,TAG=${tag}${STEPS:+,STEPS=$STEPS} hpc/train_throw_run10.pbs"
    if [ -n "${DRY:-}" ]; then echo "$cmd"; else echo "${tag}: $($cmd)"; fi
}

for scale in $SCALES; do
    for ang in $ANGLES; do
        for seed in $SEEDS; do
            submit "$ang" "$scale" "$seed" 0.003
        done
    done
done
for ang in $ANGLES; do
    submit "$ang" 1.0 0 "$CONTROL_ENT"
done
