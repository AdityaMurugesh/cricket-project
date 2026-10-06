#!/bin/bash
# Submit run10 as packed jobs (see train_throw_run10_pack.pbs for why):
# one per (budget, seed) at ent_coef=0.003, plus one ent_coef=0.01 control
# at the nominal budget, seed 0. Run from ~/cricket-project.
#
# Safe to re-run: each job only gets the angles that have no eval.json yet,
# and a (budget, seed, ent) with nothing left is skipped. Make sure no
# earlier run10 jobs are still queued/running first, or they will be doubled.
#   DRY=1 bash hpc/submit_run10_pack.sh   # print the qsub lines only
set -euo pipefail

ANGLES="${ANGLES:-230 240 250 260 265 270 275 280 285 290 300}"
SCALES="${SCALES:-0.75 1.0 1.5}"
SEEDS="${SEEDS:-0 1}"

submit() {  # scale seed ent
    local s100 e todo=""
    s100=$(awk -v s="$1" 'BEGIN { printf "%03d", s * 100 }')
    e=$(echo "$3" | tr -d '.')
    for ang in $ANGLES; do
        [ -f "logs/throw_ppo_run10_a${ang}_s${s100}_e${e}_${2}/eval.json" ] || todo="${todo:+$todo:}$ang"
    done
    if [ -z "$todo" ]; then
        echo "scale=$1 seed=$2 ent=$3: all angles done, skipping"
        return
    fi
    local cmd="qsub -v SCALE=$1,SEED=$2,ENT=$3,ANGLES=${todo}${STEPS:+,STEPS=$STEPS} hpc/train_throw_run10_pack.pbs"
    if [ -n "${DRY:-}" ]; then echo "$cmd"; else echo "scale=$1 seed=$2 ent=$3 angles=${todo}: $($cmd)"; fi
}

for scale in $SCALES; do
    for seed in $SEEDS; do
        submit "$scale" "$seed" 0.003
    done
done
submit 1.0 0 0.01
