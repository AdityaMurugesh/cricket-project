#!/bin/bash
# run15: train under motor noise. run13b's no-shaping setup (resolve at release),
# 4M steps, 1.0x and 1.5x x 230/240/250/260 deg x seeds 0-1, four conditions:
#   none    no noise (baseline with the same seeds and steps)
#   timing  release-timing jitter SD 12 ms
#   torque  signal-dependent torque noise SD 0.15
#   both    both
# 16 jobs; the per-user queue limit is 12, so this waits for free slots.
#   nohup bash hpc/submit_run15.sh > logs/submit_run15.log 2>&1 &
cd "$(dirname "$0")/.."
declare -A NOISE=(
    [none]=""
    [timing]="--release-timing-sd-ms 12"
    [torque]="--torque-noise-sd 0.15"
    [both]="--release-timing-sd-ms 12 --torque-noise-sd 0.15"
)
for cond in none timing torque both; do
    for sc in 1.5 1.0; do
        for seed in 0 1; do
            until id=$(qsub -v "SCALE=${sc},SEED=${seed},STEPS=4000000,VARIANT=run15_${cond},EXTRA=--no-shaping ${NOISE[$cond]}" \
                    hpc/train_throw_run13_pack.pbs 2>/dev/null); do
                sleep 60
            done
            echo "$(date +%T) ${cond} scale=${sc} seed=${seed} ${id}"
        done
    done
done
echo "all run15 jobs submitted"
