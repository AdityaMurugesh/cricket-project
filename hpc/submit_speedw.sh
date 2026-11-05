#!/bin/bash
# Does more speed reward give a timing-tolerant bowler nearer human pace? (two_week_plan.md, 5 Nov)
# BATCH=4096 + TIMING=12 as kin_bowl_torque_b4096_t12_*, speed weight 1.5 and 3 instead of 0.5.
# TASKW rescales the whole reward so a good delivery is still worth ~85, as at weight 0.5 / task 10
# (scaling does not change the optimum, and keeps the value loss where it was).
# Then scores every run with hpc/kin_eval.pbs once they all end.
cd ~/cricket-project || exit 1
git fetch origin main || exit 1
git merge --ff-only origin/main || exit 1
grep -q 'SPEEDW' hpc/train_runup.pbs || exit 1
ids=""; runs=""
for pair in 1.5:4 3:2; do
    w=${pair%:*}; t=${pair#*:}
    for ang in 250 270 290; do
        for seed in 0 1; do
            id=$(qsub -v DELIVERY=T3a,SEED=$seed,TASK=kinematic,ANGLE=$ang,ARMCTL=torque,STEPS=30000000,BATCH=4096,TIMING=12,SPEEDW=$w,TASKW=$t hpc/train_runup.pbs) || exit 1
            ids="$ids:$id"
            runs="$runs kin_bowl_torque_b4096_t12_w${w/./p}_a${ang}_T3a_${seed}"
            echo "$id w=$w a=$ang s=$seed"
        done
    done
done
qsub -W depend=afterany$ids -v RUNS="${runs# }" hpc/kin_eval.pbs
