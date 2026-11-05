#!/bin/bash
# Is a timing-tolerant bowler possible near human pace? (two_week_plan.md, 5 Nov)
# As kin_bowl_torque_b4096_t12_*, but a good delivery pays in proportion to min(speed / 68 km/h, 1)^k.
# hpc/submit_speedw.sh raised the speed weight instead, which hardly changed the trade-off: 70 km/h against
# 50 km/h was worth 1.35x at weight 0.5, 1.39x at weight 3. Here it is about 3.4x at k=4 and 11.7x at k=8.
# Then scores every run with hpc/kin_eval.pbs once they all end.
cd ~/cricket-project || exit 1
git fetch origin main || exit 1
git merge --ff-only origin/main || exit 1
grep -q 'SPEEDP' hpc/train_runup.pbs || exit 1
ids=""; runs=""
for k in 4 8; do
    for ang in 250 270 290; do
        for seed in 0 1; do
            id=$(qsub -v DELIVERY=T3a,SEED=$seed,TASK=kinematic,ANGLE=$ang,ARMCTL=torque,STEPS=30000000,BATCH=4096,TIMING=12,SPEEDP=$k hpc/train_runup.pbs) || exit 1
            ids="$ids:$id"
            runs="$runs kin_bowl_torque_b4096_t12_p${k}_a${ang}_T3a_${seed}"
            echo "$id k=$k a=$ang s=$seed"
        done
    done
done
qsub -W depend=afterany$ids -v RUNS="${runs# }" hpc/kin_eval.pbs
