#!/bin/bash
# PPO from scratch with the arm handed over 500 ms before release (KinematicBowlEnv.arm_handover_s), the fair
# baseline for the trajectory-optimised frontier (scripts/kin_frontier.py --handover 0.5) and for PPO started
# from cloned optimised deliveries. b4096 + t12 at three angles x 2 seeds, and noise-free b4096 at seed 0.
# Then scores every run with hpc/kin_eval.pbs once they all end.
cd ~/cricket-project || exit 1
git fetch origin main || exit 1
git merge --ff-only origin/main || exit 1
grep -q 'HANDOVER' hpc/train_runup.pbs || exit 1
ids=""; runs=""
for ang in 250 270 290; do
    for cfg in "12 0" "12 1" " 0"; do
        set -- $cfg
        if [ $# -eq 2 ]; then t=$1; seed=$2; else t=""; seed=$1; fi
        id=$(qsub -v DELIVERY=T3a,SEED=$seed,TASK=kinematic,ANGLE=$ang,ARMCTL=torque,STEPS=30000000,BATCH=4096,HANDOVER=500${t:+,TIMING=$t} hpc/train_runup.pbs) || exit 1
        ids="$ids:$id"
        runs="$runs kin_bowl_torque_b4096${t:+_t$t}_h500_a${ang}_T3a_${seed}"
        echo "$id a=$ang t=${t:-none} s=$seed"
    done
done
qsub -W depend=afterany$ids -v RUNS="${runs# }" hpc/kin_eval.pbs
