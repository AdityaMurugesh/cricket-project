#!/bin/bash
# Run ON THE HPC login node (hpc/submit_g1.ps1 uploads and runs it): update the code, check the G1 model works
# with the HPC's mujoco, a 2-generation CMA-ES smoke run, then submit pack g1_1.
#   bash ~/cricket-project/hpc/g1_remote_setup.sh          (add "nosubmit" to stop before qsub)
set -e
cd ~/cricket-project
git fetch -q origin && git merge --ff-only origin/main || { echo "merge blocked (local changes on the HPC?): git status, stash, rerun"; git status --short | head; exit 1; }
test -f data/mocap/ref_G1_T3aL.npz || { echo "data/mocap/ref_G1_T3aL.npz missing: scp it first"; exit 1; }
module load python/3.14.0-gcc-12.5.0-4km4ttx
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python - <<'EOF'
import mujoco, numpy as np, time
print("mujoco", mujoco.__version__)
from envs.g1_bowl_env import G1BowlEnv
e = G1BowlEnv(delivery="T3aL")
o, _ = e.reset(seed=0)
t = time.time(); n = 0; done = False
while not done:
    o, r, term, trunc, info = e.step(np.zeros(e.action_space.shape)); n += 1; done = term or trunc
print("G1 env ok: obs", o.shape, "steps", n, "reward", round(float(r), 3), f"{(time.time() - t) / n * 1e3:.1f} ms/step")
EOF
mkdir -p logs /tmp/g1_smoke
python scripts/kin_frontier.py --robot g1 --delivery T3aL --angle 270 --handover 0.5 --knot-dt 0.02 --diagonal \
    --sigma 0.3 --gens 2 --popsize 8 --p-cont --drag --stiff-elbow --jitter-sd-ms 4 --tau 0 --warm-id --workers 2 \
    --out /tmp/g1_smoke 2>&1 | grep -v Warning | tail -5
echo "smoke ok"
[ "$1" = "nosubmit" ] && exit 0
qsub -l select=1:ncpus=32:mem=60gb -l walltime=06:00:00 -v CMDS=hpc/packs/g1_1.txt,P=6 hpc/cmds.pbs
qstat -u "$USER" | tail -3
