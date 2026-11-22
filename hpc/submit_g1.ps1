# From the LAPTOP, with the FortiClient VPN connected: copy the G1 reference (gitignored participant data) to the HPC,
# then run hpc/g1_remote_setup.sh there (git pull, mujoco/G1 check, CMA smoke run, qsub of hpc/packs/g1_1.txt).
#   powershell -ExecutionPolicy Bypass -File hpc\submit_g1.ps1            (add -NoSubmit to stop before qsub)
param([switch]$NoSubmit)
$ErrorActionPreference = "Stop"
$key  = "C:\Users\ASUS\Downloads\HPC SSH\24a1cseb0002-private-key"
$hpc  = "24a1cseb0002@ad-hpc-ln.iitd-ad.local"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-NetConnection ad-hpc-ln.iitd-ad.local -Port 22 -InformationLevel Quiet)) {
    Write-Host "HPC not reachable: connect the FortiClient VPN first"; exit 1
}
# one source per scp call (multi-source scp silently dropped files before)
scp -i $key "$root\data\mocap\ref_G1_T3aL.npz" "${hpc}:cricket-project/data/mocap/ref_G1_T3aL.npz"
scp -i $key "$root\hpc\g1_remote_setup.sh" "${hpc}:g1_remote_setup.sh"
$arg = ""; if ($NoSubmit) { $arg = "nosubmit" }
ssh -i $key $hpc "bash ~/g1_remote_setup.sh $arg"
