#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=buzzard_polychord
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=09:00:00
#SBATCH --output=buzzard_polychord.log
#SBATCH --error=buzzard_polychord.error
#
# Production polychord nested sampler on the BUZZARD data vector.
#
# Same forward model / ini as mock_polychord.sh, but the likelihood compares
# against the Buzzard sim data vector (data/mock/dv_buzzard_jkcov.npz, 10-radii
# built by validations/build_buzzard_dv_10r.py, mock JK covariance reused per
# data/README_FOR_ARWA.md). Output goes to a SEPARATE buzzard/ base_dir so it
# never touches the self-closure polychord results in .../mock_mcmc/polychord/.
#
# Layout: 1 node, 64 MPI ranks (one per physical core), 2 logical CPUs/rank.
# 64 physical cores = the shared-QOS 0.5-node cap.
#
# Throughput (calibrated 2026-08-09, debug job 56545431, live_points=100):
#   - 37 evals/sec across all 64 ranks; ~1.65 s/sample mean.
#   - At live_points=500, expect ~1M likelihood calls (cf. closure run
#     1.17M / 8h41m) -> ~9 hr wall. 9h cap; polychord checkpoints internally,
#     so a wall-time hit resumes by re-submitting with polychord.resume=T.
# NOTE: unlike the closure run, logL is NOT ~0 at fiducial -- the Buzzard
# recovery bias (model misspecification) is a real result, not an error.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}

BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard/polychord
mkdir -p ${BUZZ_DIR}/clusters

# resume=T: continue the Buzzard chain from its checkpoint (buzzard_polychord.resume).
# The first 9h slice (job 56546840) + a 30-min debug slice (56578937) reached
# ndead=16032 but did not converge; this resumes from there. Re-submit again if
# it walls before log(Z) leaves "Still Active".
srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord polychord.resume=T \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_jkcov.npz \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard/chain.txt
