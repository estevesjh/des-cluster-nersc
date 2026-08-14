#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --account=des
#SBATCH --job-name=buzz_realcov_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00
#SBATCH --output=buzzard_polychord_realcov_debug.log
#SBATCH --error=buzzard_polychord_realcov_debug.error
#
# Buzzard full-shear fit with REALISTIC covariance, DEBUG smoke test.
# DV = dv_buzzard_realcov.npz: h-fixed Buzzard data + Matteo/Costanzi DES-Y1 NC
# covariance + DES-Y1-WL shear diagonal errors (built by
# validations/build_buzzard_dv_realcov.py). Tests whether realistic (not tight
# JK) errors un-rail the posterior and recover cosmology. Full 120-vector (no
# scale cut). Debug (30 min) confirms the dense covs load + sample under MPI.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}

BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_realcov/polychord
mkdir -p ${BUZZ_DIR}/clusters

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord polychord.resume=F polychord.live_points=100 \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_realcov.npz \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_realcov/chain.txt
