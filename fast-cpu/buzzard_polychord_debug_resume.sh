#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --account=des
#SBATCH --job-name=buzzard_polychord_dbg_resume
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00
#SBATCH --output=buzzard_polychord_debug_resume.log
#SBATCH --error=buzzard_polychord_debug_resume.error
#
# RESUME the Buzzard production polychord chain on the DEBUG queue (30 min).
#
# The production run (job 56546840) hit the 9h wall before converging. This
# continues it from the checkpoint (buzzard_polychord.resume, live_points=500)
# for one more 30-min debug slice. It will NOT converge in 30 min but advances
# the chain and confirms the resume path works; keep re-submitting (this script
# or the 9h buzzard_polychord.sh with resume=T) until log(Z) leaves "Still
# Active".
#
# CRITICAL: resume=T with the SAME base_dir and the ini-default live_points=500
# (do NOT override to 100 like buzzard_polychord_debug.sh — nlive must match the
# checkpoint). Do NOT clear the base_dir: the .resume file must survive.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}

BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard/polychord
mkdir -p ${BUZZ_DIR}/clusters   # checkpoint must already exist; do NOT clear

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord polychord.resume=T \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_jkcov.npz \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard/chain.txt
