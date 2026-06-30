#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --account=des
#SBATCH --job-name=mock_emcee_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00
#SBATCH --output=mock_emcee_debug.log
#SBATCH --error=mock_emcee_debug.error
#
# Debug-QOS shake-down of the emcee production pipeline.  Same setup as
# mock_emcee.sh (1 node, 128 logical cores = 64 physical, --smp=64,
# 64 walkers, resume=T) but capped at the 30-min debug wall, so the
# chain starts and writes a partial output that the production
# resubmit can extend.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}
mkdir -p /pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/emcee

srun -n 1 cosmosis --smp=64 cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=emcee
