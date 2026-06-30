#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=mock_emcee_2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --time=02:00:00
#SBATCH --output=mock_emcee_2h.log
#SBATCH --error=mock_emcee_2h.error
#SBATCH --signal=USR1@600
#
# 2h shared-QOS shake-down of the emcee production pipeline.
# Same setup as mock_emcee.sh except wall is 2h (instead of 9h).
# Useful for confirming whether the shared queue is still under the
# "Reserved for maintenance" reservation that blocked the earlier
# 9h job.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}
mkdir -p /pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/emcee

srun -n 1 cosmosis --smp=64 cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=emcee
