#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --account=des
#SBATCH --job-name=mock_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00
# cpus-per-task=128 == 1 full Milan node (128 logical cores).  Inside,
# cosmosis --smp=64 fans 64 worker pipelines onto 64 physical cores;
# each worker gets 2 logical cores so internal numpy/scipy can use
# OMP_NUM_THREADS=2 if needed.  See mock_emcee_{a,b}.sh for the production
# equivalent (--cpus-per-task=2 --ntasks=64 on shared QOS).
#SBATCH --output=mock_test.log
#SBATCH --error=mock_test.error
#
# Apriori smoke + prior-coverage check: draws 128 samples from the
# prior box, evaluates the full pipeline + likelihood at each, no MCMC
# dynamics.  --smp=64 fans the 128 draws across 64 worker pipelines
# (2 chunks of 64 walkers in apriori's chunk-loop), each worker on 2
# logical cores so internal numpy/scipy can thread.
#
# The C++ .so modules link libcuda.so.1 at compile time but only use
# Cuba CPU integration paths at runtime; the stub libcuda on CPU nodes
# resolves dlopen.  "Parameters never used" warnings in the log are a
# CosmoSIS false positive (Python ini-tracking doesn't see C++
# DataBlock reads).

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}
mkdir -p /pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/apriori

srun -n 1 cosmosis --smp=128 cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=apriori
