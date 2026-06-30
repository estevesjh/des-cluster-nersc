#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=mock_emcee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --time=09:00:00
#SBATCH --output=mock_emcee.log
#SBATCH --error=mock_emcee.error
#SBATCH --signal=USR1@600
#
# Production widePlanck mock-MCMC chain.
#
# Layout: 1 node, half-node shared-QOS allocation (cpus-per-task=128 =
# 64 physical Milan cores = the maximum a shared-QOS job can request;
# requesting 256 logical hits the "More resources requested than allowed
# for logical queue" rejection per ~/.claude/CLAUDE.md NERSC rules).
#
# Sampler:
#   emcee, walkers = 64 (>= 3*nparam+1 = 31 for nparam=10), samples = 10000
#   resume = T (the [runtime] block) so a re-submit after wall-time
#   timeout picks up where it left off.
#   --smp=64 inside the srun fans 64 walker pipelines across the 64
#   physical cores; each walker gets 2 logical cores so internal
#   numpy/scipy can use OMP_NUM_THREADS=2 if needed.
#
# CPU job (constraint=cpu, no --gpus); the C++ .so files dlopen the
# cuda stub on cpu nodes without invoking GPU.
#
# --signal=USR1@600 sends SIGUSR1 to cosmosis 10 minutes before wall
# time elapses so it can flush the chain to disk and exit cleanly.
# Re-submit the same script (resume=T) to extend.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}
mkdir -p /pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/emcee

# Override the [output] filename so emcee does NOT clobber the polychord
# chain.txt (both samplers default to .../mock_mcmc/chain.txt). The Jun-2026
# emcee run was lost this way when polychord overwrote the shared file.
EMCEE_CHAIN=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/emcee/chain.txt

# Don't let a non-zero exit (e.g. SIGUSR1 wall-time flush) skip the sigma_8
# post-processing — we still want it on whatever samples were flushed.
# resume=T: EXTEND the existing JK-cov emcee chain (6400 steps, not yet
# converged at tau~300). Appends toward the samples=10000 target; re-submit to
# extend further. The chain was confirmed to be the jkcov run before resuming.
srun -n 1 cosmosis --smp=64 cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=emcee runtime.resume=T output.filename=${EMCEE_CHAIN} || true

# Append the derived sigma_8 column (emulator P(k) -> cluster_toolkit
# sigma2_at_R). Cheap, serial; runs after the chain is flushed. Re-runs on
# resume are harmless (regenerates from the latest chain).
python validations/add_sigma8_to_chain.py \
     ${EMCEE_CHAIN} \
     /pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/emcee/chain_sigma8.txt
