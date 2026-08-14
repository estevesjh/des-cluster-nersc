#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --account=des
#SBATCH --job-name=buzz_small_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00
#SBATCH --output=buzzard_polychord_small_debug.log
#SBATCH --error=buzzard_polychord_small_debug.error
#
# SMALL-SCALE shear split, DEBUG smoke test.
# Fits NC (12) + shear only on R in [0.2, 1.2] Mpc/h (radii 0-5 -> 72), via the
# likelihood_cp.py radial scale-cut (shear_r_min/shear_r_max). Same forward
# model as mock_mcmc_cp_camb.ini + the h-fixed DV; only the shear scale range
# differs from the large-scale run. 1-halo dominated; diagnostic for whether the
# 2-halo term drives the fit bias.
#
# Debug (30 min, live_points=100): confirms the mask works under MPI + srun and
# checkpoints write. Will NOT converge -- use buzzard_polychord_small.sh for that.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}

BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_small/polychord
mkdir -p ${BUZZ_DIR}/clusters

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord polychord.resume=F polychord.live_points=100 \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_jkcov_hfix.npz \
        likelihoods.shear_r_min=0.2 likelihoods.shear_r_max=1.2 \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_small/chain.txt
