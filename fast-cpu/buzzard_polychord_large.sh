#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=buzz_large
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=09:00:00
#SBATCH --output=buzzard_polychord_large.log
#SBATCH --error=buzzard_polychord_large.error
#
# LARGE-SCALE shear split, PRODUCTION.
# NC (12) + shear on R in [1.2, 5.0] Mpc/h (radii 6-9 -> 48) via the
# likelihood_cp.py radial scale-cut. Same forward model + h-fixed DV as the
# small-scale run; only the shear scale range differs. 2-halo contributes here.
# Frozen-physics speed -> ~3h expected inside the 9h cap; resume with
# polychord.resume=T if it walls.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}

BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_large/polychord
mkdir -p ${BUZZ_DIR}/clusters

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord polychord.resume=F \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_jkcov_hfix.npz \
        likelihoods.shear_r_min=1.2 likelihoods.shear_r_max=5.0 \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_large/chain.txt
