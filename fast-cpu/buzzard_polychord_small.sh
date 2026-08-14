#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=buzz_small
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=09:00:00
#SBATCH --output=buzzard_polychord_small.log
#SBATCH --error=buzzard_polychord_small.error
#
# SMALL-SCALE shear split, PRODUCTION.
# NC (12) + shear on R in [0.2, 1.2] Mpc/h (radii 0-5 -> 72) via the
# likelihood_cp.py radial scale-cut. Same forward model + h-fixed DV as the
# large-scale run; only the shear scale range differs. Compare the recovered
# cosmology of small vs large to localize whether the 2-halo term drives the
# fit bias. Frozen-physics speed -> ~3h expected inside the 9h cap; resume with
# polychord.resume=T if it walls.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}

BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_small/polychord
mkdir -p ${BUZZ_DIR}/clusters

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord polychord.resume=F \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_jkcov_hfix.npz \
        likelihoods.shear_r_min=0.2 likelihoods.shear_r_max=1.2 \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_small/chain.txt
