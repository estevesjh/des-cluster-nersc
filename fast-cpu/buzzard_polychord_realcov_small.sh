#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=buzz_rc_small
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=09:00:00
#SBATCH --output=buzzard_polychord_realcov_small.log
#SBATCH --error=buzzard_polychord_realcov_small.error
#
# small-scale Buzzard fit with REALISTIC covariance.
# DV = dv_buzzard_realcov.npz (Matteo Y1 NC cov + Y1-WL shear diagonal), shear
# restricted to R in [0.2, 1.2] Mpc/h via likelihood_cp.py shear_r_min/max.
# Companion to the full realcov run; compare small vs large cosmology recovery
# under realistic errors (supersedes the tight-cov split).
# BUZZARD OVERRIDE INI (issues #1/#2 + y3_cluster_cpp#12): drives
# mock_mcmc_cp_camb_buzzard.ini (dense nz=400 distances grid; observed-z
# edges deliberately unchanged -- the DV harvester bins z_obs at
# 0.20/0.35/0.50/0.65 and the seam is a TRUE-z cut, see the ini header).
# Outputs go to fresh *_zfix dirs so prior converged chains stay comparable.
set -euo pipefail
cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh
which cosmosis; python --version
cd ${DES_CLUSTER_NERSC_DIR}
BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_realcov_small_zfix/polychord
mkdir -p ${BUZZ_DIR}/clusters
srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb_buzzard.ini \
     -p runtime.sampler=polychord polychord.resume=F \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_realcov.npz \
        likelihoods.shear_r_min=0.2 likelihoods.shear_r_max=1.2 \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_realcov_small_zfix/chain.txt
