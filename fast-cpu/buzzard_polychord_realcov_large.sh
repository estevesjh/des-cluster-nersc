#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=buzz_rc_large
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=09:00:00
#SBATCH --output=buzzard_polychord_realcov_large.log
#SBATCH --error=buzzard_polychord_realcov_large.error
#
# large-scale Buzzard fit with REALISTIC covariance.
# DV = dv_buzzard_realcov.npz (Matteo Y1 NC cov + Y1-WL shear diagonal), shear
# restricted to R in [1.2, 5.0] Mpc/h via likelihood_cp.py shear_r_min/max.
# Companion to the full realcov run; compare small vs large cosmology recovery
# under realistic errors (supersedes the tight-cov split).
# *** HOLD (des-nersc#3): do NOT submit until the DV is rebuilt from
# xtang126's current npz -- the dv_buzzard_* files this script loads are
# stale (NC x1.186, harvester 0.35 z-edges, unresolved radius/Sigma_crit
# conventions), while the model config below uses the CORRECT
# seam-excluding edges: running the pair as-is is inconsistent. ***
# BUZZARD OVERRIDE INI (issues #1/#2 + y3_cluster_cpp#12): drives
# mock_mcmc_cp_camb_buzzard.ini -- observed-z edges [0.20,0.33) [0.37,0.50)
# [0.50,0.65) per the xtang126 MockDataVector.ipynb code (cell 3; the
# notebook header and this repo's build_buzzard_datavector.py 0.35 edges
# are both wrong/non-production) + dense nz=400 distances grid. Outputs
# go to fresh *_zfix dirs so prior converged chains stay comparable.
set -euo pipefail
cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh
which cosmosis; python --version
cd ${DES_CLUSTER_NERSC_DIR}
BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_realcov_large_zfix/polychord
mkdir -p ${BUZZ_DIR}/clusters
srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb_buzzard.ini \
     -p runtime.sampler=polychord polychord.resume=F \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_realcov.npz \
        likelihoods.shear_r_min=1.2 likelihoods.shear_r_max=5.0 \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_realcov_large_zfix/chain.txt
