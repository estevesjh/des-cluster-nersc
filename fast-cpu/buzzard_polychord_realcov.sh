#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=buzz_realcov
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=09:00:00
#SBATCH --output=buzzard_polychord_realcov.log
#SBATCH --error=buzzard_polychord_realcov.error
#
# Buzzard full-shear fit with REALISTIC covariance, PRODUCTION.
# DV = dv_buzzard_realcov.npz: h-fixed Buzzard data + Matteo/Costanzi DES-Y1 NC
# covariance (top-left 12x12 of Cov_ij_bestfit_DESY1_105) + DES-Y1-WL shear
# diagonal errors (wl_cov.txt diagonal, interpolated to the 10 pipeline radii).
# At fiducial chi2 ~782/132 (vs ~19700 with the tight JK cov), so realistic
# errors should un-rail the posterior. Frozen-physics speed -> ~3h inside the
# 9h cap; resume with polychord.resume=T if it walls.

# z-EDGE FIX (issue #2): drives mock_mcmc_cp_camb_buzzard.ini, which
# overrides the observed-z bin edges to the Buzzard mock binning
# [0.20,0.33) [0.37,0.50) [0.50,0.65) (box seam excluded). Outputs go to
# fresh *_zfix dirs so the pre-fix converged chains stay comparable.
set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}

BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_realcov_zfix/polychord
mkdir -p ${BUZZ_DIR}/clusters

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb_buzzard.ini \
     -p runtime.sampler=polychord polychord.resume=F \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_realcov.npz \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_realcov_zfix/chain.txt
