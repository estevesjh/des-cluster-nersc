#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=buzzard_polychord_hfix
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=09:00:00
#SBATCH --output=buzzard_polychord_hfix.log
#SBATCH --error=buzzard_polychord_hfix.error
#
# Buzzard polychord on the h-UNIT-FIXED data vector (dv_buzzard_jkcov_hfix.npz:
# shear DeltaSigma x h_buzz=0.7 to convert physical M_sun/pc^2 -> little-h
# h*M_sun/pc^2 to match the model; invcov_Shear / h^2; NC unchanged). STOPGAP
# until the mock is shipped in little-h units by xtang126.
#
# Fresh run (resume=F) in a SEPARATE base_dir (buzzard_hfix/) so the previous
# converged (unfixed) chain in buzzard/ is preserved for comparison. If h stops
# railing and Omega_m*h returns toward fiducial, the h-unit fix is confirmed.

# BUZZARD OVERRIDE INI (issues #1/#2 + y3_cluster_cpp#12): drives
# mock_mcmc_cp_camb_buzzard.ini (dense nz=400 distances grid; observed-z
# edges deliberately unchanged -- the DV harvester bins z_obs at
# 0.20/0.35/0.50/0.65 and the seam is a TRUE-z cut, see the ini header).
# Outputs go to fresh *_zfix dirs so prior converged chains stay comparable.
set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}

BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_hfix_zfix/polychord
mkdir -p ${BUZZ_DIR}/clusters

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb_buzzard.ini \
     -p runtime.sampler=polychord polychord.resume=F \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_jkcov_hfix.npz \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_hfix_zfix/chain.txt
