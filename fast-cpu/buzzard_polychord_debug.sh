#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --account=des
#SBATCH --job-name=buzzard_polychord_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00
#SBATCH --output=buzzard_polychord_debug.log
#SBATCH --error=buzzard_polychord_debug.error
#
# Debug-QOS shake-down of polychord on the BUZZARD data vector.
#
# Same forward model / ini as mock_polychord_debug.sh, but the likelihood is
# pointed at the Buzzard sim data vector (data/mock/dv_buzzard_jkcov.npz,
# 10-radii/120 built by validations/build_buzzard_dv_10r.py, reusing the mock
# JK covariance per data/README_FOR_ARWA.md). Unlike the self-closure run,
# logL is NOT ~0 at fiducial — model misspecification shows as recovery bias.
#
# Output is written to a SEPARATE buzzard/ base_dir so it never touches the
# self-closure polychord results in .../mock_mcmc/polychord/.
#
# Debug wall cap 30 min: polychord will not converge, but confirms the
# pipeline loads the Buzzard DV, all 64 ranks run, and checkpoints write.
# Resume/extend with a buzzard production script (copy of mock_polychord.sh
# with the same -p overrides and live_points=500).

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}

BUZZ_DIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard/polychord
mkdir -p ${BUZZ_DIR}/clusters

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord runtime.resume=F polychord.live_points=100 \
        likelihoods.filename=${DES_CLUSTER_NERSC_DIR}/data/mock/dv_buzzard_jkcov.npz \
        polychord.base_dir=${BUZZ_DIR} \
        polychord.polychord_outfile_root=buzzard_polychord \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard/chain.txt
