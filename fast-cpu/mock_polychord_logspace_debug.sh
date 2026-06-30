#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --account=des
#SBATCH --job-name=mock_poly_log_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00
#SBATCH --output=mock_polychord_logspace_debug.log
#SBATCH --error=mock_polychord_logspace_debug.error
#
# Debug-QOS shake-down of polychord with the NEW log-space likelihood
# (likelihoods.log_space=T). Goal: confirm it samples cleanly end-to-end and
# the posterior recovers fiducial, before committing a full 9h shared run.
#
# Differences from mock_polychord_debug.sh:
#   - likelihoods.log_space=T : evaluate the Gaussian on ln(NC), ln(shear)
#   - live_points = 25        : minimal nlive for a fast smoke (NOT converged)
#   - separate base_dir + output filename so this does NOT clobber the
#     production polychord/ checkpoint or chain.txt.
#
# CPU job (constraint=cpu); C++ .so modules resolve cuda stubs cleanly.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}
OUTDIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/polychord_logspace_debug
mkdir -p ${OUTDIR}/clusters

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord runtime.resume=F \
        likelihoods.log_space=T likelihoods.verbose=F \
        polychord.live_points=25 polychord.base_dir=${OUTDIR} \
        output.filename=${OUTDIR}/chain.txt
