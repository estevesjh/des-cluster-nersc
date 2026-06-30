#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --account=des
#SBATCH --job-name=mock_polychord_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00
#SBATCH --output=mock_polychord_debug.log
#SBATCH --error=mock_polychord_debug.error
#
# Debug-QOS shake-down of polychord on the mock pipeline.
#
# Layout: 1 node, 64 MPI ranks (one per physical core), 2 logical CPUs per
# rank for internal OMP parallelism. Polychord uses MPI for parallelization,
# unlike emcee which uses --smp within a single rank.
#
# Polychord parameters (tuned for 10-parameter space):
#   nlive = 500 (active points; default ~25*nparam, here ~50*nparam for safety)
#   nrepeat = 60 (number of slice samples per active point)
#   nfail = 60 (max failed slice samples before termination)
#   These are conservative defaults; tune down for faster convergence if needed.
#
# CPU job (constraint=cpu); C++ .so modules resolve cuda stubs cleanly.
#
# Debug wall time cap: 30 min. Polychord will likely not reach convergence
# in this time but will write checkpoints that can be resumed with
# mock_polychord.sh.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}
# Polychord writes internal files to <base_dir>/clusters/ — must pre-exist
# or the Fortran runtime errors with "Cannot open file ./clusters/..."
mkdir -p /pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/polychord/clusters

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord runtime.resume=F polychord.live_points=100
