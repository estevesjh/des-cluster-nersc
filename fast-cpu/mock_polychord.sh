#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=mock_polychord
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=09:00:00
#SBATCH --output=mock_polychord.log
#SBATCH --error=mock_polychord.error
#
# Production polychord nested sampler on the mock pipeline.
#
# Layout: 1 node, 64 MPI ranks (one per physical core), 2 logical CPUs per
# rank for internal OMP parallelism. Polychord uses MPI for parallelization
# (each rank works on independent live-point updates), unlike emcee --smp.
#
# Polychord output: nested sampling chain + evidence (model comparison).
#
# Throughput (calibrated 2026-06-04 from debug run, live_points=100):
#   - 36 evals/sec across all 64 ranks
#   - <nlike> = 127 per dead-point update (slice samples × repeats)
#   - At live_points=500, expect ~3000-5000 dead-point updates for convergence
#     (typical for 10D), so ~600k-1M total likelihood calls -> 5-8 hr wall.
#   12h wall time gives a healthy buffer; polychord checkpoints internally
#   so a wall-time hit can be resumed by re-submitting (resume=T in ini).

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}
# Polychord writes internal files to <base_dir>/clusters/ — must pre-exist
# or the Fortran runtime errors with "Cannot open file ./clusters/..."
mkdir -p /pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/polychord/clusters

# resume=F: fresh start. This run uses the NEW jkcov DV
# (mock_dv_widePlanck_jkcov.npz); any checkpoint from the old Y1-cov run is
# invalid and was removed. The previous run converged in <6h (under the 9h
# cap), so a wall-time resume is not expected. If it ever does time out,
# flip to resume=T and re-submit.
srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord polychord.resume=F
