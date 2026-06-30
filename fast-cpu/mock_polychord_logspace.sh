#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=mock_polychord_log
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=09:00:00
#SBATCH --output=mock_polychord_logspace.log
#SBATCH --error=mock_polychord_logspace.error
#
# Production polychord nested sampler on the mock pipeline, LOG-SPACE
# likelihood (likelihoods.log_space=T — now the ini default, set explicitly
# here for clarity). Same DV (mock_dv_widePlanck_jkcov.npz) as the linear-space
# production run; only the likelihood transform differs, so this is a direct
# A/B of mixing/contours.
#
# Writes to a SEPARATE base_dir + chain so it does not clobber the linear-space
# production run (mock_polychord.sh -> polychord/, chain.txt).
#
# Layout: 1 node, 64 MPI ranks (one per physical core), 2 logical CPUs/rank.
# Throughput per mock_polychord.sh: ~600k-1M likelihood calls -> 5-8 hr at
# live_points=500. 9h cap gives buffer; polychord checkpoints internally so a
# wall-time hit can be resumed (flip resume=F -> resume=T and re-submit).

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}
OUTDIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/polychord_logspace
mkdir -p ${OUTDIR}/clusters

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord polychord.resume=F \
        likelihoods.log_space=T \
        polychord.base_dir=${OUTDIR} \
        output.filename=${OUTDIR}/chain.txt
