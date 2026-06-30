#!/bin/bash -l
#SBATCH --qos=shared
#SBATCH --account=des
#SBATCH --job-name=mock_polychord_rsm
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=03:00:00
#SBATCH --output=mock_polychord_resume.log
#SBATCH --error=mock_polychord_resume.error
#
# RESUME continuation of the LINEAR-space production polychord run
# (mock_polychord.sh, job 55014977) which was still compressing
# (log(Z)~-22.8, sigma~0.19 > tolerance=0.1) when its 9h wall approached,
# with two HOD directions (log10_Mmin, sigma_lambda) showing NaN posterior
# sigma -- i.e. not yet enough effective samples. +3h lets it converge.
#
# Resume mechanism (verified in polychord_sampler.py):
#   - write_resume is ALWAYS true -> base_dir/mock_polychord.resume (~23 MB)
#     is continuously checkpointed.
#   - read_resume = polychord.resume; so polychord.resume=T continues from
#     that checkpoint instead of restarting.
#   - runtime.resume=T (ini default) opens chain.txt in append mode (r+),
#     so the existing chain is extended, not clobbered.
#
# Same base_dir + chain.txt as the parent run (NOT a separate dir) -- that is
# deliberate: resume must read the parent's checkpoint in place.
#
# Submit with a dependency so it only starts AFTER the parent ends (else two
# jobs write the same base_dir and corrupt the .resume):
#   sbatch --dependency=afterany:55014977 fast-cpu/mock_polychord_resume.sh
# afterany (not afterok): a wall-timeout is a non-zero exit, which would
# strand an afterok child forever (DependencyNeverSatisfied).

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

which cosmosis
python --version

cd ${DES_CLUSTER_NERSC_DIR}
OUTDIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/polychord

srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord runtime.resume=T \
        polychord.resume=T \
        likelihoods.log_space=F \
        polychord.base_dir=${OUTDIR} \
        output.filename=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/chain.txt
