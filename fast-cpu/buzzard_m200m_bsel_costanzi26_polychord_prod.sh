#!/bin/bash -l
#SBATCH --account=des
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --job-name=bsel_c26_prod
#SBATCH --output=/pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu/logs/buzzard_m200m_bsel_costanzi26_prod_%j.log
#SBATCH --error=/pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu/logs/buzzard_m200m_bsel_costanzi26_prod_%j.error
#
# Buzzard selection-bias test, run 2: model WITH the Costanzi26 selection bias (A, alpha, beta, gamma flat priors).
#   ini:    des-cluster-nersc cosmosis-models/buzzard_m200m_bsel_costanzi26.ini (fast 0d max-model pipeline,
#           s8_v2 emulators, sigma_8 sampled; chain paths are set in the ini)
#   DV:     des-cluster-nersc data/mock/dv_buzzard_m200m_bsel_pipeline.npz -- the M200m mock DV times
#           the mock's injected B_sel (Costanzi+2026 Eq. C1, Matteo's DeltaSigma constants).
#   Cost:   ~0.7 s (run 1) / ~0.8 s (run 2) per likelihood call per rank; the As chain took ~3 h on 64 ranks.
# The debug job (debug QOS, 30 min) verifies MPI and runs the first ~26 min of the chain;
# the prod job (shared, afterok on debug) resumes it.  Re-submit the prod script to continue.
set -euo pipefail

# Environment: the owner's standard init (CosmoSIS + y3cl_je + Y3_CLUSTER_CPP_DIR on pscratch).
set +u
source /global/homes/j/jesteves/cosmosis_init.sh
set -u
export OMP_NUM_THREADS=1        # cosmosis_init.sh sets 4; one thread per MPI rank here
export DES_CLUSTER_NERSC_DIR=/pscratch/sd/j/jesteves/github/des-cluster-nersc
# PolyChord built against the conda MPICH that y3cl_je's mpi4py uses (the shared install's
# libchord_mpi.so links Cray MPICH 8.1/gcc 9.1, gone since cpe 26.03, and two MPI stacks
# cannot share one srun launch).  A private copy of the cosmosis package goes first.
export PYTHONPATH=${DES_CLUSTER_NERSC_DIR}/fast-cpu/cosmosis_condampi:${PYTHONPATH}
# "Parameter ... never used" warnings for the C++ modules are a CosmoSIS false positive.
which cosmosis
python --version

# MPI: y3cl_je's mpi4py links conda MPICH 4.2.3, which only joins the job's communicator
# under srun --mpi=pmi2 (cpe 26.03: plain srun gives every rank size 1).  Check before running.
NR=64
srun --mpi=pmi2 -n ${NR} python -c "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size())" > mpi_check_${SLURM_JOB_ID}.txt
NSIZE=$(awk '{print $2}' mpi_check_${SLURM_JOB_ID}.txt | sort -u)
NRANK=$(awk '{print $1}' mpi_check_${SLURM_JOB_ID}.txt | sort -u | wc -l)
echo "MPI check: distinct ranks=${NRANK}, reported sizes=${NSIZE}"
rm -f mpi_check_${SLURM_JOB_ID}.txt
if [ "${NSIZE}" != "${NR}" ] || [ "${NRANK}" != "${NR}" ]; then echo "MPI BROKEN: aborting"; exit 1; fi

# Must match [polychord] base_dir and [output] filename in the ini.
RUN=${DES_CLUSTER_NERSC_DIR}/chains/buzzard_m200m_bsel_costanzi26
mkdir -p ${RUN}/polychord/clusters
cd ${RUN}

srun --mpi=pmi2 -n ${NR} cosmosis --mpi ${DES_CLUSTER_NERSC_DIR}/cosmosis-models/buzzard_m200m_bsel_costanzi26.ini \
     -p polychord.resume=T runtime.resume=T pipeline.timing=F
