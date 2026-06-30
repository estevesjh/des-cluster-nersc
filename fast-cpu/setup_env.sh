#!/bin/bash
# Source this file (do NOT execute) before running the fast-cpu pipeline.
#
# fast-cpu = mock_mcmc_cp_camb.ini production pipeline from
# /pscratch/sd/j/jesteves/y3_cluster_cpp_fresh, documented in
# docs/pipeline_modules.tex.  The C++ modules in
# release-build/src/modules/{num_counts_sel, b_sel_marg_cpu, sigma_prj_cpu}
# link libcuda.so.1 at compile time but only use Cuba CPU integration at
# runtime.  The stub libcuda on --constraint=cpu nodes resolves dlopen fine.
# "Parameters never used" warnings are a CosmoSIS false positive (Python
# ini-tracking doesn't see C++ DataBlock reads).

# ---- toolchain swaps required by the fresh build (see BUILDING.md §4) ----
module swap cudatoolkit/12.9 cudatoolkit/12.2 2>/dev/null
module swap gcc-native/13.2 gcc-native/12.3 2>/dev/null
export PATH=$(echo $PATH | tr ':' '\n' | grep -v homebrew | tr '\n' ':' | sed 's/:$//')

# ---- shared install tree ----
export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}

export INTEGRATION_TOOLS_DIR=${TOP_DIR}/y3_pipe_under
export CUBA_DIR=${INTEGRATION_TOOLS_DIR}/cuba
export CUBA_CPP_DIR=${INTEGRATION_TOOLS_DIR}/cubacpp
export GPU_INT_DIR=${INTEGRATION_TOOLS_DIR}/gpuintegration

# ---- fast-cpu Y3 pipeline source: the canonical checkout ----
export Y3PIPE_DIR=/pscratch/sd/j/jesteves/y3_cluster_cpp
export Y3_CLUSTER_CPP_DIR=${Y3PIPE_DIR}
export Y3_CLUSTER_WORK_DIR=${Y3PIPE_DIR}/release-build

# ---- mock-MCMC driver repo (where this setup_env.sh lives) ----
export DES_CLUSTER_NERSC_DIR=/pscratch/sd/j/jesteves/github/des-cluster-nersc

# ---- PYTHONPATH so bsel.py can import y3_buzzard.prj_params ----
# Without this the SLURM batch env can't find the y3_buzzard package.
export PYTHONPATH=${Y3_CLUSTER_CPP_DIR}:${PYTHONPATH:-}

export OMP_NUM_THREADS=1

# ---- CosmoSIS env (y3cl_je conda env, cosmosis-global is outdated) ----
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc \
       /global/common/software/des/common/Conda_Envs/y3cl_je
