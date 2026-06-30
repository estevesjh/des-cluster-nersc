#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --account=des
#SBATCH --job-name=gen_mock_dv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --time=00:15:00
#SBATCH --output=generate_mock_dv.log
#SBATCH --error=generate_mock_dv.error
#
# Generate the widePlanck mock data vector
# (data/mock/mock_dv_widePlanck.npz) by running the full Costanzi-2026
# pipeline at fiducial HOD with sampler=test and packaging
# numcountssel/vals + (gamma_t^1h_avg + gamma_t^prj) into an NPZ
# alongside the Y1 inverse covariance matrices.
#
# Output: ${DES_CLUSTER_NERSC_DIR}/data/mock/mock_dv_widePlanck.npz
#   data_NC      (12,)    numcountssel/vals
#   data_Shear   (180,)   12 bins x 15 radii
#   invcov_NC    (12,12)
#   invcov_Shear (180,180)
#
# Pre-flight: CPU job (constraint=cpu, no --gpus). The C++ .so files
# dlopen the cuda stub on cpu nodes without invoking GPU. Debug QOS so
# the run is bounded; ~5 minutes wall in practice.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

echo "which cosmosis: $(which cosmosis)"
echo "python: $(python --version)"

cd ${DES_CLUSTER_NERSC_DIR}
mkdir -p ${DES_CLUSTER_NERSC_DIR}/data/mock
mkdir -p ${DES_CLUSTER_NERSC_DIR}/chains/generate_mock_dv/test

srun -n 1 cosmosis cosmosis-models/generate_mock_dv.ini

echo "--- Done. Verifying output ---"
python3 -c "
import numpy as np
f = np.load('${DES_CLUSTER_NERSC_DIR}/data/mock/mock_dv_widePlanck.npz')
print('Keys:', list(f.keys()))
for k in sorted(f.keys()):
    a = f[k]
    print(f'  {k}: shape={a.shape} min={a.min():.4e} max={a.max():.4e}')
nc = f['data_NC']
sh = f['data_Shear']
assert nc.shape == (12,), f'NC shape {nc.shape} != (12,)'
assert sh.shape == (180,), f'Shear shape {sh.shape} != (180,)'
assert nc.min() > 1.0, f'NC min too low: {nc.min()}'
print('PASS: mock DV validated (12 NC + 180 Shear)')
"
