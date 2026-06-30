#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --account=des
#SBATCH --job-name=buzzard_dv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cpu
#SBATCH --time=00:25:00
#SBATCH --output=build_buzzard_dv.log
#SBATCH --error=build_buzzard_dv.error
#
# Build the Buzzard convergence-test data vector and validate it end-to-end
# in one debug-qos allocation:
#   1. run validations/build_buzzard_datavector.py  -> data/mock/mock_dv_buzzard.npz
#   2. assert DV shapes / finiteness
#   3. run mock_mcmc_buzzard.ini with sampler=test (single pipeline eval)
#   4. dump pipeline theory (NC, shear) vs the Buzzard data vector so we can
#      eyeball the ordering + radial-units mapping (CRITICAL check).
#
# CPU job: the C++ .so modules dlopen the cuda stub on cpu nodes (see
# fast-cpu/setup_env.sh).  Debug qos so the run is bounded.

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh

echo "which cosmosis: $(which cosmosis)"
echo "python: $(python --version)"

cd ${DES_CLUSTER_NERSC_DIR}
mkdir -p ${DES_CLUSTER_NERSC_DIR}/data/mock
mkdir -p /pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard/test

# --- 1+2. Build + shape assertions --------------------------------------
echo "=== [1] building Buzzard data vector ==="
srun -n 1 python validations/build_buzzard_datavector.py \
     --out ${DES_CLUSTER_NERSC_DIR}/data/mock/mock_dv_buzzard.npz

echo "=== [2] verifying DV shapes ==="
python3 -c "
import numpy as np
f = np.load('${DES_CLUSTER_NERSC_DIR}/data/mock/mock_dv_buzzard.npz', allow_pickle=False)
nc, sh = f['data_NC'], f['data_Shear']
icn, ics = f['invcov_NC'], f['invcov_Shear']
assert nc.shape == (12,),     f'NC {nc.shape}'
assert sh.shape == (180,),    f'Shear {sh.shape}'
assert icn.shape == (12,12),  f'invcov_NC {icn.shape}'
assert ics.shape == (180,180),f'invcov_Shear {ics.shape}'
assert np.all(nc > 0) and np.all(np.isfinite(sh)), 'bad DV values'
print('PASS: shapes (12)/(180)/(12,12)/(180,180), NC>0, Shear finite')
print('NC      :', np.round(nc,1))
print('NC.reshape(z=3,lam=4):'); print(np.round(nc.reshape(3,4),1))
"

# --- 3. Single pipeline evaluation (sampler=test) -----------------------
echo "=== [3] sampler=test (one pipeline eval against Buzzard DV) ==="
srun -n 1 cosmosis cosmosis-models/mock_mcmc_buzzard.ini \
     -p runtime.sampler=test

# --- 4. Theory-vs-data overlay (ordering + units sanity) ----------------
# The test sampler dumps the datablock to the [test] save_dir; pull the
# pipeline theory back out and compare to the data vector per bin.
echo "=== [4] theory vs data overlay ==="
TESTDIR=/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard/test
python3 -c "
import numpy as np, os
dv = np.load('${DES_CLUSTER_NERSC_DIR}/data/mock/mock_dv_buzzard.npz')
td = '${TESTDIR}'
def load(sub):
    p = os.path.join(td, sub, 'vals.txt')
    return np.loadtxt(p, comments='#').ravel() if os.path.exists(p) else None
nc_th = load('numcountssel')
s1h   = load('shear1hmissel')
sprj  = load('shear_prj')
print('--- NUMBER COUNTS: data vs theory (12 bins) ---')
if nc_th is not None:
    for i,(d,t) in enumerate(zip(dv['data_NC'], nc_th)):
        print(f'  bin {i:2d}: data={d:9.1f}  theory={t:9.1f}  ratio={t/d:6.3f}')
else:
    print('  (numcountssel/vals.txt not found in', td, ')')
if s1h is not None and sprj is not None:
    nc_tile = np.repeat(nc_th, 15)
    sh_th = np.where(nc_tile>0, s1h/nc_tile, 0.0) + sprj
    d = dv['data_Shear']
    print('--- SHEAR: data vs theory summary (180) ---')
    print(f'  data  range:  [{d.min():.3e}, {d.max():.3e}]')
    print(f'  theory range: [{sh_th.min():.3e}, {sh_th.max():.3e}]')
    r = np.where(d!=0, sh_th/d, np.nan)
    print(f'  theory/data ratio: median={np.nanmedian(r):.3f} '
          f'p16={np.nanpercentile(r,16):.3f} p84={np.nanpercentile(r,84):.3f}')
    print('  (order-of-magnitude agreement => ordering + radial units OK;')
    print('   a clean ~(1+z)*h offset => radial conversion flipped)')
else:
    print('  (shear vals.txt not found)')
"
echo "--- Done. ---"
