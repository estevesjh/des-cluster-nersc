# Buzzard validation suite

Component-by-component validation of the DES-Y3 cluster pipeline against the
Buzzard v1.9.8 simulation, at Buzzard fiducial cosmology
(FlatLambdaCDM H0=70, Om0=0.286, DeRose+2019). Each test compares one pipeline
ingredient to the Buzzard halo catalog and checks the mismatch is within the
relevant covariance.

Common conventions: sky area 4143 deg^2, z in [0.2, 0.65] with the seam
[0.33, 0.37] excised, masses/lengths in little-h (M_sun/h, Mpc/h).

| # | Test | Script | Status |
|---|------|--------|--------|
| 1 | **HMF** dn/dlogM vs Tinker | `test_hmf.py` | ⚠️ infra done; needs the **volume-complete rockstar catalog** (available catalogs are the redMaPPer/cluster sample ~580-600k, not complete → Buzzard reads 5-50x below Tinker, a selection artifact). |
| 2 | **Survey area** Ω(z) | (closed) | ✅ done — see y3_cluster_cpp **issue #8** (Ω(z) hardcoded in omega_z_des.hh, no ini hook). |
| 3 | **c(M)** concentration vs Child18 | `test_cM.py` | ✅ done — Buzzard c=R200m/rs is 1.1-1.4× Child18 (median 1.24), justifies concentration_amplitude~1.25. Fig also in y3_cluster_cpp docs/figs + **issue #21**. |
| 4 | **ρ_m,0 × (1+z)** density evolution | `investigate_rhom_1pz.py` | 🔬 investigating — fit NFW to Buzzard M200m/rs in (mass, z) bins to test the (1+z) surface-density factor. y3_cluster_cpp **issue #22**. |

## Running

Tests 1 and 4 read the Buzzard halo profile catalog via
`mock_cluster_buzzard/src/fileLoc.py` (NERSC paths) and, for the pipeline side,
a `sampler=test` datablock dump at Buzzard fiducial:

```bash
# produce a Buzzard-fiducial datablock (from des-cluster-nersc/):
source fast-cpu/setup_env.sh
export Y3_CLUSTER_CPP_DIR=/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev
cosmosis cosmosis-models/mock_mcmc_cp_camb_buzzard.ini \
    -p runtime.sampler=test test.save_dir=/tmp/buzz_db \
       halo_model.concentration_amplitude=1.25

python validations/buzzard/test_hmf.py --db /tmp/buzz_db
python validations/buzzard/test_cM.py          # self-contained (reads the catalog)
python validations/buzzard/investigate_rhom_1pz.py
```

## Notes / findings
- **c(M)** is z-dependent: c_Buzz/c_Child18 ≈ 1.39 for z<0.33, ≈ 1.22 for z>0.37.
  A z-split concentration (+ the (1+z) factor) gives the best Buzzard ΔΣ fit
  (χ²≈3.0/dof on R≥0.4); see `../buzzard_vs_theory_zsplit_conc.png`.
- Small radii R<0.4 Mpc/h are not trusted (baryons/resolution) — quote χ² on R≥0.4.
- The covariance to judge these against is the real Buzzard one
  (`dataVec_mock_May10th2023.npz`, ~6-16% per-radius); do NOT fabricate errors.
