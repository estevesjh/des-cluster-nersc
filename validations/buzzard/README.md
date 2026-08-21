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
| 4a | **ρ(R200m) × (1+z)³** density evolution (3D) | `investigate_rhom_1pz.py` | ✅ done — NFW fit to truth ρ(r), read at R200m: ρ(R200m)/ρ_m(z=0) ∝ (1+z)³·⁴, mass-collapsed; **+10.5%** box-seam step at z=0.33. y3_cluster_cpp **issue #22**. |
| 4b | **Σ(R200m) × (1+z)³** density evolution (projected) | `investigate_Sigma_R200m.py` | ✅ done — projected-NFW (Wright & Brainerd) fit to truth Σ(R), bg-subtracted: Σ(R200m)/(ρ_m(z=0)·R200m) ∝ (1+z)³·⁸, same c200m/R200m as 3D; **+18.8%** box-seam step (projection amplifies it). |

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
python validations/buzzard/test_cM.py             # self-contained (reads the catalog)
python validations/buzzard/investigate_rhom_1pz.py     # 4a: 3D ρ(R200m) vs z
python validations/buzzard/investigate_Sigma_R200m.py  # 4b: projected Σ(R200m) vs z
```

Tests 4a/4b are self-contained (read the profile catalog directly).

## Notes / findings
- **Density at R200m tracks the PHYSICAL mean density (tests 4a/4b).** Fit an NFW
  to the truth 3D ρ(r) *and* projected Σ(R), locate R200m by spherical overdensity
  (mean enclosed = 200·ρ_m(z)), and read the *measured* density there. Both give a
  mass-collapsed band ∝ (1+z)³ (n≈3.4 for ρ, 3.8 for Σ), with matching c200m≈4–5
  and R200m≈0.6–1.8 Mpc. The pipeline builds its 1-halo with **comoving ρ_m0 frozen
  at z=0** (i.e. (1+z)⁰), so it structurally omits this growth — the physical origin
  of the empirical (1+z) correction. (In projected ΔΣ the LOS integral dilutes
  (1+z)³ to the effective ~(1+z)^0.7–1 that flattens the fit.) The (1+z)³ is partly
  definitional (R200m is set by 200·ρ_m(z)); the genuine content is the **mass
  collapse**, the **sensible c200m**, and the **box-seam step** below.
- **Buzzard simulation-box seam at z=0.33 is a real ~10–19% density step.** Relative
  to the (1+z)³ reference, z<0.33 halo boundaries are under-dense (ρ: 0.90, Σ: 0.79)
  while z>0.37 sit on the line (ρ: 1.00, Σ: 0.94): a **+10.5% (3D) / +18.8%
  (projected)** step across the seam. Projection amplifies it (integrates the
  under-dense low-z outskirts). This is the amplitude deficit that makes the z0 bin
  (0.2–0.33) the ΔΣ-fit outlier needing the stronger concentration boost — a
  stitching artifact, not smooth physics. Plots `rho_R200m_measured_over_rhom0_vs_z.png`,
  `Sigma_R200m_over_rhom0_vs_z.png`.
- **c(M)** is z-dependent: c_Buzz/c_Child18 ≈ 1.39 for z<0.33, ≈ 1.22 for z>0.37.
  A z-split concentration (+ the (1+z) factor) gives the best Buzzard ΔΣ fit
  (χ²≈3.0/dof on R≥0.4); see `../buzzard_vs_theory_zsplit_conc.png`.
- Small radii R<0.4 Mpc/h are not trusted (baryons/resolution) — quote χ² on R≥0.4.
- The covariance to judge these against is the real Buzzard one
  (`dataVec_mock_May10th2023.npz`, ~6-16% per-radius); do NOT fabricate errors.
