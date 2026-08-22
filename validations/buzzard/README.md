# Buzzard validation suite

Component-by-component validation of the DES-Y3 cluster weak-lensing pipeline
against the Buzzard v1.9.8 simulation, at Buzzard fiducial cosmology
(FlatLambdaCDM H0=70, Ω_m=0.286, DeRose+2019).

**The storyline.** We check the pipeline ingredients one at a time against the
Buzzard truth halos, and it builds to a single conclusion: **the Buzzard halos
carry the physical (1+z)³ density growth, while the pipeline is built with comoving
ρ_m0 frozen at z=0** — the origin of the empirical (1+z) correction the fit needs.
The remaining open question is the *exact* (1+z) power for the projected observable;
we now have the CLensPy fitting recipe and the stacking machinery to answer it and
to re-make the λ_obs-binned ΔΣ data vector.

📄 **Full write-up with figures: [`REPORT_buzzard_1pz_density.md`](REPORT_buzzard_1pz_density.md)**

Conventions: quality cuts `pid==-1`, `0≤cosi≤1`, box-seam excised (0.33≤z≤0.37),
Mvir≥10¹³ M⊙. The truth profiles (ρ, Σ, ΔΣ) and their radii are **physical**
(M⊙/Mpc³, M⊙/pc², physical Mpc); catalog masses/radii are little-h. Concentrations
are quoted as **physical M200m** (200·ρ_m(z)) unless a figure says CLensPy/frozen-z0
(the two differ by (1+z)^~1.3 — see the report §2).

## The tests

| # | Test | Script(s) | Status |
|---|------|-----------|--------|
| 1 | **HMF** dn/dlogM vs Tinker | `test_hmf.py` | ⚠️ infra only — needs a **volume-complete rockstar catalog** (the available catalogs are the redMaPPer/cluster sample, not complete). |
| 2 | **Survey area** Ω(z) | (closed) | ✅ y3_cluster_cpp **issue #8** (Ω(z) hardcoded in `omega_z_des.hh`). |
| 3 | **c(M)** vs Child18 | `test_cM.py` | ✅ Buzzard c=R200m/rs ≈ 1.24× Child18 → `concentration_amplitude~1.25`. **issue #21**. |
| 4 | **Density at R200m ∝ (1+z)³** (ρ, Σ, ΔΣ) | `investigate_R200m_clenspy.py` | ✅ physical (1+z)³, mass-collapsed; pipeline is comoving. **issue #22**. See report. |
| 5 | **Fitting recipe** NFW+2h, R>0.2 | `fit_profiles_clenspy.py` | ✅ CLensPy MAX(NFW, b·2h); recovers consistent M,c across ρ/Σ/ΔΣ. |
| 6 | **Stacking & selection** | `heidi_stacking_demo.py`, `richness_vs_mass_concentration.py` | ✅ Heidi's (logM,z)-matched stack = selection bias; λ_obs adds **no** concentration bias (mass-mixing lowers c ~5–10%). |
| 7 | **λ_obs ΔΣ data vector + JK cov** (pipeline units) | `../build_buzzard_dv_deltasigma_heidi.py`, `jk_convergence_deltasigma.py` | ✅ ΔΣ (not γ_t) little-h; Heidi matched stack; K=50 JK cov (validated converged). → `data/mock/dv_buzzard_deltasigma_heidi.npz`. |

All NFW / ΔΣ / 2-halo evaluation uses **CLensPy** (`clenspy.halo.NfwProfile`,
`clenspy.halo.TwoHaloTerm`, `clenspy.cosmology.PkGrid`) — one code path everywhere.

## Running

```bash
# HMF needs a pipeline datablock (Buzzard fiducial):
source fast-cpu/setup_env.sh
export Y3_CLUSTER_CPP_DIR=/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev
cosmosis cosmosis-models/mock_mcmc_cp_camb_buzzard.ini \
    -p runtime.sampler=test test.save_dir=/tmp/buzz_db halo_model.concentration_amplitude=1.25
python validations/buzzard/test_hmf.py --db /tmp/buzz_db

# Everything else is self-contained (reads the profile catalog + CLensPy directly):
python validations/buzzard/test_cM.py
python validations/buzzard/investigate_R200m_clenspy.py     # ρ/Σ/ΔΣ(R200m) vs z
python validations/buzzard/fit_profiles_clenspy.py          # NFW+2h fits + residuals
python validations/buzzard/heidi_stacking_demo.py           # (logM,z)-matched stacking
python validations/buzzard/richness_vs_mass_concentration.py  # λ_obs vs concentration
```

Requires CLensPy on the path (`/pscratch/sd/j/jesteves/github/CLensPy/src`) and the
Buzzard catalogs via `mock_cluster_buzzard/src/fileLoc.py`.

## Key findings (details + figures in the report)

- **Density at R200m follows the physical (1+z)³** in all three probes (ρ, Σ, ΔΣ),
  mass-collapsed, with the same CLensPy fit. The pipeline uses comoving ρ_m0 frozen
  at z=0, so it misses this — the physical origin of the empirical (1+z) correction.
  The same mismatch appears in the concentration convention (physical c≈5 vs
  frozen-z0 c≈8 = (1+z)^~1.3).
- **Robust fit** (R>0.2 Mpc, `MAX(NFW, b·2h)`) makes ρ/Σ/ΔΣ agree on M and c; the
  box-seam step is then ≲1% (no real density discontinuity). Small radii R<0.2 Mpc
  are untrusted (resolution/miscentering — ΔΣ turns over there).
- **λ_obs (HOD, no LSS) selection does not bias concentration** (mass-mixing lowers
  it ~5–10%); real redMaPPer LAMBDA_CHISQ *would* (orientation/LSS), the B_sel
  systematic modeled in xtang126's `MockDataVector.ipynb`.

## Next steps

1. Derive/validate the effective (1+z) power the comoving pipeline needs for the
   projected ΔΣ — now testable with the ΔΣ data vector (test 7).
2. ~~Re-make the ΔΣ data vector for λ_obs bins via Heidi's stacking~~ ✅ done (test 7,
   pipeline units, ΔΣ not γ_t).
3. ~~Build the jackknife covariance~~ ✅ done + validated converged at K=50 (test 7).
   Realism to add: fold in optical-selection bias (for redMaPPer-like) and a
   shape-noise error budget (the JK is amplitude-dominated, ~1–3%).
