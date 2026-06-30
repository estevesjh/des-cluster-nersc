# Running the mock-MCMC pipeline on real simulation data — guide for Arwa

This explains how to point the existing `fast-cpu/` polychord pipeline at **your
own simulation data vector** while **keeping the same covariance**. You do not
need to touch the likelihood code or the pipeline `.ini` — you only build a new
`.npz` data-vector file and tell the likelihood to load it.

See `fast-cpu/HOW_TO_SUBMIT.md` for how to actually submit the SLURM jobs once
your data vector is in place.

---

## 1. What the likelihood expects (the data-vector contract)

The Gaussian likelihood (`$Y3_CLUSTER_CPP_DIR/y3_buzzard/likelihood_cp.py`)
loads a single `.npz` file with **exactly these four arrays**:

| key | shape | meaning |
|---|---|---|
| `data_NC` | `(12,)` | number counts: 4 richness × 3 redshift bins |
| `data_Shear` | `(120,)` | tangential shear: 12 bins × **10 radii**, flattened (bin-major) |
| `invcov_NC` | `(12, 12)` | **inverse** covariance of the number counts |
| `invcov_Shear` | `(120, 120)` | **inverse** covariance of the shear |

> **Important — these are INVERSE covariances**, not covariances. The likelihood
> uses them directly as `chi2 = delta^T @ invcov @ delta`. If you have a
> covariance `C`, store `np.linalg.inv(C)`.

The sizes are asserted at startup (`_NC_N_BINS=12`, `_SHEAR_N=120` in
`likelihood_cp.py`). If an array is the wrong length the job fails immediately
with a clear `data_<name> has size ...` error — so a mistake here can't silently
corrupt a run.

### The shear layout (read carefully)

`data_Shear` is **12 bins × 10 radii, flattened bin-major**:
`[bin0_r0, bin0_r1, …, bin0_r9, bin1_r0, …, bin11_r9]`.

- The 12 bins are 4 richness × 3 redshift, in the same order as the `.ini`
  `lambda_bin` / `zo_low` lists.
- The 10 radii (Mpc/h comoving) are the `r_perp` grid from
  `cosmosis-models/mock_mcmc_cp_camb.ini`:
  `0.200, 0.286, 0.409, 0.585, 0.836, 1.196, 1.710, 2.445, 3.497, 5.000`.

Your simulation shear **must be sampled on these same 10 radii and 12 bins** so
it matches the theory vector the pipeline computes. If your sim uses different
radii, interpolate it onto this grid before storing.

---

## 2. What you change vs. what you keep

- **CHANGE:** `data_NC` and `data_Shear` → your measured simulation values.
- **KEEP:** `invcov_NC` and `invcov_Shear` → copy them straight from the existing
  mock file. This is exactly what you asked for: same covariance, new data.

The current mock file is
`data/mock/mock_dv_widePlanck_jkcov.npz` (Poisson NC cov + Buzzard jackknife
shear cov, rebinned to the 10-radii pipeline grid).

---

## 3. Build your data-vector file

Make a copy and swap in your data, reusing the covariances:

```python
import numpy as np

# 1. Load the existing mock to inherit its covariance (and the layout template).
mock = np.load("data/mock/mock_dv_widePlanck_jkcov.npz")

# 2. Your measured simulation observables, on the SAME binning/radii as §1.
#    data_NC    -> (12,)  : 4 richness x 3 z number counts
#    data_Shear -> (120,) : 12 bins x 10 radii, flattened bin-major
data_NC    = my_sim_number_counts      # shape (12,)
data_Shear = my_sim_shear.ravel()      # shape (120,) — bin-major flatten

# sanity checks (the likelihood will assert these too)
assert data_NC.shape    == (12,),  data_NC.shape
assert data_Shear.shape == (120,), data_Shear.shape

# 3. Keep the SAME covariance as the mock.
np.savez(
    "data/mock/dv_arwa_sim.npz",
    data_NC=data_NC,
    data_Shear=data_Shear,
    invcov_NC=mock["invcov_NC"],        # unchanged
    invcov_Shear=mock["invcov_Shear"],  # unchanged
)
```

> If your shear array is shaped `(12, 10)` (bin, radius), `.ravel()` gives the
> correct bin-major order. Double-check `data_Shear[:10]` is bin 0's radial
> profile.

For a worked example of harvesting a data vector from a pipeline run, see
`validations/build_buzzard_datavector.py` (it builds a 180-vector / 15-radii
version — note the radial count differs from the active 120/10 pipeline, so use
it as a template, not a drop-in).

---

## 4. Point the pipeline at your file

Two options — **command-line override is easiest** (no file edits):

```bash
# from the repo root, after sbatch-style env setup
... cosmosis cosmosis-models/mock_mcmc_cp_camb.ini \
    -p runtime.sampler=polychord \
       likelihoods.filename=/pscratch/.../data/mock/dv_arwa_sim.npz
```

In a SLURM script, just add that `likelihoods.filename=...` to the `srun ... -p`
line (copy `fast-cpu/mock_polychord.sh` and add the override).

Or edit the `[likelihoods]` section of `cosmosis-models/mock_mcmc_cp_camb.ini`:

```ini
[likelihoods]
file = ${Y3_CLUSTER_CPP_DIR}/y3_buzzard/likelihood_cp.py
filename = ${DES_CLUSTER_NERSC_DIR}/data/mock/dv_arwa_sim.npz   ; <-- your file
log_space = F
```

---

## 5. Important: this is no longer a closure test

The mock data vector was the pipeline's own noiseless theory, so `logL ≈ 0` at
the truth and the posterior must recover the fiducial parameters. With **real
simulation data**, that is no longer true:

- `logL` at the input cosmology will **not** be ~0 — model misspecification
  (projection, radial grid, selection bias, sim systematics) shows up as a real
  offset.
- A recovery **bias** (posterior away from the sim's input cosmology) is now a
  physics result to interpret, not a bug.

This is exactly the role `cosmosis-models/mock_mcmc_buzzard.ini` plays (it
compares against a Buzzard sim DV instead of self-closure theory) — you can use
it as a reference for a "real-data" config. Note it currently expects a
180-vector (15 radii); if you stick with the 120/10 layout above, use
`mock_mcmc_cp_camb.ini` as your base instead.

---

## 6. Quick checklist

- [ ] Shear on the **10 r_perp radii** and **12 bins** from the `.ini` (interpolate if not).
- [ ] `data_Shear` flattened **bin-major** → `(120,)`.
- [ ] `data_NC` → `(12,)`.
- [ ] Reuse `invcov_NC` / `invcov_Shear` from `mock_dv_widePlanck_jkcov.npz` (**inverse** covariances).
- [ ] `np.savez` the four keys with the **exact names** above.
- [ ] Run `mock_polychord_debug.sh` first (point it at your file) before the 9 h production run.
- [ ] Remember: `logL` ≠ 0 and a recovery bias is now meaningful, not an error.

Questions: ping Johnny. Pipeline submission details are in
`fast-cpu/HOW_TO_SUBMIT.md`.
