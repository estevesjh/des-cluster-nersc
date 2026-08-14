#!/usr/bin/env python
"""Build the Buzzard convergence-test data vector.

Unlike validations/build_mock_datavector.py and cosmosis-models/generate_mock_dv.py
(both *self-closure*: they harvest the pipeline's own theory so logL=0 at truth),
this script packages the *Buzzard simulation* data vector as the "data" the
cosmosis pipeline must fit. Recovering the Buzzard truth then becomes a genuine
test of the forward model, not a tautology.

It reproduces the deterministic analytical route of
mock_cluster_buzzard/MockDataVector.ipynb (seed=42):
    photo-z proxy -> sample_lambda_true -> sample_lambda_obs
    -> Omega(z) footprint weight -> weighted NC + C1-corrected DeltaSigma stack
    -> gamma_t = DeltaSigma_obs * Sigma_crit^-1
then maps onto the pipeline data-vector contract (likelihood_cp.py):
    data_NC      (12,)     z-major, lambda-fast      idx = iz*4 + il
    data_Shear   (180,)    z-major, lambda, r fast   idx = iz*60 + il*15 + ir
    invcov_NC    (12,12)   from Cov_ij_bestfit_DESY1_105.txt (top-left 12x12)
    invcov_Shear (180,180) from wl_cov.txt

The slow empirical (Wu+22 bootstrap) route is intentionally skipped.

Usage:
    python validations/build_buzzard_datavector.py \
        [--out data/mock/mock_dv_buzzard.npz] \
        [--mock-repo /pscratch/sd/j/jesteves/github/mock_cluster_buzzard]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

REPO_ROOT = Path(__file__).resolve().parents[1]

_NC_N_BINS = 12
_SHEAR_N_R = 15
_SHEAR_N = _NC_N_BINS * _SHEAR_N_R   # 180

# DES Y1 binning (matches the pipeline ini and the Buzzard notebook).
LBDBINS = np.array([20, 30, 45, 60, 500])
ZMIN_LIST = np.array([0.20, 0.35, 0.50])
ZMAX_LIST = np.array([0.35, 0.50, 0.65])

# Pipeline shear radial grid (shear_prj r_perp, comoving Mpc/h) -- one bin's
# worth; identical across the 12 bins in mock_mcmc_cp_camb.ini.
R_PERP_CMPC_H = np.array([
    0.0426, 0.0669, 0.1045, 0.1652, 0.2607, 0.4117, 0.6505, 1.0257,
    1.6181, 2.5537, 4.0265, 6.3490, 10.0107, 15.7832, 24.8771,
])

# Buzzard ground-truth parameters (the values the mock was built with). These
# are what the convergence test should recover. log1e10As is APPROXIMATE:
# derived from sigma_8 = 0.82, not an exact A_s the mock was drawn at.
BUZZARD_TRUTH = {
    "h0":           0.700,
    "omega_m":      0.300,
    "omega_b":      0.046,
    "n_s":          0.960,
    "log1e10As":    3.05,     # APPROX (sigma_8 = 0.82)
    "log10_Mmin":   11.3853,
    "log10_ratio":  1.3112,   # log10(M1/Mmin)
    "alpha":        0.8587,
    "epsilon":      0.2839,
    "sigma_lambda": 0.1809,
}


def _load_mock_halos(mock_repo: Path):
    """Reproduce the notebook's halo selection + RNG-driven lambda sampling.

    Returns a dict with the per-halo arrays the data-vector build needs.
    """
    import fitsio
    from astropy.table import Table, join

    src = mock_repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from fileLoc import FileLocs
    from costanzi_selection import (
        sample_lambda_true, sample_lambda_obs, load_prj_posterior_mean,
    )
    from omega_z_des import survey_weight, omega_z_deg2, BUZZARD_AREA_DEG2

    cosmo0 = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.046, Tcmb0=2.725)
    prj_file = mock_repo / "data" / "prj_params_DESY3_lss_lin_dep_getdist_v1.txt"
    prj_interp = load_prj_posterior_mean(str(prj_file))

    ZMIN, ZMAX, LOGM_MIN = 0.2, 0.65, 13.0
    rng = np.random.default_rng(seed=42)

    floc = FileLocs(machine="nersc")
    data_h, _ = fitsio.read(floc.halo_run_fname, header=True)
    data, _ = fitsio.read(floc.profile_output_fname, header=True)

    select_good = (
        (data["pid"] == -1)
        & (data["cosi"] >= 0) & (data["cosi"] <= 1)
        & ((data["redshift"] < 0.33) | (data["redshift"] > 0.37))
    )
    redshift_sg = data["redshift"][select_good]
    logMvir_sg = np.log10(data["Mvir"])[select_good]
    sel = (redshift_sg >= ZMIN) & (redshift_sg <= ZMAX) & (logMvir_sg >= LOGM_MIN)

    _mock = Table(data[select_good][sel])
    _data_h = Table(data_h)
    _data_h.rename_column("HALOID", "haloid")
    mock = join(_mock, _data_h)
    print(f"[build_buzzard] halos in mock: {len(mock)}")

    # --- RNG call #1: photo-z proxy (must precede the lambda draws) ---
    SIGMA_Z_FRAC = 0.01
    z_true = np.asarray(mock["redshift"], dtype=float)
    z_obs = z_true + SIGMA_Z_FRAC * (1.0 + z_true) * rng.standard_normal(size=len(mock))
    z_obs = np.clip(z_obs, 0.02, None)

    # --- RNG call #2: lambda_true ---
    Mvir = np.asarray(mock["Mvir"], dtype=float)
    lambda_true = sample_lambda_true(Mvir, z_true, rng=rng).astype(float)

    # --- RNG call #3: lambda_obs (drawn at z_obs, matching the notebook) ---
    lambda_obs, *_ = sample_lambda_obs(lambda_true, z_obs, prj_interp, rng=rng)

    w_omega = survey_weight(z_true, area_buzzard_deg2=BUZZARD_AREA_DEG2)

    return {
        "cosmo0": cosmo0,
        "floc": floc,
        "lambda_obs": np.asarray(lambda_obs, dtype=float),
        "z_obs": z_obs,
        "DSigma": np.asarray(mock["DeltaSigma"], dtype=float),
        "w_omega": w_omega,
        "omega_z_deg2": omega_z_deg2,
        "radii_phys_mpc": None,  # filled from rbp below
    }


def _bsel_c1_dsigma(R_phys_mpc, lam_rep, z_rep, h):
    """C26 Eq. C1 bias on DeltaSigma (Matteo mean-fit constants)."""
    A, alpha, beta, gamma, C = 0.12, 4.11, -0.18, 1.82, 1.00
    R_cMpc_h = np.asarray(R_phys_mpc, float) * (1.0 + z_rep) * h
    R0 = (lam_rep / 100.0) ** 0.2 * (1.0 + z_rep)
    x = R_cMpc_h / R0
    return A * x ** alpha * (1.0 + x ** gamma) ** ((beta - alpha) / gamma) + C


def _build_observables(halos, radii_phys_mpc):
    """Footprint-weighted NC (4,3) and analytical C1 shear (4,3,15)."""
    cosmo0 = halos["cosmo0"]
    h = cosmo0.h
    lob = halos["lambda_obs"]
    zob = halos["z_obs"]
    DSigma = halos["DSigma"]
    w = halos["w_omega"]
    floc = halos["floc"]

    # Sigma_crit^-1(z_lens): same construction as the notebook cell.
    bTable = np.load(floc.mock_boost_factor_1d)
    betaEff = bTable["betaEff"]
    zlens = bTable["zlens"]
    const = 6.01e-19 * 1e6
    dlens = cosmo0.comoving_distance(zlens).value * 1e6
    sigma_crit_inv_vec_z = (const * dlens[:, None] * betaEff)[:, -1]

    nl, nz, nr = len(LBDBINS) - 1, len(ZMIN_LIST), len(radii_phys_mpc)
    NC = np.zeros((nl, nz))
    NC_raw = np.zeros((nl, nz), dtype=int)
    gamma_obs_c1 = np.zeros((nl, nz, nr))

    for il in range(nl):
        in_lam = (lob >= LBDBINS[il]) & (lob < LBDBINS[il + 1])
        for iz in range(nz):
            m = in_lam & (zob >= ZMIN_LIST[iz]) & (zob < ZMAX_LIST[iz])
            NC_raw[il, iz] = int(m.sum())
            if not m.any() or w[m].sum() <= 0:
                continue
            NC[il, iz] = float(w[m].sum())

            wbin = w[m]
            dsig_stack = np.average(DSigma[m], axis=0, weights=wbin)
            lam_rep = float(np.average(lob[m], weights=wbin))
            z_rep = float(np.average(zob[m], weights=wbin))

            bsel = _bsel_c1_dsigma(radii_phys_mpc, lam_rep, z_rep, h)
            dsig_obs = dsig_stack * bsel
            # Store DeltaSigma (M_sun/pc^2), NOT gamma_t. The pipeline lensing
            # observable + the Y1 wl_cov.txt covariance are on the DeltaSigma
            # scale (sigma ~ 0.18..76 M_sun/pc^2); gamma_t = DSigma*Sigma_crit^-1
            # would be ~1e4x smaller and incompatible with the covariance.
            gamma_obs_c1[il, iz] = dsig_obs

    return NC, NC_raw, gamma_obs_c1


def _interp_shear_to_pipeline_grid(gamma_buzz, radii_phys_mpc, h):
    """Log-log interp Buzzard shear (4,3,15 on phys-Mpc) onto pipeline r_perp.

    Pipeline r_perp is comoving Mpc/h; convert to physical Mpc per z-bin via
    R_phys = r_perp / (h*(1+z_rep)) and interpolate each (il,iz) profile.
    """
    nl, nz, _ = gamma_buzz.shape
    out = np.zeros((nl, nz, _SHEAR_N_R))
    logr_src = np.log(radii_phys_mpc)
    for iz in range(nz):
        z_rep = 0.5 * (ZMIN_LIST[iz] + ZMAX_LIST[iz])
        r_target_phys = R_PERP_CMPC_H / (h * (1.0 + z_rep))
        logr_tgt = np.log(r_target_phys)
        for il in range(nl):
            y = gamma_buzz[il, iz]
            sgn = np.sign(np.median(y[y != 0])) if np.any(y != 0) else 1.0
            ay = np.abs(y)
            safe = ay > 0
            logy = np.full_like(ay, -np.inf)
            logy[safe] = np.log(ay[safe])
            interp_log = np.interp(logr_tgt, logr_src[safe], logy[safe])
            out[il, iz] = sgn * np.exp(interp_log)
    return out


def _pack_nc(NC_buzz):
    """(il=4, iz=3) -> (12,) with idx = iz*4 + il (z-major, lambda-fast)."""
    return NC_buzz.T.ravel()


def _pack_shear(gamma):
    """(il=4, iz=3, ir=15) -> (180,) idx = iz*60 + il*15 + ir."""
    return np.transpose(gamma, (1, 0, 2)).ravel()


def _load_invcovs():
    cov_nc_file = REPO_ROOT / "y1_rerun" / "data_files" / "Cov_ij_bestfit_DESY1_105.txt"
    cov_wl_file = REPO_ROOT / "y1_rerun" / "data_files" / "wl_cov.txt"
    cov_nc = np.loadtxt(cov_nc_file)[:_NC_N_BINS, :_NC_N_BINS]
    cov_wl = np.loadtxt(cov_wl_file)
    if cov_wl.shape != (_SHEAR_N, _SHEAR_N):
        raise ValueError(f"wl_cov.txt shape {cov_wl.shape} != ({_SHEAR_N},{_SHEAR_N})")
    print(f"[build_buzzard] cov cond: NC={np.linalg.cond(cov_nc):.2e} "
          f"Shear={np.linalg.cond(cov_wl):.2e}")
    return np.linalg.inv(cov_nc), np.linalg.inv(cov_wl)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/mock/mock_dv_buzzard.npz")
    ap.add_argument("--mock-repo",
                    default="/pscratch/sd/j/jesteves/github/mock_cluster_buzzard")
    args = ap.parse_args()

    mock_repo = Path(args.mock_repo).resolve()
    sys.path.insert(0, str(mock_repo / "src"))
    import radial_bins_phys_mpc as rbp  # noqa: E402
    radii_phys_mpc = np.asarray(rbp.rp_phys_mpc, dtype=float)

    halos = _load_mock_halos(mock_repo)
    NC_buzz, NC_raw, gamma_buzz = _build_observables(halos, radii_phys_mpc)

    print("[build_buzzard] NC raw (Buzzard 4143 deg^2):")
    print(NC_raw)
    print("[build_buzzard] NC weighted (DES Omega(z) footprint):")
    print(np.round(NC_buzz, 1))

    h = halos["cosmo0"].h
    gamma_pipe_grid = _interp_shear_to_pipeline_grid(gamma_buzz, radii_phys_mpc, h)

    data_NC = _pack_nc(NC_buzz)
    data_Shear = _pack_shear(gamma_pipe_grid)
    assert data_NC.shape == (_NC_N_BINS,)
    assert data_Shear.shape == (_SHEAR_N,)
    assert np.all(data_NC > 0), f"NC has non-positive bins: {data_NC}"
    assert np.all(np.isfinite(data_Shear)), "Shear has non-finite entries"

    invcov_NC, invcov_Shear = _load_invcovs()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        data_NC=data_NC,
        data_Shear=data_Shear,
        invcov_NC=invcov_NC,
        invcov_Shear=invcov_Shear,
        fiducial_param_names=np.array(list(BUZZARD_TRUTH.keys())),
        fiducial_param_values=np.array(list(BUZZARD_TRUTH.values()), dtype=float),
    )

    print(f"[build_buzzard] wrote {out_path}")
    print(f"  data_NC    (12):  min={data_NC.min():.1f} max={data_NC.max():.1f}")
    print(f"  data_Shear (180): min={data_Shear.min():.3e} max={data_Shear.max():.3e}")
    print(f"  truth: {BUZZARD_TRUTH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
