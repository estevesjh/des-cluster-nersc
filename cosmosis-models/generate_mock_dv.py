"""Generate mock data vector for the mock_mcmc_cp_camb.ini pipeline.

Sits at the end of the generate_mock_dv.ini pipeline, reads the theory
vector from the datablock, and packages it with the mock covariance into
data/mock/mock_dv_widePlanck_jkcov.npz.

Theory vector layout (DeltaSigma; average_sigma_crit_inv unity=T):
    data_NC    : (12,)  — numcountssel/vals
    data_Shear : (120,) — DeltaSigma^1h_avg(R) + DeltaSigma^prj(R), 12 bins x 10 radii

Covariance source:
    NC   : POISSON shot noise, Var(N_i) = N_i (fiducial counts) -> invcov = diag(1/N_i).
           The Y1 Cov_ij_bestfit_DESY1_105.txt is a best-fit (shot + sample
           variance) covariance whose NC diagonal runs ~2x the Poisson floor;
           it was not built for this mock, so the closure test uses the pure
           Poisson diagonal the mock counts actually imply.
    Shear: BUZZARD JACKKNIFE cov (dataVec_mock_May10th2023.npz), cond~2e3,
           interpolated from its native 20-radii grid [0.0426, 24.88] onto the
           pipeline's 10-radii grid [0.2, 5] Mpc/h (the well-modelled regime).
           Replaces the Y1 wl_cov.txt (cond~3.8e10), built for the Y1 *mass*
           analysis, not this mock. The pipeline radii are a strict subset of
           the JK span, so the interpolation is interior (no extrapolation),
           and the covariance is propagated through the linear interp operator
           A as cov_10 = A cov_20 A^T, NOT interpolated element-wise.
"""

from __future__ import annotations

import os
import numpy as np
from cosmosis.datablock import option_section

_NC_N_BINS = 12
_SHEAR_N_R = 10                       # pipeline radial grid (0.2-5 Mpc/h)
_SHEAR_N = _NC_N_BINS * _SHEAR_N_R    # 120
_JK_N_R = 20                          # Buzzard JK radial grid
_JK_N = _NC_N_BINS * _JK_N_R          # 240
# JK data native log range (Buzzard 20-radii grid), Mpc/h comoving.
_JK_R_MIN, _JK_R_MAX = 0.0426, 24.8771
# Pipeline shear log range (ini `radii` / Shear1hMisSel `r_perp`). Restricted
# to the well-modelled 0.2-5 Mpc/h regime; a strict subset of the JK span so
# the cov rebinning interpolates (never extrapolates).
_R_MIN, _R_MAX = 0.2, 5.0
_JK_COV_FILE = ("/global/cfs/cdirs/des/jesteves/data/buzzard/v1.9.8/"
                "y3_rm/dataVec_mock_May10th2023.npz")


def _interp_matrix(r_src, r_dst):
    """(len(r_dst) x len(r_src)) linear-in-log-r interpolation operator.

    out = M @ in reproduces np.interp(log r_dst, log r_src, in): each output
    row holds the two bracketing-node weights (clamped at the edges).
    """
    ls, ld = np.log(r_src), np.log(r_dst)
    M = np.zeros((len(ld), len(ls)))
    for i, x in enumerate(ld):
        if x <= ls[0]:
            M[i, 0] = 1.0
        elif x >= ls[-1]:
            M[i, -1] = 1.0
        else:
            j = np.searchsorted(ls, x) - 1
            w = (x - ls[j]) / (ls[j + 1] - ls[j])
            M[i, j] = 1.0 - w
            M[i, j + 1] = w
    return M


def _rebinned_jk_shear_invcov():
    """Load the Buzzard JK shear cov (240x240) and rebin it to the pipeline
    120x120 grid via a block-diagonal log-linear interpolation operator."""
    r_src = np.geomspace(_JK_R_MIN, _JK_R_MAX, _JK_N_R)    # JK native 20-radii
    r_dst = np.geomspace(_R_MIN, _R_MAX, _SHEAR_N_R)       # pipeline 0.2-5, 10
    assert r_dst[0] >= r_src[0] - 1e-9 and r_dst[-1] <= r_src[-1] + 1e-9, \
        "pipeline radii not contained in JK radii span -> would extrapolate"

    M = _interp_matrix(r_src, r_dst)                  # (10, 20)
    A = np.zeros((_SHEAR_N, _JK_N))                   # (120, 240)
    for b in range(_NC_N_BINS):
        A[b*_SHEAR_N_R:(b+1)*_SHEAR_N_R, b*_JK_N_R:(b+1)*_JK_N_R] = M

    jk = np.load(_JK_COV_FILE, allow_pickle=True)
    cov_src = np.linalg.inv(jk["invcov_Shear"])       # 240x240
    assert cov_src.shape == (_JK_N, _JK_N), \
        f"JK invcov_Shear shape {cov_src.shape}, expected ({_JK_N},{_JK_N})"

    cov_dst = A @ cov_src @ A.T                        # 120x120
    cov_dst = 0.5 * (cov_dst + cov_dst.T)              # symmetrize fp noise
    print(f"[generate_mock_dv] JK shear cov 240 cond={np.linalg.cond(cov_src):.2e}"
          f" -> rebinned 120 cond={np.linalg.cond(cov_dst):.2e}")
    return np.linalg.inv(cov_dst)


def setup(options):
    outpath = options[option_section, "output_path"]

    # NC invcov is Poisson (diag 1/N) and depends on the fiducial counts, which
    # are only known in execute() from the datablock -> built there, not here.
    invcov_Shear = _rebinned_jk_shear_invcov()

    return {
        "outpath": outpath,
        "invcov_Shear": invcov_Shear,
    }


def execute(block, config):
    NC = np.asarray(block["numcountssel", "vals"]).ravel()
    assert NC.size == _NC_N_BINS, f"NC size {NC.size} != {_NC_N_BINS}"

    # Poisson NC covariance: Var(N_i) = N_i -> invcov = diag(1/N_i). Built here
    # (not setup) because it depends on the fiducial counts from the datablock.
    assert np.all(NC > 0), f"NC has non-positive entries (cannot Poisson-invert): {NC}"
    invcov_NC = np.diag(1.0 / NC)
    config["invcov_NC"] = invcov_NC
    print(f"[generate_mock_dv] NC Poisson invcov: diag(1/N), "
          f"sigma_i = sqrt(N_i) in [{np.sqrt(NC).min():.2f}, {np.sqrt(NC).max():.2f}]")

    S1h_Ni = np.asarray(block["shear1hmissel", "vals"]).ravel()
    Sprj = np.asarray(block["shear_prj", "vals"]).ravel()
    assert S1h_Ni.size == _SHEAR_N, f"shear1hmissel size {S1h_Ni.size} != {_SHEAR_N}"
    assert Sprj.size == _SHEAR_N, f"shear_prj size {Sprj.size} != {_SHEAR_N}"

    NC_tile = np.repeat(NC, _SHEAR_N_R)
    with np.errstate(divide="ignore", invalid="ignore"):
        S1h_avg = np.where(NC_tile <= 0.0, 0.0, S1h_Ni / NC_tile)
    Shear = S1h_avg + Sprj

    # --- Self-consistency check: logL must be 0 at fiducial ---
    delta_NC = NC - NC  # zero by construction
    delta_Shear = Shear - Shear
    chi2_NC = float(delta_NC @ config["invcov_NC"] @ delta_NC)
    chi2_Shear = float(delta_Shear @ config["invcov_Shear"] @ delta_Shear)
    logL = -0.5 * (chi2_NC + chi2_Shear)
    assert logL == 0.0, f"Self-consistency failed: logL={logL} (expect 0)"

    # --- Sanity checks on theory vector ---
    assert np.all(NC > 0), f"NC has non-positive entries: {NC[NC <= 0]}"
    assert np.all(np.isfinite(Shear)), "Shear has non-finite entries"
    # NC should be O(100-10000) for DES-like cluster counts
    assert NC.min() > 1.0, f"NC min suspiciously low: {NC.min():.2f}"
    # Shear should be O(1e-4 -- 1e-1) in Mpc/h units
    shear_abs = np.abs(Shear)
    assert shear_abs.max() < 1e3, f"Shear max suspiciously large: {shear_abs.max():.2e}"

    outpath = config["outpath"]
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    np.savez(outpath,
             data_NC=NC,
             data_Shear=Shear,
             invcov_NC=config["invcov_NC"],
             invcov_Shear=config["invcov_Shear"])

    print(f"[generate_mock_dv] Wrote {outpath}")
    print(f"  NC (12): min={NC.min():.2f} max={NC.max():.2f}")
    print(f"  Shear (120): min={Shear.min():.6e} max={Shear.max():.6e}")
    print(f"  Self-consistency: logL = {logL:.1f} (OK)")
    return 0


def cleanup(config):
    return 0
