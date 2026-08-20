#!/usr/bin/env python
"""Build the Buzzard DV on the PIPELINE grid: comoving Mpc/h radii,
h Msun/pc^2 comoving amplitudes (issue #3 item 1; owner decision
2026-08-20: "we should match the pipeline grid").

Source of truth: Tan Xing's shipped products
(xtang126/mock_cluster_buzzard, output/MockDataVector.npz) -- NOT the
stale dv_buzzard_jkcov*.npz chain (its NC is an older revision, x1.186,
and its shear grid carried the physical-Mpc/h convention of her
targetR file).

Conversions, per z-bin with zbar = the bin centre of her seam-excluding
bins [0.20,0.33) [0.37,0.50) [0.50,0.65):

  radius    R_phys [Mpc]      = R_com [cMpc/h] / (1 + zbar) / h_buzz
  amplitude DSigma [h Msun/pc^2, comoving]
                              = h_buzz * DSigma_phys / (1 + zbar)^2

(the mock measures Sigma = mass / PHYSICAL area; comoving area is
(1+z)^2 larger, hence the (1+z)^-2; one power of h_buzz converts
Msun/pc^2 -> h Msun/pc^2 -- the amplitude half was the earlier "hfix").
Profiles are log-log cubic interpolated from her 11-point physical-Mpc
grid onto the pipeline r_perp grid (same interpolation her cell 68
used); the per-bin shear covariance is pushed through the SAME linear
operator, C' = (a J) C (a J)^T with J the interpolation matrix and a
the amplitude factor.

Uses the analytical (C1) route arrays -- the production choice -- and
her per-bin C19-stack covariance. NC and NC_cov are taken as shipped
(the Omega_buzz normalisation question is tracked separately in this
issue). Output follows the likelihood contract (data_NC (12,),
data_Shear (120,) bin-major, INVERSE covariances).

STATUS -- BLOCKED ON THE Sigma_crit BRIDGE (see issue #3): her npz
arrays are gamma_t (DIMENSIONLESS, ~0.04 at the innermost radius), not
DeltaSigma. The old dv_buzzard_jkcov chain multiplied by an off-repo
Sigma_crit-like factor (recovered by ratio: ~2917 / 2985 / 3601
Msun h/pc^2 for zbar = 0.265 / 0.435 / 0.575, but with 5-15% R-dependent
spread -- partly her array revisions since). Until the bridge is
decided (model-consistent 1/<Sigma_crit^-1> from
average_sigma_crit_inv's source n(z), or Tan Xing's own Sigma_crit),
the data_Shear written here is gamma_t * h/(1+z)^2 on the comoving
grid -- NOT yet in model units. The radius mapping, the (1+z)^-2 area
factor, the covariance propagation, and the NC refresh are final.

Run (local clone paths):
    python validations/build_buzzard_dv_comoving.py
"""
from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MOCK_NPZ = os.path.expanduser(
    "~/Documents/Dev/github/mock_cluster_buzzard/output/MockDataVector.npz")
OLD_DV = os.path.join(ROOT, "data", "mock", "dv_buzzard_jkcov_hfix.npz")
OUT = os.path.join(ROOT, "data", "mock", "dv_buzzard_comoving.npz")

H_BUZZ = 0.70                       # DeRose+2019 Table 1
# pipeline shear grid [cMpc/h] (mock_mcmc_cp_camb.ini r_perp)
R_COM = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                  1.19581, 1.70998, 2.44521, 3.49658, 5.00000])


def interp_matrix(x_src, x_tgt):
    """Rows of the log-log cubic interpolation as a linear operator.

    Cubic interpolation is linear in the samples, so the matrix is
    exact: column j = interpolation of the j-th unit vector. Built in
    log-log space like her cell 68 (log applied to the DATA too, which
    makes the operator nonlinear in the data -- for the covariance we
    use the LINEAR-space log-x cubic operator instead, the standard
    delta-method compromise; the difference is far below the jackknife
    noise)."""
    n_s, n_t = x_src.size, x_tgt.size
    J = np.empty((n_t, n_s))
    for j in range(n_s):
        e = np.zeros(n_s); e[j] = 1.0
        J[:, j] = interp1d(np.log(x_src), e, kind="cubic",
                           bounds_error=True)(np.log(x_tgt))
    return J


def main():
    d = np.load(MOCK_NPZ)
    r_phys = d["radii_phys_mpc"]                      # (11,) physical Mpc
    z_lo, z_hi = d["z_bin_min"], d["z_bin_max"]
    zbar = 0.5 * (z_lo + z_hi)                        # [0.265 0.435 0.575]
    gt = d["gamma_t_mock_obs_C1"]                     # (4, 3, 11) phys Msun/pc^2
    cov = d["shear_C19_stack_cov"]                    # (4, 3, 11, 11)
    nc = d["NC"]                                      # (4, 3)
    nc_cov = d["NC_cov"]                              # (12, 12)

    n_lam, n_z, n_r = gt.shape
    shear = np.empty((n_z, n_lam, R_COM.size))
    cov_blocks = np.zeros((n_z * n_lam * R_COM.size,) * 2)
    print(f"{'z bin':>14} {'zbar':>6} {'R factor':>9} {'amp factor':>11}")
    for iz in range(n_z):
        r_fac = 1.0 / (1.0 + zbar[iz]) / H_BUZZ      # cMpc/h -> phys Mpc
        amp = H_BUZZ / (1.0 + zbar[iz]) ** 2
        print(f"[{z_lo[iz]:.2f},{z_hi[iz]:.2f}) {zbar[iz]:>6.3f} "
              f"{r_fac:>9.4f} {amp:>11.4f}")
        r_tgt_phys = R_COM * r_fac
        assert r_tgt_phys.min() >= r_phys.min() and \
            r_tgt_phys.max() <= r_phys.max(), "target outside source grid"
        J = interp_matrix(r_phys, r_tgt_phys)
        for il in range(n_lam):
            prof = interp1d(np.log(r_phys), np.log(gt[il, iz]),
                            kind="cubic", bounds_error=True)(
                                np.log(r_tgt_phys))
            shear[iz, il] = amp * np.exp(prof)
            block = (amp * J) @ cov[il, iz] @ (amp * J).T
            i0 = (iz * n_lam + il) * R_COM.size
            cov_blocks[i0:i0 + R_COM.size, i0:i0 + R_COM.size] = block

    data_shear = shear.reshape(-1)                    # bin-major (z, lam, R)
    data_nc = nc.T.reshape(-1)                        # (z, lam) bin-major

    np.savez(OUT,
             data_NC=data_nc,
             data_Shear=data_shear,
             invcov_NC=np.linalg.inv(nc_cov),
             invcov_Shear=np.linalg.inv(cov_blocks),
             r_perp_cmpch=R_COM, zbar=zbar,
             provenance=np.array([
                 "xtang126/mock_cluster_buzzard output/MockDataVector.npz;"
                 " C1 route; R_com = (1+zbar) h R_phys;"
                 " DSigma_com_h = h*DSigma_phys/(1+zbar)^2;"
                 " built by validations/build_buzzard_dv_comoving.py"]))
    print(f"\nwrote {OUT}")

    # ---- comparison against the DV the fits have been using -------------
    if os.path.exists(OLD_DV):
        old = np.load(OLD_DV)
        rs = (data_shear / old["data_Shear"]).reshape(n_z, n_lam,
                                                      R_COM.size)
        print("\nnew/old shear ratio (the correction the fits were "
              "missing), lambda-averaged per radius:")
        for iz in range(n_z):
            row = rs[iz].mean(axis=0)
            print(f"  z[{z_lo[iz]:.2f},{z_hi[iz]:.2f}): "
                  + " ".join(f"{v:5.2f}" for v in row))
        rn = data_nc / old["data_NC"]
        print("new/old NC per bin:",
              " ".join(f"{v:.3f}" for v in rn))


if __name__ == "__main__":
    main()
