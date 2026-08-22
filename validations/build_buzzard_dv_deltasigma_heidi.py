"""Buzzard DeltaSigma data vector (lambda_obs bins) via Heidi's mass-matched
stacking, in the PIPELINE units, with a spatial-jackknife DeltaSigma covariance.

Follows xtang126/mock_cluster_buzzard/MockDataVector.ipynb, BUT keeps DeltaSigma
(NOT gamma_t -- no Sigma_crit multiply anywhere) and writes it in the pipeline's
little-h convention so it can be compared to the pipeline DeltaSigma theory directly.

Route: EMPIRICAL / Heidi. For each (lambda_obs, z) bin the DeltaSigma is the
(log10 M, z)-matched stack of ALL good halos re-weighted to the selected sample's
(M,z) distribution (`stacked_profile_weighted_by_mass_redshift`). lambda_obs is the
C19 HOD forward model (Costanzi: HOD + intrinsic + projection scatter, NO LSS/
orientation correlation), so this matched stack equals the naive lambda_obs stack
(B_sel~1) -- the clean, selection-unbiased signal.

================================ UNITS LEDGER ================================
 measured DeltaSigma : physical Msun/pc^2  on R_SIG [physical Mpc] (15-pt catalog)
 pipeline convention (build_buzzard_dv_gt_obs_c1.py, anchored to analytic NFW):
   radii     R_pipe = R_SIG * h            [physical Mpc/h]   (x h only; NO (1+z))
   amplitude DS_pipe = DS_phys * (1/h)     [little-h h Msun/pc^2]
 regrid (log-log) onto R_GRID = geomspace(0.2, 5, 10) [Mpc/h]
 -> data_Shear holds DeltaSigma (little-h), NOT gamma_t.
 Bins reordered lambda-major (xtang) -> z-major (pipeline bin = z*4 + lambda).
=============================================================================
"""
import os
import warnings
import numpy as np
import sys
sys.path.insert(0, "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/src")
import fitsio
from fileLoc import FileLocs
from costanzi_selection import (sample_lambda_true, sample_lambda_obs,
                                load_prj_posterior_mean)
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

H = 0.70                                                   # DeRose+2019
ZMIN, ZMAX, LOGM_MIN = 0.20, 0.65, 13.0
LBDBINS = np.array([20, 30, 45, 60, 500])
ZMIN_LIST = np.array([0.20, 0.37, 0.50])
ZMAX_LIST = np.array([0.33, 0.50, 0.65])
DM, DZ = 0.1, 0.05                                         # mass-match grid (log10 M, z)
K_JK = 50
PRJ_FILE = ("/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/data/"
            "prj_params_DESY3_lss_lin_dep_getdist_v1.txt")
OUT = ("/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock/"
       "dv_buzzard_deltasigma_heidi.npz")

# radial grids (physical Mpc for the catalog; Mpc/h for the pipeline)
_e2 = np.exp(np.linspace(np.log(0.0323), np.log(30.0), 16))
R_SIG = np.sqrt(_e2[:-1] * _e2[1:])                       # 15-pt physical Mpc
R_GRID = np.geomspace(0.2, 5.0, 10)                       # pipeline grid, Mpc/h


def regrid(ds_phys):
    """physical DeltaSigma(R_SIG) -> little-h DeltaSigma(R_GRID) [pipeline units]."""
    R_src = R_SIG * H                                     # physical Mpc/h
    ds_h = ds_phys * (1.0 / H)                            # little-h amplitude
    good = np.isfinite(ds_h) & (ds_h > 0)
    return np.exp(np.interp(np.log(R_GRID), np.log(R_src[good]), np.log(ds_h[good])))


def main():
    fl = FileLocs(machine="nersc")
    p = fitsio.read(fl.profile_output_fname)
    hcat = fitsio.read(fl.halo_run_fname)                 # row-matched (verified)
    good = ((p["pid"] == -1) & (p["cosi"] >= 0) & (p["cosi"] <= 1)
            & ((p["redshift"] < 0.33) | (p["redshift"] > 0.37))
            & (p["redshift"] >= ZMIN) & (p["redshift"] <= ZMAX)
            & (np.log10(p["Mvir"]) >= LOGM_MIN))
    Mvir = p["Mvir"][good].astype(float)
    lgM = np.log10(Mvir)
    z = p["redshift"][good].astype(float)
    DS = np.asarray(p["DeltaSigma"])[good]                # (N,15) physical Msun/pc^2
    ra, dec = hcat["RA"][good].astype(float), hcat["DEC"][good].astype(float)
    N = good.sum()
    print(f"{N} good halos (z[{ZMIN},{ZMAX}], seam-excised, logM>={LOGM_MIN})")

    # --- C19 HOD lambda_obs (no LSS correlation) ---
    rng = np.random.default_rng(42)
    prj = load_prj_posterior_mean(PRJ_FILE)
    ltrue = sample_lambda_true(Mvir, z, rng=rng).astype(float)
    lobs, *_ = sample_lambda_obs(ltrue, z, prj, rng=rng)

    # --- (log10 M, z) mass-match cells; spatial-jackknife patches ---
    mb = np.arange(lgM.min() - DM, lgM.max() + 2 * DM, DM)
    zb = np.arange(z.min() - DZ, z.max() + 2 * DZ, DZ)
    im = np.clip(np.digitize(lgM, mb) - 1, 0, len(mb) - 2)
    iz = np.clip(np.digitize(z, zb) - 1, 0, len(zb) - 2)
    ncell = (len(mb) - 1) * (len(zb) - 1)
    cell = im * (len(zb) - 1) + iz                        # (N,)
    patch = KMeans(n_clusters=K_JK, random_state=0, n_init=1).fit_predict(
        np.column_stack([ra, dec]))
    print(f"mass-match cells={ncell}, K_JK={K_JK}")

    nR = R_SIG.size
    # reference-pool per-cell sums/counts (full and per-patch) via np.add.at
    csum = np.zeros((ncell, nR)); ccnt = np.zeros(ncell)
    csum_p = np.zeros((ncell, K_JK, nR)); ccnt_p = np.zeros((ncell, K_JK))
    np.add.at(csum, cell, DS)
    np.add.at(ccnt, cell, 1.0)
    np.add.at(csum_p, (cell, patch), DS)
    np.add.at(ccnt_p, (cell, patch), 1.0)

    def matched(Nsel_cell):
        """Heidi mass-matched stack from cell weights + full-pool cell means."""
        w = Nsel_cell
        cm = np.where(ccnt[:, None] > 0, csum / np.maximum(ccnt[:, None], 1), 0.0)
        return (w[:, None] * cm).sum(0) / w.sum()

    def matched_jk(Nsel_cell, Nsel_cell_p, k):
        """same, leaving out patch k from BOTH weights and reference cell-means."""
        cnt_k = ccnt - ccnt_p[:, k]
        cm = np.where(cnt_k[:, None] > 0,
                      (csum - csum_p[:, k]) / np.maximum(cnt_k[:, None], 1), 0.0)
        w = Nsel_cell - Nsel_cell_p[:, k]
        return (w[:, None] * cm).sum(0) / w.sum()

    nl, nz = len(LBDBINS) - 1, len(ZMIN_LIST)
    DS_lz = np.zeros((nl, nz, 10))                        # data vector (pipeline units)
    cov_lz = np.zeros((nl, nz, 10, 10))
    NC = np.zeros((nl, nz))
    for il in range(nl):
        for j in range(nz):
            sel = ((lobs >= LBDBINS[il]) & (lobs < LBDBINS[il + 1])
                   & (z >= ZMIN_LIST[j]) & (z < ZMAX_LIST[j]))
            NC[il, j] = sel.sum()
            Nsel = np.zeros(ncell); np.add.at(Nsel, cell[sel], 1.0)
            Nsel_p = np.zeros((ncell, K_JK))
            np.add.at(Nsel_p, (cell[sel], patch[sel]), 1.0)
            DS_lz[il, j] = regrid(matched(Nsel))          # full mass-matched stack
            reps = np.array([regrid(matched_jk(Nsel, Nsel_p, k)) for k in range(K_JK)])
            drep = reps - reps.mean(0)
            cov_lz[il, j] = (K_JK - 1) / K_JK * drep.T @ drep
            print(f"  lam[{LBDBINS[il]:>3},{LBDBINS[il+1]:>3}) z[{ZMIN_LIST[j]:.2f},"
                  f"{ZMAX_LIST[j]:.2f}) N={int(NC[il,j]):5d}  DS[0]={DS_lz[il,j,0]:7.2f}"
                  f"  fracerr[0]={np.sqrt(cov_lz[il,j,0,0])/DS_lz[il,j,0]:.3f}")

    # --- lambda-major -> z-major (pipeline bin = z*4 + lambda) ---
    zm = lambda A: np.transpose(A, (1, 0) + tuple(range(2, A.ndim))).reshape(
        nl * nz, *A.shape[2:])
    data_Shear = zm(DS_lz).reshape(nl * nz, 10)          # (12,10)
    cov_blocks = zm(cov_lz)                               # (12,10,10)
    data_NC = zm(NC).ravel()                              # (12,)

    # block-diagonal 120x120 cov + inverse
    cov_Shear = np.zeros((120, 120))
    invcov_Shear = np.zeros((120, 120))
    for b in range(12):
        cov_Shear[b * 10:(b + 1) * 10, b * 10:(b + 1) * 10] = cov_blocks[b]
        invcov_Shear[b * 10:(b + 1) * 10, b * 10:(b + 1) * 10] = np.linalg.inv(cov_blocks[b])
    invcov_NC = np.diag(1.0 / np.maximum(data_NC, 1.0))   # Poisson diagonal

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT,
             data_Shear=data_Shear.ravel().astype(float),   # DeltaSigma, little-h
             cov_Shear=cov_Shear.astype(float),
             invcov_Shear=invcov_Shear.astype(float),
             data_NC=data_NC.astype(float),
             invcov_NC=invcov_NC.astype(float),
             radii=R_GRID.astype(float), data_h=np.float64(H),
             lambda_bins=LBDBINS.astype(float),
             z_bin_min=ZMIN_LIST.astype(float), z_bin_max=ZMAX_LIST.astype(float),
             units=np.str_("DeltaSigma [little-h h Msun/pc^2], radii [physical Mpc/h]; "
                           "Heidi (logM,z)-matched stack of lambda_obs bins; JK cov K=50"))
    print(f"\nwrote {OUT}")
    print(f"  data_Shear = DeltaSigma (little-h), NOT gamma_t")
    print(f"  data_NC (z-major): {np.round(data_NC,0)}")
    print(f"  radii [Mpc/h]: {np.round(R_GRID,3)}")


if __name__ == "__main__":
    main()
