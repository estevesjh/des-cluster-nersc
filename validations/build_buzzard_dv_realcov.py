"""Buzzard DV with REALISTIC covariance (Matteo NC + Y1-WL shear errors).

Motivation: the JK covariance is unrealistically tight (jackknife of the mock
stacked mean, no shape noise ~4%), which makes the fit model-error-dominated and
rails the posterior. Replace it:
  (a) NC   -> Matteo/Costanzi DES-Y1 abundance covariance
              (y1_rerun/data_files/Cov_ij_bestfit_DESY1_105.txt, top-left 12x12;
              ~1.3-1.6x looser than Poisson + sample-variance off-diagonals).
  (b) Shear-> DIAGONAL of the DES-Y1 WL covariance (wl_cov.txt), interpolated
              from its 15-radii grid to the pipeline 10 radii. The FULL wl_cov
              is near-singular (cond ~4e10) so we keep only its diagonal (the
              realistic shape-noise error envelope), which is well-conditioned.

Data (data_NC, data_Shear) are unchanged from dv_buzzard_jkcov_hfix.npz.
Output: data/mock/dv_buzzard_realcov.npz
"""
import numpy as np

ROOT = "/pscratch/sd/j/jesteves/github/des-cluster-nersc"
SRC = f"{ROOT}/data/mock/dv_buzzard_jkcov_hfix.npz"
NC_COV = f"{ROOT}/y1_rerun/data_files/Cov_ij_bestfit_DESY1_105.txt"
WL_COV = f"{ROOT}/y1_rerun/data_files/wl_cov.txt"
OUT = f"{ROOT}/data/mock/dv_buzzard_realcov.npz"

N_BINS, N_R = 12, 10
R_PIPE = np.geomspace(0.2, 5.0, N_R)
# wl_cov 15-radii grid (old r_perp, Mpc/h) — the grid wl_cov.txt is sampled on.
R_WL = np.array([0.0426, 0.0669, 0.1045, 0.1652, 0.2607, 0.4117, 0.6505,
                 1.0257, 1.6181, 2.5537, 4.0265, 6.3490, 10.0107, 15.7832, 24.8771])

d = np.load(SRC)
data_NC = np.asarray(d["data_NC"], float)
data_Shear = np.asarray(d["data_Shear"], float)

# (a) NC: Matteo Y1 abundance covariance, top-left 12x12 -> invert
Cnc = np.loadtxt(NC_COV)[:N_BINS, :N_BINS]
Cnc = 0.5 * (Cnc + Cnc.T)
invcov_NC = np.linalg.inv(Cnc)

# (b) Shear: diagonal of wl_cov, interpolated 15 -> 10 radii per bin (log-log)
wl = np.loadtxt(WL_COV)
sig_wl = np.sqrt(np.diag(wl)).reshape(N_BINS, 15)          # (12,15) DeltaSigma sigma
sig_10 = np.empty((N_BINS, N_R))
for b in range(N_BINS):
    sig_10[b] = np.exp(np.interp(np.log(R_PIPE), np.log(R_WL), np.log(sig_wl[b])))
sig_shear = sig_10.ravel()
invcov_Shear = np.diag(1.0 / sig_shear**2)

np.savez(OUT, data_NC=data_NC, data_Shear=data_Shear,
         invcov_NC=invcov_NC, invcov_Shear=invcov_Shear)

# ---- report + validate error scale ----
fracNC = np.sqrt(np.diag(Cnc)) / data_NC
fracSH = sig_shear / np.abs(data_Shear)
print(f"wrote {OUT}")
print(f"  NC   cov cond={np.linalg.cond(Cnc):.1e}  frac err {fracNC.min():.1%}-{fracNC.max():.1%} (median {np.median(fracNC):.1%})")
print(f"  Shear diag cov cond=1.0  frac err {fracSH.min():.1%}-{fracSH.max():.1%} (median {np.median(fracSH):.1%})")
print(f"  (was: JK shear ~4% median; Poisson NC ~ sqrt(N))")

# chi2 of the fiducial + old best-fit theory under the NEW cov
mk = np.load(f"{ROOT}/data/mock/mock_dv_widePlanck_jkcov.npz")
fidSH = mk["data_Shear"].astype(float); fidNC = mk["data_NC"].astype(float)
dNC, dSH = data_NC - fidNC, data_Shear - fidSH
print(f"\n  chi2(fiducial vs data) under realistic cov: "
      f"NC={dNC@invcov_NC@dNC:.0f}/12  Shear={dSH@invcov_Shear@dSH:.0f}/120  "
      f"total={dNC@invcov_NC@dNC + dSH@invcov_Shear@dSH:.0f}/132")
print(f"  (was ~19700/132 with the tight JK cov)")
