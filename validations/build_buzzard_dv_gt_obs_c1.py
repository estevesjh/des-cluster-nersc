"""Re-make Tan Xing's Buzzard cluster DV (gamma_t_obs_C1 route) directly in
the pipeline COMOVING little-h convention, on the pipeline shear grid
r_grid = geomspace(0.2, 5, 10) [comoving Mpc/h]. The likelihood then reads it
as-is (data_units = comoving); no conversion happens at likelihood time.

Source : mock_cluster_buzzard/output/MockDataVector_scinv.npz
  gamma_t_obs_C1 (4,3,11)  = analytical C19 stack x B_sel_C1 (dimensionless)
  B_sel_C1, shear_C19_stack_cov, scinv_bin, z_rep, NC, NC_cov, radii_phys

Per (lambda,z) bin, with z = z_rep (numbercount-kernel bin mean), h = 0.70:
  DeltaSigma_phys = gamma_t_obs_C1 / scinv_bin                 [physical Msun/pc^2]
  R_comoving[Mpc/h]      = radii_phys[Mpc] * (1+z) * h
  DeltaSigma_comoving_h  = DeltaSigma_phys * h^2 / (1+z)^2     [comoving h Msun/pc^2]
     h^2   : pipeline Sigma convention (haloModel rhoc0 = 2.775e11 Msun/Mpc^3/h^2)
     (1+z)^-2 : physical -> comoving surface density (mass / comoving area)
The comoving DeltaSigma is regridded (log-log) onto r_grid ONCE, here; the
block-diagonal covariance is amplitude-scaled and regridded with the same
linear interp matrix (W C W^T), stacked, and inverted.

Bins reordered lambda-major (Tan Xing) -> z-major (pipeline bin = z*4 + lambda).
"""
import os
import numpy as np

MC = "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/output/MockDataVector_scinv.npz"
OUT = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock/dv_buzzard_gt_obs_c1_comoving.npz"
H = 0.70
N_L, N_Z, N_R = 4, 3, 11
R_GRID = np.geomspace(0.2, 5.0, 10)          # comoving Mpc/h, pipeline shear grid

d = np.load(MC)
gt, Bsel = d["gamma_t_obs_C1"], d["B_sel_C1"]          # (4,3,11)
covg = d["shear_C19_stack_cov"]                        # (4,3,11,11)
scinv, zrep = d["scinv_bin"], d["z_rep"]               # (4,3)
NC, NC_cov = d["NC"], d["NC_cov"]                       # (4,3), (12,12) lambda-major
R_phys = d["radii_phys_mpc"]                            # (11,) physical Mpc

DS = gt / scinv[:, :, None]                             # physical Msun/pc^2
cov_obs = covg * Bsel[:, :, :, None] * Bsel[:, :, None, :]
cov_DS = cov_obs / (scinv[:, :, None, None] ** 2)      # physical

zm = lambda A: np.transpose(A, (1, 0) + tuple(range(2, A.ndim))).reshape(N_L * N_Z, *A.shape[2:])
DS, cov_DS, zbin = zm(DS), zm(cov_DS), zm(zrep)         # (12,11),(12,11,11),(12,)

lng = np.log(R_GRID)
data_Shear = np.empty((12, 10))
var_Shear = np.empty((12, 10))     # DIAGONAL cov: regridding the full jackknife
# matrix (W C W^T) is ill-conditioned (interp correlates adjacent r_grid points
# -> cond ~1e20); regridding the variances only is stable. The C19 raw 11x11
# per bin is full-rank (cond ~5e3) but its off-diagonals don't survive the
# 11->10 regrid, so we keep the diagonal (per-radius errors).
for b in range(12):
    z = zbin[b]
    R_com = R_phys * H                                 # physical Mpc -> physical Mpc/h (x h;
    #   pipeline r_perp is physical Mpc/h: R_mis = theta*D_A, D_A physical; matches
    #   xtang126 targetR build R_src = radii_phys*0.70. No (1+z): NOT comoving.)
    amp = 1.0 / H                                       # physical Msun/pc^2 -> little-h h Msun/pc^2
    #   (x 1/h): anchored to analytic NFW -- pipeline 1-halo = little-h NFW to 1%, and
    #   DeltaSigma_littleh / DeltaSigma_physical = 1/h. (NOT x h^2; that was wrong by 1/h^3.)
    lnR = np.log(R_com)
    data_Shear[b] = np.exp(np.interp(lng, lnR, np.log(DS[b] * amp)))
    var_phys = np.diag(cov_DS[b])                       # (11,)
    var_Shear[b] = np.interp(lng, lnR, var_phys) * amp * amp

# NC: reorder lambda-major -> z-major
perm = np.array([(p % 4) * N_Z + (p // 4) for p in range(12)])
data_NC = zm(NC)
invcov_NC = np.linalg.inv(NC_cov[np.ix_(perm, perm)])
invcov_Shear = (1.0 / var_Shear.ravel())               # 1-D diagonal invcov

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, data_NC=data_NC.astype(float), invcov_NC=invcov_NC.astype(float),
         data_Shear=data_Shear.ravel().astype(float), invcov_Shear=invcov_Shear.astype(float),
         radii=R_GRID.astype(float), z_bin=zbin.astype(float), data_h=np.float64(H),
         units=np.str_("comoving little-h: DeltaSigma [h Msun/pc^2], radii [comoving Mpc/h]"))
print(f"wrote {OUT}")
print(f"  data_NC (z-major): {np.round(data_NC,0)}")
print(f"  data_Shear comoving, bin0 (lam20-30 z0.2-0.33): {np.round(data_Shear[0],2)}")
print(f"  r_grid: {np.round(R_GRID,3)}")
print(f"  invcov_Shear: 1-D diagonal, size {invcov_Shear.size}, all finite={np.all(np.isfinite(invcov_Shear))}")
