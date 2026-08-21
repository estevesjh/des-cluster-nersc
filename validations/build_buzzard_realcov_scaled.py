"""Scale the current Buzzard shear DV covariance to the REAL Buzzard-halo
covariance's fractional errors, at the current data's actual (little-h) units.

The gt_obs_c1 DV's C19 analytical stack covariance is unphysically tight (~2%).
The real Buzzard-halo covariance (dataVec_mock_May10th2023.npz, made from Buzzard
halos; used by the old y1_mock inis) has fractional errors ~6% at small/mid R
growing to ~10-16% at large R. We take THAT covariance's fractional error per
radius (mean over the 12 bins), interpolate onto the current 10-radius grid, and
multiply by the CURRENT data amplitude -> a realistic diagonal covariance in the
correct units. No data is invented; the fractional errors are the measured
Buzzard ones, the amplitude is the current data's.
"""
import numpy as np

OLD = "/global/cfs/cdirs/des/jesteves/data/buzzard/v1.9.8/y3_rm/dataVec_mock_May10th2023.npz"
CUR = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock/dv_buzzard_gt_obs_c1_comoving.npz"
OUT = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock/dv_buzzard_gt_obs_c1_realcov.npz"

# old Buzzard covariance -> fractional error per radius (20-radius grid)
old = np.load(OLD)
sig_old = np.sqrt(np.diag(np.linalg.inv(old["invcov_Shear"])))
r_old = np.array([0.30, 0.35, 0.40, 0.47, 0.54, 0.63, 0.73, 0.85, 0.98, 1.14,
                  1.32, 1.53, 1.77, 2.06, 2.38, 2.77, 3.21, 3.72, 4.31, 5.00])
frac_old = np.mean(np.abs(sig_old / old["data_Shear"]).reshape(12, 20), axis=0)

# current DV (10-radius grid), keep data + units; replace covariance
cur = np.load(CUR)
data = cur["data_Shear"]
R = cur["radii"]
frac = np.interp(np.log(R), np.log(r_old), frac_old)   # flat-extrapolate below 0.3
sig = np.tile(frac, 12) * np.abs(data)                 # real frac err x actual amplitude
invcov = 1.0 / sig ** 2

np.savez(OUT, data_NC=cur["data_NC"], invcov_NC=cur["invcov_NC"],
         data_Shear=data, invcov_Shear=invcov, radii=R,
         units=str(cur["units"]) if "units" in cur.files else "",
         cov_note=np.str_("frac err from dataVec_mock_May10th2023 (Buzzard halos) "
                          "interpolated to 10-radius grid x current data amplitude"))
print(f"wrote {OUT}")
print(f"  frac err (10-radius) [%]: {np.round(frac * 100, 1)}")
