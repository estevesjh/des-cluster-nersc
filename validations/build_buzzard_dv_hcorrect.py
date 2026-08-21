"""Correct h-unit conversion for the Buzzard shear data vector.

Supersedes build_buzzard_dv_hfix.py, which multiplied DeltaSigma by h (0.70) --
the WRONG direction. First-principles anchor (pipeline deltaSigmaNFW_Analytical,
same physical halo in little-h vs physical units) gives

    DeltaSigma_theory[h Msun/pc^2] / DeltaSigma_physical[Msun/pc^2] = 1/h = 1.4286

so the physical mock (xtang126) must be multiplied by 1/h (NOT h) to land in the
pipeline's little-h units. The old hfix (x h) is off by 1/h^2 = 2.041.

    data_Shear   *= 1/h          (physical Msun/pc^2 -> h Msun/pc^2)
    invcov_Shear *= h^2          (cov scales as DeltaSigma^2)
    NC unchanged (a count, h-dimensionless).
"""
import numpy as np

ROOT = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock"
SRC = f"{ROOT}/dv_buzzard_jkcov.npz"        # physical Msun/pc^2 source
OUT = f"{ROOT}/dv_buzzard_jkcov_hcorrect.npz"
H_BUZZ = 0.70
INV_H = 1.0 / H_BUZZ

d = np.load(SRC)
data_NC = d["data_NC"]                        # count, h-independent -> unchanged
invcov_NC = d["invcov_NC"]
data_Shear = d["data_Shear"] * INV_H          # physical -> little-h  (x 1/h)
invcov_Shear = d["invcov_Shear"] * H_BUZZ**2  # cov x 1/h^2 -> invcov x h^2

assert data_Shear.shape == (120,) and invcov_Shear.shape == (120, 120)
np.savez(OUT, data_NC=data_NC, data_Shear=data_Shear,
         invcov_NC=invcov_NC, invcov_Shear=invcov_Shear)
print(f"wrote {OUT}  (shear x {INV_H:.4f}=1/h, invcov x {H_BUZZ**2:.3f}=h^2)")
print(f"  vs hfix (x{H_BUZZ}): correct data is {INV_H/H_BUZZ:.3f}x=1/h^2 higher")
