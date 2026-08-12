"""STOPGAP: h-unit fix for the Buzzard shear data vector.

The mock (xtang126/mock_cluster_buzzard) stores DeltaSigma in PHYSICAL
M_sun/pc^2, but the pipeline model emits DeltaSigma in little-h units
(h*M_sun/pc^2, comoving Mpc/h) -- verified by tracing the C++ NFW/2h
normalization (both terms are h*M_sun/pc^2; the "M_sun/h/pc^2" code comments
are wrong). The harvester (build_buzzard_datavector.py) converts the radii at
h_buzz=0.7 but leaves the amplitude physical, so the stored shear is off from
the model by ONE power of h -> the fit absorbs it along Omega_m*h ~ const and
rails h to the prior floor.

Fix: multiply the shear DeltaSigma by h_buzz (one power) to put it in the
model's little-h units; scale invcov_Shear by 1/h_buzz^2. NC is a count
(h-dimensionless) -> left unchanged (its z-dependent deficit is a separate,
real effect). This is a STOPGAP until Tan Xing ships the mock in little-h units.

Empirical check: this reduces the median data/model shear offset from ~1.58x
toward ~1.1x (residual = real profile/abundance differences + radial tilt).
"""
import numpy as np

ROOT = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock"
SRC = f"{ROOT}/dv_buzzard_jkcov.npz"
OUT = f"{ROOT}/dv_buzzard_jkcov_hfix.npz"
H_BUZZ = 0.70

d = np.load(SRC)
data_NC = d["data_NC"]                       # count, h-independent -> unchanged
invcov_NC = d["invcov_NC"]                    # unchanged
data_Shear = d["data_Shear"] * H_BUZZ         # physical M_sun/pc^2 -> h*M_sun/pc^2
invcov_Shear = d["invcov_Shear"] / H_BUZZ**2  # cov scales by h^2 -> invcov by 1/h^2

assert data_Shear.shape == (120,) and invcov_Shear.shape == (120, 120)
np.savez(OUT, data_NC=data_NC, data_Shear=data_Shear,
         invcov_NC=invcov_NC, invcov_Shear=invcov_Shear)

# report the offset before/after vs fiducial theory
fid = np.load(f"{ROOT}/mock_dv_widePlanck_jkcov.npz")
T = fid["data_Shear"].astype(float)
r0 = np.median(T / d["data_Shear"].astype(float))
r1 = np.median(T / data_Shear.astype(float))
print(f"wrote {OUT}  (shear x {H_BUZZ}, invcov / {H_BUZZ**2:.3f})")
print(f"median(theory/data) shear:  before={r0:.3f}  after={r1:.3f}  (target ~1.0)")
print(f"NC left unchanged (count, h-independent)")
