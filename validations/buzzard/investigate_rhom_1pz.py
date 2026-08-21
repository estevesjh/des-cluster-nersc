"""Buzzard validation 4/4 (investigation): the rho_m,0 x (1+z) factor.

The pipeline 1-halo NFW is built with the COMOVING mean density rho_m0 (frozen
z=0; halo_model_cosmosis.py first_halo_term(z=0)). If the Buzzard DeltaSigma
instead follows the PHYSICAL density rho_m(z) = rho_m0 (1+z)^3, the pipeline
under-predicts at high z by a (1+z) power -- which is exactly the empirical
rho_m(z) (1+z) factor that flattens the Buzzard z-tilt (see
../buzzard_vs_theory_zsplit_conc.png and likelihood_cp shear_1pz_power).

This starter GROUNDS the convention question in the Buzzard halos themselves:
take the reported (M200m, rs) per (mass, z) bin and build the analytic NFW
DeltaSigma with rho_ref = rho_m0 (comoving) vs rho_m0 (1+z)^3 (physical). The
amplitude ratio between the two is the (1+z) factor the pipeline is missing.

NEXT STEP (tracked in the issue): confirm against the STACKED Buzzard DeltaSigma
per (mass, z) bin -- requires decoding the profile catalog's (3,15) DeltaSigma
grid (3 = ? , 15 radii) and its radial axis, then a per-bin NFW fit.
"""
import sys
import os
import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/src")
sys.path.insert(0, "/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev/y3_buzzard")
import fitsio
from fileLoc import FileLocs
from nfwModel import deltaSigmaNFW_Analytical

RHOC_LH = 2.77533742639e+11        # (Msun/h)/(Mpc/h)^3
OM = 0.286
R_TEST = 0.5                        # Mpc/h, representative 1-halo radius
Z_SPLIT = 0.33


def main():
    d = fitsio.read(FileLocs(machine="nersc").profile_output_fname)
    sel = ((d["pid"] == -1) & (d["cosi"] >= 0) & (d["cosi"] <= 1)
           & (d["redshift"] >= 0.2) & (d["redshift"] <= 0.65)
           & ((d["redshift"] < 0.33) | (d["redshift"] > 0.37))
           & (d["rs"] > 0) & (d["R200m"] > 0) & (d["M200m"] >= 1e13))
    M = d["M200m"][sel]
    z = d["redshift"][sel]
    c = d["R200m"][sel] / d["rs"][sel]

    print("(1+z) density-evolution factor from Buzzard (M200m, rs):")
    print("  DeltaSigma_NFW(R=0.5) built with rho_ref = rho_m0 (comoving, pipeline)")
    print("  vs rho_m0 (1+z)^3 (physical); ratio = the (1+z) factor the pipeline misses.\n")
    print(f"{'z-bin':>12}{'<z>':>7}{'<M200m>':>11}{'<c>':>6}{'dS(com)':>10}{'dS(phys)':>10}{'ratio':>8}{'(1+z)^n n=':>12}")
    for lab, m in [("z<0.33", z < Z_SPLIT), ("z>0.37", z > 0.37)]:
        zc = np.median(z[m]); Mc = np.median(M[m]); cc = np.median(c[m])
        d_com = deltaSigmaNFW_Analytical(np.array([R_TEST]), Mc, cc,
                                         rho_c=OM * RHOC_LH)[0] / 1e12
        d_phys = deltaSigmaNFW_Analytical(np.array([R_TEST]), Mc, cc,
                                          rho_c=OM * RHOC_LH * (1 + zc) ** 3)[0] / 1e12
        ratio = d_phys / d_com
        n = np.log(ratio) / np.log(1 + zc)
        print(f"{lab:>12}{zc:7.3f}{Mc:11.3e}{cc:6.2f}{d_com:10.3f}{d_phys:10.3f}"
              f"{ratio:8.3f}{n:12.2f}")
    print("\n=> if Buzzard DeltaSigma tracks the PHYSICAL rho_m(z), the pipeline "
          "(comoving rho_m0) is low by this ratio at each z -> the observed z-tilt.\n"
          "   The empirical fix (shear_1pz_power=1) applies (1+z)^1; the density "
          "argument above gives n~1.5-2 (partly offset by comoving-vs-physical\n"
          "   surface-density (1+z)^-2), so the net is ~(1+z)^~1. Confirm vs the "
          "stacked Buzzard DeltaSigma (see module docstring / issue).")


if __name__ == "__main__":
    main()
