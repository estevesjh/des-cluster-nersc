"""Build a RUNNABLE Buzzard data vector on the current 10-radii pipeline grid.

The harvested Buzzard DV (mock_dv_buzzard.npz) is on 15 radii [0.0426..24.877];
the active pipeline (mock_mcmc_cp_camb.ini + likelihood_cp.py) expects 10 radii
[0.2..5.0] = 120-vector. Per data/README_FOR_ARWA.md:
  - interpolate the sim shear onto the pipeline's 10 radii (log-log),
  - REUSE the mock covariance (invcov_NC, invcov_Shear) from the jkcov file.

Output: data/mock/dv_buzzard_jkcov.npz  (data_NC(12), data_Shear(120),
invcov_NC(12,12), invcov_Shear(120,120)).
"""
import numpy as np

ROOT = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock"
OUT = f"{ROOT}/dv_buzzard_jkcov.npz"

N_BINS = 12
R_BUZZ = np.array([0.0426, 0.0669, 0.1045, 0.1652, 0.2607, 0.4117, 0.6505,
                   1.0257, 1.6181, 2.5537, 4.0265, 6.3490, 10.0107, 15.7832,
                   24.8771])
R_FID = np.geomspace(0.2, 5.0, 10)
NR_B, NR_F = R_BUZZ.size, R_FID.size


def main():
    bz = np.load(f"{ROOT}/mock_dv_buzzard.npz", allow_pickle=True)
    mk = np.load(f"{ROOT}/mock_dv_widePlanck_jkcov.npz", allow_pickle=True)

    NC = np.asarray(bz["data_NC"], float)                 # (12,) keep Buzzard NC
    SH_b = np.asarray(bz["data_Shear"], float)            # (180,)

    # log-log interpolate each bin's DeltaSigma(R) onto the 10 pipeline radii
    lRb, lRf = np.log(R_BUZZ), np.log(R_FID)
    SH_f = np.empty(N_BINS * NR_F)
    for b in range(N_BINS):
        y = SH_b[b * NR_B:(b + 1) * NR_B]
        SH_f[b * NR_F:(b + 1) * NR_F] = np.exp(np.interp(lRf, lRb, np.log(y)))

    invNC = np.asarray(mk["invcov_NC"], float)            # reuse mock cov
    invSH = np.asarray(mk["invcov_Shear"], float)

    assert NC.shape == (12,) and SH_f.shape == (120,)
    assert invNC.shape == (12, 12) and invSH.shape == (120, 120)

    np.savez(OUT, data_NC=NC, data_Shear=SH_f,
             invcov_NC=invNC, invcov_Shear=invSH)

    # report: SNR of the Buzzard vector, and chi2 of fiducial theory vs it
    fidNC = np.asarray(mk["data_NC"], float)
    fidSH = np.asarray(mk["data_Shear"], float)
    dNC, dSH = NC - fidNC, SH_f - fidSH
    print(f"wrote {OUT}")
    print(f"  Buzzard NC    SNR = {np.sqrt(NC @ invNC @ NC):.1f}")
    print(f"  Buzzard Shear SNR = {np.sqrt(SH_f @ invSH @ SH_f):.1f}  (n=120)")
    print(f"  chi2(fiducial vs Buzzard):  NC = {dNC @ invNC @ dNC:8.1f}/12")
    print(f"                              Shear = {dSH @ invSH @ dSH:8.1f}/120")
    print(f"                              TOTAL = {dNC @ invNC @ dNC + dSH @ invSH @ dSH:8.1f}/132")


if __name__ == "__main__":
    main()
