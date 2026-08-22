"""Buzzard DeltaSigma vs PIPELINE theory: fixed c + DENSITY-level (1+z)^3.

Option (b): the (1+z)^3 is applied at the DENSITY level (rho_m0 -> rho_m(z)) INSIDE
the 1-halo, via the new halo_model knob `one_halo_z_density`, so it propagates through
the NFW into DeltaSigma correctly (concentration stays fixed). Because the 1-halo
table is z-agnostic, we run the pipeline once per z-bin (one_halo_z_density = the bin
z) and stitch each z-bin's 1-halo from its matching run. The 2-halo + NC are z_density-
independent. Comoving (z_density=0) run overlaid for contrast.

theory DeltaSigma = shear1hmissel/vals / tile(NC,10) + shear_prj/cl   (little-h)
Data: dv_buzzard_deltasigma_heidi.npz (lambda_obs Heidi stack, pipeline units, JK cov).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = ("/tmp/claude-81868/-pscratch-sd-j-jesteves-github-des-cluster-nersc/"
        "66432ad4-7037-45cc-8a5a-4525036237fe/scratchpad")
COM = os.path.join(BASE, "buzz_theory_c125")                 # z_density=0 (comoving)
PHYS = {0: os.path.join(BASE, "buzz_1h_zd0.27"),             # z-bin -> physical run
        1: os.path.join(BASE, "buzz_1h_zd0.44"),
        2: os.path.join(BASE, "buzz_1h_zd0.57")}
DV = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock/dv_buzzard_deltasigma_heidi.npz"
LBD = [20, 30, 45, 60, 500]; ZC = [0.27, 0.44, 0.57]; RMIN = 0.4


def theory_of(sd):
    NC = np.loadtxt(f"{sd}/numcountssel/vals.txt").ravel()
    S1h = np.loadtxt(f"{sd}/shear1hmissel/vals.txt").ravel()
    Sprj = np.loadtxt(f"{sd}/shear_prj/cl.txt").ravel()
    return (S1h / np.repeat(NC, 10) + Sprj).reshape(12, 10)


def main():
    th_com = theory_of(COM)                                  # (12,10) comoving
    th_phys = np.zeros((12, 10))
    for iz in range(3):
        t = theory_of(PHYS[iz])
        th_phys[iz * 4:(iz + 1) * 4] = t[iz * 4:(iz + 1) * 4]   # stitch this z-bin

    d = np.load(DV, allow_pickle=True)
    R = d["radii"]; data = d["data_Shear"].reshape(12, 10); cov = d["cov_Shear"]
    keep = R > RMIN

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    chi2_p = chi2_c = 0.0; ndof = 0
    print("pipeline: fixed c (c_amp=1.25) + DENSITY (1+z)^3 vs comoving, vs Buzzard DV")
    for b in range(12):
        iz, il = b // 4, b % 4
        ax = axes[iz, il]; y = data[b]
        C = cov[b * 10:(b + 1) * 10, b * 10:(b + 1) * 10]
        idx = np.where(keep)[0]; iCk = np.linalg.inv(C[np.ix_(idx, idx)])
        rp = (y - th_phys[b])[keep]; rc = (y - th_com[b])[keep]
        chi2_p += rp @ iCk @ rp; chi2_c += rc @ iCk @ rc; ndof += keep.sum()
        thd_p = np.median((th_phys[b] / y)[keep]); thd_c = np.median((th_com[b] / y)[keep])
        ax.errorbar(R[keep], y[keep], yerr=np.sqrt(np.diag(C))[keep], fmt="o",
                    color="k", ms=4, capsize=2, label="Buzzard ΔΣ (JK)")
        ax.errorbar(R[~keep], y[~keep], yerr=np.sqrt(np.diag(C))[~keep], fmt="o",
                    mfc="none", mec="0.6", ms=4)
        ax.loglog(R, th_phys[b], "-", color="crimson", lw=2,
                  label="c1.25 + density (1+z)³" if b == 0 else None)
        ax.loglog(R, th_com[b], "--", color="royalblue", lw=1.3,
                  label="c1.25 comoving (z=0)" if b == 0 else None)
        ax.axvspan(R[0] * 0.8, RMIN, color="0.9", zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.text(0.04, 0.06, f"z {ZC[iz]:.2f}\nλ[{LBD[il]},{LBD[il+1] if LBD[il+1]<500 else 999})\n"
                f"th/dat={thd_p:.2f}\n(com {thd_c:.2f})", transform=ax.transAxes,
                fontsize=8, va="bottom")
        if iz == 2:
            ax.set_xlabel("R [Mpc/h]")
        if il == 0:
            ax.set_ylabel(r"$\Delta\Sigma$ [h M$_\odot$/pc$^2$]")
        if b == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Buzzard ΔΣ vs PIPELINE — fixed c (1.25) + DENSITY-level (1+z)³ in 1-halo "
                 f"(R>{RMIN}; JK cov)   χ²phys={chi2_p:.0f}/{ndof}  χ²com={chi2_c:.0f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ("/pscratch/sd/j/jesteves/github/des-cluster-nersc/chains/"
           "buzzard_vs_pipeline_physdens.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"chi2 physical-density = {chi2_p:.0f}/{ndof};  comoving = {chi2_c:.0f}/{ndof}")
    print("th/dat physical per bin:", np.round([np.median((th_phys[b]/data[b])[keep]) for b in range(12)],2))
    print("th/dat comoving per bin:", np.round([np.median((th_com[b]/data[b])[keep]) for b in range(12)],2))
    print("physical/comoving ratio per bin (the density (1+z)^3 effect on DS):",
          np.round([np.median((th_phys[b]/th_com[b])[keep]) for b in range(12)],3))
    print("wrote", out)


if __name__ == "__main__":
    main()
