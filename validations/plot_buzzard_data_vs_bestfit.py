"""Data vs best-fit diagnostic for the h-fixed Buzzard run.

Overlays the Buzzard data vector (dv_buzzard_jkcov_hfix.npz) against the pipeline
theory evaluated at the max-likelihood point of the h-fixed chain (job 56669090).
Shows NC + DeltaSigma(R) per bin with covariance error bars and the chi2, so the
residual misfit (radial tilt in shear, z-trend in NC) is visible directly.

Theory assembly (DV contract): NC = numcountssel/vals;
Shear = shear1hmissel/vals / repeat(NC,10) + shear_prj/vals.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock"
BF = "/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_hfix/bestfit_test"
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(_HERE), "chains", "buzzard_hfix_data_vs_bestfit.png")

N_BINS, N_R = 12, 10
RADII = np.geomspace(0.2, 5.0, N_R)
LBD = ["20-30", "30-45", "45-60", ">60"]
ZED = ["0.2-0.35", "0.35-0.5", "0.5-0.65"]


def col(d, sec):
    return np.loadtxt(f"{d}/{sec}/vals.txt", comments='#').ravel()


def main():
    dv = np.load(f"{ROOT}/dv_buzzard_jkcov_hfix.npz")
    dNC, dSH = dv["data_NC"].astype(float), dv["data_Shear"].astype(float)
    icNC, icSH = dv["invcov_NC"].astype(float), dv["invcov_Shear"].astype(float)
    sNC = np.sqrt(np.diag(np.linalg.inv(icNC)))
    sSH = np.sqrt(np.diag(np.linalg.inv(icSH)))

    # best-fit theory
    tNC = col(BF, "numcountssel")
    tSH = col(BF, "shear1hmissel") / np.repeat(tNC, N_R) + col(BF, "shear_prj")

    chi2_NC = float((dNC - tNC) @ icNC @ (dNC - tNC))
    chi2_SH = float((dSH - tSH) @ icSH @ (dSH - tSH))
    print(f"chi2  NC={chi2_NC:.1f}/12   Shear={chi2_SH:.1f}/120   total={chi2_NC+chi2_SH:.1f}/132")

    dg, tg, sg = dSH.reshape(N_BINS, N_R), tSH.reshape(N_BINS, N_R), sSH.reshape(N_BINS, N_R)

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(5, 4, height_ratios=[1.4, 1, 1, 1, 0.9], hspace=0.5, wspace=0.32)

    # NC (top, spanning)
    axn = fig.add_subplot(gs[0, :])
    x = np.arange(N_BINS)
    axn.errorbar(x - 0.08, dNC, yerr=sNC, fmt="o", color="#d1495b", capsize=3, label="Buzzard data")
    axn.plot(x + 0.08, tNC, "s", color="#30638e", ms=7, label="best-fit theory")
    for i in range(N_BINS):
        axn.annotate(f"{dNC[i]/tNC[i]:.2f}", (x[i], max(dNC[i], tNC[i])),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axn.set_yscale("log"); axn.set_xticks(x)
    axn.set_xticklabels([f"{LBD[i%4]}\nz{ZED[i//4]}" for i in range(N_BINS)], fontsize=7)
    axn.set_ylabel("number counts")
    axn.set_title(f"NC: data vs best-fit  (chi2={chi2_NC:.0f}/12; labels=data/theory)", fontsize=11)
    axn.legend(loc="upper right")

    # shear per bin
    for b in range(N_BINS):
        ax = fig.add_subplot(gs[1 + b // 4, b % 4])
        ax.errorbar(RADII, dg[b], yerr=sg[b], fmt="o", ms=3, color="#d1495b", capsize=2, label="data")
        ax.plot(RADII, tg[b], "s-", ms=3, lw=1.2, color="#30638e", label="best-fit")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"lam {LBD[b % 4]}, z {ZED[b // 4]}", fontsize=8)
        if b % 4 == 0: ax.set_ylabel(r"$\Delta\Sigma$", fontsize=9)
        if b // 4 == 2: ax.set_xlabel("R [Mpc/h]", fontsize=9)
        if b == 0: ax.legend(fontsize=7, loc="lower left")

    # residual (data-theory)/sigma for shear
    axr = fig.add_subplot(gs[4, :])
    pull = ((dSH - tSH) / sSH).reshape(N_BINS, N_R)
    for b in range(N_BINS):
        axr.plot(RADII, pull[b], "o-", ms=2, lw=0.6, alpha=0.6)
    axr.axhline(0, color="k", lw=1); axr.axhline(2, color="0.6", ls=":"); axr.axhline(-2, color="0.6", ls=":")
    axr.set_xscale("log"); axr.set_xlabel("R [Mpc/h]"); axr.set_ylabel(r"(data$-$fit)/$\sigma$")
    axr.set_title("shear pull per bin (residual radial structure = leftover misspecification)", fontsize=10)

    fig.suptitle(f"Buzzard h-fixed: data vs best-fit (job 56669090)  "
                 f"chi2={chi2_NC+chi2_SH:.0f}/132 = {(chi2_NC+chi2_SH)/132:.1f}/dof", fontsize=13, y=0.995)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
