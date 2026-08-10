"""Pre-flight visual check: Buzzard sim data vs current-pipeline fiducial theory.

Answers "does the cosmology fit look plausible before we launch a chain?"
- NC: 12 bins, Buzzard counts vs fiducial-HOD theory (same 12 bins -> chi2 valid).
- Shear: DeltaSigma(R). Buzzard is on 15 radii [0.0426..24.877]; the current
  pipeline fiducial is on 10 radii [0.2..5.0]. Both are DeltaSigma
  (M_sun h / pc^2), so we overlay by PHYSICAL R to judge amplitude+shape match.
  (A runnable Buzzard DV must be rebinned onto the 10-radii grid first; this
  plot shows whether that rebin will land on top of the model.)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Resolve paths relative to this file so the test runs on any checkout.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # validations/ -> repo root
ROOT = os.path.join(_REPO, "data", "mock")
OUT = os.path.join(_REPO, "buzzard_fit_check.png")

N_BINS = 12
R_BUZZ = np.array([0.0426, 0.0669, 0.1045, 0.1652, 0.2607, 0.4117, 0.6505,
                   1.0257, 1.6181, 2.5537, 4.0265, 6.3490, 10.0107, 15.7832,
                   24.8771])                       # 15 radii, buzzard native
R_FID = np.geomspace(0.2, 5.0, 10)                 # 10 radii, current pipeline
NR_B, NR_F = R_BUZZ.size, R_FID.size
LBD = ["20-30", "30-45", "45-60", ">60"]
ZED = ["0.2-0.35", "0.35-0.5", "0.5-0.65"]
R_LO, R_HI = 0.2, 5.0                               # current pipeline shear window


def sig(invcov):
    return np.sqrt(np.diag(np.linalg.inv(np.asarray(invcov, float))))


def main():
    bz = np.load(f"{ROOT}/mock_dv_buzzard.npz", allow_pickle=True)
    fd = np.load(f"{ROOT}/mock_dv_widePlanck_jkcov.npz", allow_pickle=True)

    NC_b, NC_f = np.asarray(bz["data_NC"], float), np.asarray(fd["data_NC"], float)
    sNC_b = sig(bz["invcov_NC"])
    SH_b = np.asarray(bz["data_Shear"], float)
    sSH_b = sig(bz["invcov_Shear"])
    SH_f = np.asarray(fd["data_Shear"], float)
    sSH_f = sig(fd["invcov_Shear"])

    # NC chi2 of fiducial theory vs Buzzard data (same 12 bins -> valid)
    dNC = NC_b - NC_f
    chi2_NC = float(dNC @ np.asarray(bz["invcov_NC"], float) @ dNC)
    print(f"NC: chi2(fiducial vs Buzzard) = {chi2_NC:.1f} for 12 bins")
    print("NC per-bin Buzzard/theory ratio:")
    for i in range(N_BINS):
        print(f"  bin{i:2d} ({LBD[i%4]:>5s}, z {ZED[i//4]:>9s}): "
              f"data={NC_b[i]:8.1f}  theory={NC_f[i]:8.1f}  ratio={NC_b[i]/NC_f[i]:.3f}")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.3, 1, 1, 1], hspace=0.42, wspace=0.30)

    # ---- NC: top row spanning all 4 cols ----
    axn = fig.add_subplot(gs[0, :])
    x = np.arange(N_BINS)
    axn.errorbar(x - 0.08, NC_b, yerr=sNC_b, fmt="o", color="#d1495b",
                 capsize=3, label="Buzzard data")
    axn.plot(x + 0.08, NC_f, "s", color="#30638e", label="fiducial theory")
    for i in range(N_BINS):
        axn.annotate(f"{NC_b[i]/NC_f[i]:.2f}", (x[i], max(NC_b[i], NC_f[i])),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axn.set_yscale("log")
    axn.set_xticks(x)
    axn.set_xticklabels([f"{LBD[i%4]}\nz{ZED[i//4]}" for i in range(N_BINS)], fontsize=7)
    axn.set_ylabel("number counts")
    axn.set_title(f"Number counts: Buzzard vs fiducial theory  "
                  f"(chi2={chi2_NC:.0f}/12; ratios = data/theory)", fontsize=11)
    axn.legend(loc="upper right")

    # ---- Shear: 3x4 grid, one panel per bin ----
    for b in range(N_BINS):
        r, c = 1 + b // 4, b % 4
        ax = fig.add_subplot(gs[r, c])
        slb = slice(b * NR_B, (b + 1) * NR_B)
        slf = slice(b * NR_F, (b + 1) * NR_F)
        ax.errorbar(R_BUZZ, SH_b[slb], yerr=sSH_b[slb], fmt="o", ms=4,
                    color="#d1495b", capsize=2, label="Buzzard (15 r)")
        ax.plot(R_FID, SH_f[slf], "s-", ms=4, lw=1.2, color="#30638e",
                label="fiducial (10 r)")
        ax.axvspan(R_LO, R_HI, color="0.85", alpha=0.4, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"lam {LBD[b % 4]}, z {ZED[b // 4]}", fontsize=8)
        if c == 0:
            ax.set_ylabel(r"$\Delta\Sigma$", fontsize=9)
        if r == 3:
            ax.set_xlabel("R [Mpc/h]", fontsize=9)
        if b == 0:
            ax.legend(fontsize=6, loc="lower left")

    fig.suptitle("Buzzard pre-flight fit check  (shaded = 0.2-5.0 Mpc/h pipeline window)",
                 fontsize=13, y=0.995)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
