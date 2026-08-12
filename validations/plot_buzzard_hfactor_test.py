"""h-factor test: is the Buzzard-vs-model shear offset a single power of h?

Overlays the Buzzard data DeltaSigma (dv_buzzard_jkcov.npz) against the
fiducial-cosmology theory (mock_dv_widePlanck_jkcov.npz), finds the single
multiplicative constant c that best collapses data onto theory
(c = median(theory/data)), and checks whether c ~ h (0.68-0.70). If one
constant collapses ALL bins/radii, the offset is a pure amplitude h-factor
(the little-h unit mismatch in build_buzzard_datavector.py); a radial tilt
would mean an additional radial-h term.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock"
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(_HERE), "chains", "buzzard_hfactor_test.png")

N_BINS, N_R = 12, 10
RADII = np.geomspace(0.2, 5.0, N_R)
LBD = ["20-30", "30-45", "45-60", ">60"]
ZED = ["0.2-0.35", "0.35-0.5", "0.5-0.65"]
H_FID, H_BUZZ = 0.6766, 0.700


def main():
    dat = np.load(f"{ROOT}/dv_buzzard_jkcov.npz")
    fid = np.load(f"{ROOT}/mock_dv_widePlanck_jkcov.npz")
    D = dat["data_Shear"].astype(float)                       # Buzzard data (120)
    T = fid["data_Shear"].astype(float)                       # fiducial theory (120)
    sD = np.sqrt(np.diag(np.linalg.inv(dat["invcov_Shear"].astype(float))))

    r = T / D                                                 # theory/data per point
    c = np.median(r)                                          # best single collapse factor
    print(f"median(theory/data) = {c:.4f}   (h_fid={H_FID}, h_buzz={H_BUZZ})")
    print(f"  -> data*c overlays theory; c compared to h: c/h_fid={c/H_FID:.3f}, c/h_buzz={c/H_BUZZ:.3f}")
    print(f"ratio spread (theory/data): 16/50/84 pct = "
          f"{np.percentile(r,16):.3f}/{np.percentile(r,50):.3f}/{np.percentile(r,84):.3f}")

    Dg, Tg, sg = D.reshape(N_BINS, N_R), T.reshape(N_BINS, N_R), sD.reshape(N_BINS, N_R)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.32)

    # 12 per-bin overlays: theory, raw data, data*c
    for b in range(N_BINS):
        ax = fig.add_subplot(gs[b // 4 + (1 if b // 4 >= 0 else 0), b % 4]) if False else fig.add_subplot(gs[b // 4, b % 4])
        ax.plot(RADII, Tg[b], "s-", ms=3, lw=1.2, color="#30638e", label="fiducial theory")
        ax.errorbar(RADII, Dg[b], yerr=sg[b], fmt="o", ms=3, color="#d1495b",
                    capsize=2, alpha=0.8, label="Buzzard data")
        ax.plot(RADII, Dg[b] * c, "^", ms=4, color="#2a9d3a", label=f"data x {c:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"lam {LBD[b % 4]}, z {ZED[b // 4]}", fontsize=8)
        if b % 4 == 0: ax.set_ylabel(r"$\Delta\Sigma$", fontsize=9)
        if b // 4 == 2: ax.set_xlabel("R [Mpc/h]", fontsize=9)
        if b == 0: ax.legend(fontsize=6, loc="lower left")

    # bottom row: ratio before/after collapse
    axr = fig.add_subplot(gs[3, :2])
    for b in range(N_BINS):
        axr.plot(RADII, Dg[b] / Tg[b], "o-", ms=2, lw=0.6, alpha=0.6)
    axr.axhline(1 / c, color="k", ls="--", lw=1.5, label=f"median = {1/c:.2f} = 1/{c:.2f}")
    axr.axhline(1 / H_BUZZ, color="orange", ls=":", lw=1.5, label=f"1/h_buzz = {1/H_BUZZ:.2f}")
    axr.axhline(1.0, color="0.5", lw=1)
    axr.set_xscale("log"); axr.set_title("data/theory (raw) — coherent offset?", fontsize=10)
    axr.set_xlabel("R [Mpc/h]"); axr.set_ylabel("data/theory"); axr.legend(fontsize=8)

    axr2 = fig.add_subplot(gs[3, 2:])
    for b in range(N_BINS):
        axr2.plot(RADII, (Dg[b] * c) / Tg[b], "o-", ms=2, lw=0.6, alpha=0.6)
    axr2.axhline(1.0, color="k", ls="--", lw=1.5)
    axr2.set_xscale("log"); axr2.set_ylim(0.5, 1.7)
    axr2.set_title(f"(data x {c:.2f})/theory — collapsed?", fontsize=10)
    axr2.set_xlabel("R [Mpc/h]"); axr2.set_ylabel("scaled data/theory")

    fig.suptitle(f"Buzzard shear h-factor test:  a single constant c={c:.3f} (~h={c/H_FID:.2f} h_fid) "
                 f"collapses data->theory", fontsize=13, y=0.995)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
