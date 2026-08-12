"""chi2 breakdown for the h-fixed Buzzard best-fit: where does the misfit live?

chi2 = delta^T C^-1 delta. Per-element contribution c_i = delta_i * (C^-1 delta)_i
sums to the total chi2 (includes covariance cross-terms), so we can attribute
chi2 to each (bin, radius). Shows: (a) shear pull heatmap (data-fit)/sigma,
(b) chi2 per radius (is it large-R?), (c) chi2 per bin, (d) NC per-bin chi2.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock"
BF = "/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard_hfix/bestfit_test"
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(_HERE), "chains", "buzzard_hfix_chi2.png")
N_BINS, N_R = 12, 10
RADII = np.geomspace(0.2, 5.0, N_R)
LBD = ["20-30", "30-45", "45-60", ">60"]
ZED = ["0.2-0.35", "0.35-0.5", "0.5-0.65"]
binlab = [f"{LBD[b%4]}\n{ZED[b//4]}" for b in range(N_BINS)]


def col(sec):
    return np.loadtxt(f"{BF}/{sec}/vals.txt", comments='#').ravel()


def main():
    dv = np.load(f"{ROOT}/dv_buzzard_jkcov_hfix.npz")
    dNC, dSH = dv["data_NC"].astype(float), dv["data_Shear"].astype(float)
    icNC, icSH = dv["invcov_NC"].astype(float), dv["invcov_Shear"].astype(float)
    sSH = np.sqrt(np.diag(np.linalg.inv(icSH)))

    tNC = col("numcountssel")
    tSH = col("shear1hmissel") / np.repeat(tNC, N_R) + col("shear_prj")

    dNCr, dSHr = dNC - tNC, dSH - tSH
    chi2_NC = float(dNCr @ icNC @ dNCr)
    chi2_SH = float(dSHr @ icSH @ dSHr)
    contribNC = dNCr * (icNC @ dNCr)                 # per-element, sums to chi2_NC
    contribSH = (dSHr * (icSH @ dSHr)).reshape(N_BINS, N_R)
    pull = (dSHr / sSH).reshape(N_BINS, N_R)
    print(f"chi2  NC={chi2_NC:.0f}/12   Shear={chi2_SH:.0f}/120   total={chi2_NC+chi2_SH:.0f}/132 = {(chi2_NC+chi2_SH)/132:.1f}/dof")
    print(f"shear chi2 by radius: {np.round(contribSH.sum(0)).astype(int)}")
    print(f"  -> last 3 radii (R>2.4) = {100*contribSH[:, -3:].sum()/chi2_SH:.0f}% of shear chi2")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # (a) shear pull heatmap
    ax = fig.add_subplot(gs[0, 0])
    vlim = np.abs(pull).max()
    im = ax.imshow(pull, aspect='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(N_R)); ax.set_xticklabels([f"{r:.1f}" for r in RADII], fontsize=7, rotation=45)
    ax.set_yticks(range(N_BINS)); ax.set_yticklabels(binlab, fontsize=6)
    ax.set_xlabel("R [Mpc/h]"); ax.set_title("(a) shear pull (data-fit)/sigma", fontsize=11)
    fig.colorbar(im, ax=ax, label=r"$\sigma$")

    # (b) chi2 per radius
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(N_R), contribSH.sum(0), color="#3d7dc9")
    ax.set_xticks(range(N_R)); ax.set_xticklabels([f"{r:.1f}" for r in RADII], fontsize=7, rotation=45)
    ax.set_xlabel("R [Mpc/h]"); ax.set_ylabel(r"$\chi^2$ contribution")
    ax.set_title(f"(b) shear $\\chi^2$ per radius (total {chi2_SH:.0f})", fontsize=11)

    # (c) chi2 per bin (shear)
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(range(N_BINS), contribSH.sum(1), color="#3d7dc9")
    ax.set_xticks(range(N_BINS)); ax.set_xticklabels(binlab, fontsize=6)
    ax.set_ylabel(r"$\chi^2$ contribution"); ax.set_title("(c) shear $\\chi^2$ per bin", fontsize=11)

    # (d) NC per-bin chi2 + data/theory
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(range(N_BINS), contribNC, color="#c94f4f")
    for b in range(N_BINS):
        ax.annotate(f"{dNC[b]/tNC[b]:.2f}", (b, contribNC[b]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xticks(range(N_BINS)); ax.set_xticklabels(binlab, fontsize=6)
    ax.set_ylabel(r"$\chi^2$ contribution"); ax.set_title(f"(d) NC $\\chi^2$ per bin (total {chi2_NC:.0f}; labels=data/theory)", fontsize=11)

    fig.suptitle(f"Buzzard h-fixed best-fit chi2 breakdown: total {chi2_NC+chi2_SH:.0f}/132 = "
                 f"{(chi2_NC+chi2_SH)/132:.1f}/dof (tiny JK errors + coherent radial tilt)", fontsize=13)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
