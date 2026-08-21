"""Pipeline validation of shear_prj against the Costanzi-2026 C1 selection bias.

The DES-Y3 cluster pipeline's projection 2-halo (shear_prj) should satisfy
   (shear1hmissel/NC + shear_prj) / shear1h2hMax  ==  B_C1^{DeltaSigma}(R)
the Costanzi-2026 Appendix-C Eq. C1 selection-bias shape (the DeltaSigma curve
in xtang126/mock_cluster_buzzard fig 03). Instead the pipeline ratio blows up to
~2x at R=5 Mpc/h where the reference is ~1.08 -> shear_prj 2-halo is too large.

Layout mirrors mock_cluster_buzzard/output/figs/03_analytical_Bsel_C1_ratio.png.
Reference dump: dvtest_buzz_1h2hmax (Buzzard cosmology, compute_lensing_2h=T).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CH = "/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/dvtest_buzz_1h2hmax"
N_R = 10
R = np.geomspace(0.2, 5.0, N_R)
LBD = ["20 < λ < 30", "30 < λ < 45", "45 < λ < 60", "60 < λ < 500"]
ZED = ["0.20 < z < 0.33", "0.37 < z < 0.50", "0.50 < z < 0.65"]

col = lambda s: np.loadtxt(f"{CH}/{s}/vals.txt", comments="#").ravel()
NC = col("numcountssel")
g1h = (col("shear1hmissel") / np.repeat(NC, N_R)).reshape(12, N_R)
gprj = col("shear_prj").reshape(12, N_R)
maxmod = (col("shear1h2h_max") / np.repeat(NC, N_R)).reshape(12, N_R)
ratio = (g1h + gprj) / maxmod

# Costanzi-2026 Eq. C1, DeltaSigma flavour: B = A (R/R0)^a [1+(R/R0)^g]^((b-a)/g) + C
A, al, be, ga, C = 0.12, 4.11, -0.18, 1.82, 1.0
lam_c = np.array([25., 37.5, 52.5, 130.])
z_c = np.array([0.265, 0.435, 0.575])
def Bc1(R, lam, z):
    x = R / ((lam / 100.) ** 0.2 * (1 + z))
    return A * x ** al * (1 + x ** ga) ** ((be - al) / ga) + C

fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True, sharey=True)
for iz in range(3):
    for il in range(4):
        b = iz * 4 + il
        ax = axes[iz, il]
        ax.plot(R, ratio[b], "-o", ms=4, color="#3d7dc9", lw=1.8,
                label=r"pipeline $(1h+\mathrm{prj})/1h2h_{\max}$")
        ax.plot(R, Bc1(R, lam_c[il], z_c[iz]), "--", color="#d1801a", lw=1.8,
                label=r"Costanzi C1 $\mathcal{B}_{C1}^{\Delta\Sigma}$")
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.text(0.05, 0.92, f"{ZED[iz]}   {LBD[il]}", transform=ax.transAxes,
                fontsize=8, va="top")
        if iz == 2: ax.set_xlabel(r"$R$ [cMpc/h]")
        if il == 0: ax.set_ylabel(r"ratio")
        if b == 0: ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.86))
fig.suptitle("shear_prj validation: pipeline projection boost vs Costanzi-2026 C1 "
             r"selection bias ($\Delta\Sigma$) — pipeline is ~2× too high at large R",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shearprj_vs_bselc1.png")
plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
print("median pipeline ratio vs Costanzi B_C1 by R:")
Bref = np.array([[Bc1(R[i], lam_c[b % 4], z_c[b // 4]) for i in range(N_R)] for b in range(12)])
for i in range(N_R):
    print(f"  R={R[i]:5.2f}  pipeline={np.median(ratio[:,i]):.3f}  Costanzi={np.median(Bref[:,i]):.3f}")
print(f"wrote {out}")
