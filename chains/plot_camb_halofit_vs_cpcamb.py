""" xi_NL impact on the 2-halo shear projection: cp_camb (linear-fallback xi_NL,
current production) vs REAL CAMB halofit (takahashi) xi_NL, same fiducial cosmology.

cp_camb writes no matter_power_nl, so halo_model_cosmosis falls back to the LINEAR
P(k) for xi_NL (issue #9). Swapping in the real CSL camb module (nonlinear=both)
publishes a halofit matter_power_nl on a z-grid identical to matter_power_lin, so
xi_NL[iz] = xi_mm(k, P_nl[iz]) is the true nonlinear correlation per redshift.

Baseline dump: dvtest_new     (cp_camb, linear xi_NL)
New dump:      dvtest_camb_nl (CAMB halofit xi_NL)
Headline panel: shear_prj (the 2-halo projected term) ratio vs R, per z-bin.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
CH = "/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc"
N_BINS, N_R = 12, 10
RADII = np.geomspace(0.2, 5.0, N_R)
LBD = ["20-30", "30-45", "45-60", ">60"]
ZED = ["0.2-0.35", "0.35-0.5", "0.5-0.65"]

CP, NL, LINc = "dvtest_new", "dvtest_camb_nl", "dvtest_camb_lin"


def col(d, s):
    return np.loadtxt(f"{CH}/{d}/{s}/vals.txt", comments="#").ravel()


def load(d):
    nc = col(d, "numcountssel")
    shear_prj = col(d, "shear_prj")                       # 2-halo projected term (120)
    shear_1h = col(d, "shear1hmissel") / np.repeat(nc, N_R)
    total = shear_1h + shear_prj                          # DV-contract gamma_t
    return nc, shear_prj, total, shear_1h


def realized_s8(d):
    # CAMB writes growth_parameters/sigma_8; cp_camb (emulator) writes
    # cosmological_parameters/sigma8 (input==output by construction).
    for f in (f"{CH}/{d}/growth_parameters/sigma_8.txt",
              f"{CH}/{d}/cosmological_parameters/sigma8.txt",
              f"{CH}/{d}/cosmological_parameters/sigma_8.txt"):
        if os.path.exists(f):
            return float(np.loadtxt(f).ravel()[0])
    return float("nan")


ncCP, prjCP, totCP, h1CP = load(CP)   # cp_camb: emulator linear P(k), linear xi_NL
ncNL, prjNL, totNL, h1NL = load(NL)   # CAMB: real linear P(k), HALOFIT xi_NL
_,    prjLI, _,    h1LI = load(LINc)  # CAMB: real linear P(k), linear xi_NL (pure control)
s8CP, s8NL = realized_s8(CP), realized_s8(NL)
if not np.isfinite(s8CP):
    s8CP = 0.8238   # cp_camb is a sigma8 emulator: input==output; NC ratio=1.000 confirms match

rPRJ = (prjNL / prjCP).reshape(N_BINS, N_R)          # shear_prj: cp_camb baseline (what was asked)
rPRJpure = (prjNL / prjLI).reshape(N_BINS, N_R)      # shear_prj: PURE xi_NL (same CAMB linear P(k))
rH1 = (h1NL / h1CP).reshape(N_BINS, N_R)             # 1-halo: cp_camb->CAMB (linear-P(k) MF effect, NOT xi_NL)
rH1pure = (h1NL / h1LI).reshape(N_BINS, N_R)         # 1-halo across nonlinear toggle (must be ~1)
rNC = ncNL / ncCP

print(f"realized sigma_8: cp_camb={s8CP:.4f}  CAMB={s8NL:.4f}")
print(f"NC   CAMB/cp_camb: mean {rNC.mean():.3f}  range {rNC.min():.3f}-{rNC.max():.3f}")
print(f"1-halo across nonlinear toggle (must be ~1): max|dev|={np.max(np.abs(rH1pure-1)):.2e}")
print("shear_prj (2-halo) xi_NL impact, median over bins by R:")
for i, r in enumerate(RADII):
    print(f"  R={r:5.2f}  cp_camb-baseline={np.median(rPRJ[:,i]):.3f}  pure-xiNL={np.median(rPRJpure[:,i]):.3f}"
          f"   1-halo(MF,cp->CAMB)={np.median(rH1[:,i]):.3f}")

prjCP2, prjNL2 = prjCP.reshape(N_BINS, N_R), prjNL.reshape(N_BINS, N_R)
totCP2, totNL2 = totCP.reshape(N_BINS, N_R), totNL.reshape(N_BINS, N_R)

# ============================ FIGURE ============================
fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.32)

# rows 0-1: shear_prj (2-halo) per-bin overlay (8 of 12 bins shown: all 12)
for b in range(N_BINS):
    a = fig.add_subplot(gs[b // 4, b % 4])
    a.plot(RADII, prjCP2[b], "o-", ms=3, lw=1, color="#c94f4f", label="cp_camb (linear ξ$_{NL}$)")
    a.plot(RADII, prjNL2[b], "s-", ms=3, lw=1, color="#3d7dc9", label="CAMB halofit ξ$_{NL}$")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_title(f"shear_prj  λ {LBD[b % 4]}, z {ZED[b // 4]}", fontsize=8)
    if b % 4 == 0:
        a.set_ylabel(r"$\Delta\Sigma_{\rm 2h}$", fontsize=9)
    if b == 0:
        a.legend(fontsize=7)

# row 3 left: shear_prj ratio vs R (THE HEADLINE = xi_NL impact)
axp = fig.add_subplot(gs[3, :2])
for b in range(N_BINS):
    axp.plot(RADII, rPRJ[b], "-", lw=0.6, alpha=0.3, color="#3d7dc9")
axp.plot(RADII, np.median(rPRJ, 0), "-", color="#3d7dc9", lw=2.4,
         label="halofit / cp_camb (asked)")
axp.plot(RADII, np.median(rPRJpure, 0), "k--", lw=1.8,
         label="pure ξ$_{NL}$ (halofit / CAMB-linear)")
axp.axhline(1, color="0.5", lw=1); axp.set_xscale("log")
axp.set_xlabel("R [Mpc/h]"); axp.set_ylabel(r"shear_prj ratio")
axp.set_title("ξ$_{NL}$ impact on the 2-halo term (shear_prj)\n"
              "the two curves coincide → shear_prj ratio IS the ξ$_{NL}$ signal", fontsize=10)
axp.legend(fontsize=8)

# row 3 right: 1-halo ratio vs R -- a SEPARATE linear-P(k) mass-function effect, NOT xi_NL
axt = fig.add_subplot(gs[3, 2:])
for b in range(N_BINS):
    axt.plot(RADII, rH1[b], "-", lw=0.6, alpha=0.3, color="#c94f4f")
axt.plot(RADII, np.median(rH1, 0), "-", color="#c94f4f", lw=2.4, label="1-halo cp_camb→CAMB")
axt.plot(RADII, np.median(rH1pure, 0), "k--", lw=1.8, label="1-halo across NL toggle (≡1)")
axt.axhline(1, color="0.5", lw=1); axt.set_xscale("log")
axt.set_xlabel("R [Mpc/h]"); axt.set_ylabel(r"1-halo (shear1h/NC) ratio")
axt.set_title("1-halo shift = linear-P(k) mass-function effect (NOT ξ$_{NL}$)\n"
              "σ(M)/concentration use LINEAR P(k) → flat under NL toggle", fontsize=10)
axt.legend(fontsize=8)

fig.suptitle(
    f"Real CAMB halofit ξ$_{{NL}}$ vs cp_camb (linear-fallback) at fiducial  "
    f"[σ8: cp_camb={s8CP:.3f}, CAMB={s8NL:.3f}]",
    fontsize=13, y=0.995)
out = f"{_HERE}/buzzard_dvtest_camb_halofit_prj.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
print(f"wrote {out}")
