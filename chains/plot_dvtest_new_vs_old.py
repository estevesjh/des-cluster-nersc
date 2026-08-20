"""Likelihood-level data-vector test: NEW (estevesjh issue-4) vs OLD (marcpaterno
master) code, same mock_mcmc_cp_camb.ini at fiducial. Compares the theory NC and
DeltaSigma the likelihood sees, to isolate what the 2-halo/issue-4 code changes.
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
LBD = ["20-30", "30-45", "45-60", ">60"]; ZED = ["0.2-0.35", "0.35-0.5", "0.5-0.65"]
binlab = [f"{LBD[b%4]}\n{ZED[b//4]}" for b in range(N_BINS)]


def load(tag):
    d = f"{CH}/dvtest_{tag}"
    col = lambda s: np.loadtxt(f"{d}/{s}/vals.txt", comments='#').ravel()
    nc = col("numcountssel")
    sh = col("shear1hmissel") / np.repeat(nc, N_R) + col("shear_prj")   # DV-contract shear
    return nc, sh


ncO, shO = load("old")
ncN, shN = load("new")
rNC = ncN / ncO
rSH = (shN / shO).reshape(N_BINS, N_R)
print("NC  new/old ratio:", np.round(rNC, 3))
print(f"NC  new/old: mean {rNC.mean():.3f}  range {rNC.min():.3f}-{rNC.max():.3f}")
print(f"Shear new/old by radius (median over bins):")
for i, r in enumerate(RADII):
    print(f"  R={r:5.2f}  ratio={np.median(rSH[:,i]):.3f}")

# ---- Figure 1: NC ----
fig, ax = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.33})
x = np.arange(N_BINS)
ax[0].plot(x-0.1, ncO, 'o', ms=7, color='#c94f4f', label='OLD (marcpaterno master)')
ax[0].plot(x+0.1, ncN, 's', ms=7, color='#3d7dc9', label='NEW (issue-4)')
ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels(binlab, fontsize=7)
ax[0].set_ylabel('number counts'); ax[0].legend()
ax[0].set_title('NC theory at fiducial: NEW vs OLD code (same ini)')
ax[1].axhline(1, color='k', lw=1); ax[1].bar(x, rNC, color='#3d7dc9')
for i in x: ax[1].annotate(f"{rNC[i]:.3f}", (i, rNC[i]), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=7)
ax[1].set_xticks(x); ax[1].set_xticklabels(binlab, fontsize=7); ax[1].set_ylabel('NEW / OLD')
ax[1].set_title('NC ratio NEW/OLD')
plt.savefig(f"{_HERE}/buzzard_dvtest_nc.png", dpi=140, bbox_inches='tight'); plt.close()
print("wrote chains/buzzard_dvtest_nc.png")

# ---- Figure 2: DeltaSigma ----
dgO, dgN = shO.reshape(N_BINS, N_R), shN.reshape(N_BINS, N_R)
fig = plt.figure(figsize=(16, 11)); gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.3)
for b in range(N_BINS):
    a = fig.add_subplot(gs[b//4, b%4])
    a.plot(RADII, dgO[b], 'o-', ms=3, lw=1, color='#c94f4f', label='OLD')
    a.plot(RADII, dgN[b], 's-', ms=3, lw=1, color='#3d7dc9', label='NEW')
    a.set_xscale('log'); a.set_yscale('log'); a.set_title(f"{LBD[b%4]}, z {ZED[b//4]}", fontsize=8)
    if b % 4 == 0: a.set_ylabel(r'$\Delta\Sigma$', fontsize=9)
    if b == 0: a.legend(fontsize=7)
# ratio-vs-R row
axr = fig.add_subplot(gs[3, :])
for b in range(N_BINS): axr.plot(RADII, rSH[b], 'o-', ms=2, lw=0.6, alpha=0.5)
axr.plot(RADII, np.median(rSH, 0), 'k-', lw=2, label='median')
axr.axhline(1, color='0.5', lw=1); axr.set_xscale('log')
axr.set_xlabel('R [Mpc/h]'); axr.set_ylabel(r'$\Delta\Sigma$ NEW/OLD'); axr.legend(fontsize=9)
axr.set_title('DeltaSigma ratio NEW/OLD vs R (per bin + median)')
fig.suptitle('DeltaSigma theory at fiducial: NEW (issue-4) vs OLD (marcpaterno master)', fontsize=13, y=0.99)
plt.savefig(f"{_HERE}/buzzard_dvtest_dsigma.png", dpi=140, bbox_inches='tight'); plt.close()
print("wrote chains/buzzard_dvtest_dsigma.png")
