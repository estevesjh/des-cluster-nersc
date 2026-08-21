"""Measure the Buzzard concentration-mass relation c(M) = R200m / rs from the
halo profile catalog, and compare to Child18 (the pipeline's default).

Justifies the concentration_amplitude ~ 1.25 knob (halo_model_cosmosis.py):
Buzzard clusters are more concentrated than Child18.

Selection matches the DV build (build_buzzard_datavector.py):
  pid == -1 (host halos), 0 <= cosi <= 1, seam-excised z (z<0.33 or z>0.37),
  log10(Mvir) >= 13, 0.2 <= z <= 0.65.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/src")
sys.path.insert(0, "/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev/y3_buzzard")
import fitsio
from fileLoc import FileLocs
from haloModel import child18_mass_concentration

OUT = ["/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev/docs/figs/buzzard_cM_vs_child18.png",
       "/pscratch/sd/j/jesteves/github/des-cluster-nersc/chains/buzzard_cM_vs_child18.png"]

d = fitsio.read(FileLocs(machine="nersc").profile_output_fname)
sel = ((d["pid"] == -1) & (d["cosi"] >= 0) & (d["cosi"] <= 1)
       & ((d["redshift"] < 0.33) | (d["redshift"] > 0.37))
       & (np.log10(d["Mvir"]) >= 13.0)
       & (d["redshift"] >= 0.2) & (d["redshift"] <= 0.65)
       & (d["rs"] > 0) & (d["R200m"] > 0))
# Restrict to the CLUSTER regime M200m >= 1e13 (the DV's actual range). Below
# this r_s is under-resolved (too few particles) -> R200m/rs is unreliable
# (c collapses to ~1), so those bins are excluded, not physical.
sel &= (d["M200m"] >= 1e13)
M = d["M200m"][sel]
z = d["redshift"][sel]
c_buzz = d["R200m"][sel] / d["rs"][sel]          # Buzzard concentration
print(f"selected {sel.sum()} cluster host halos (M200m>=1e13); "
      f"c=R200m/rs median={np.median(c_buzz):.2f}")

# mass bins (log10 M200m), cluster regime
mbins = np.linspace(13.0, np.log10(M.max()), 8)
mc = 0.5 * (mbins[:-1] + mbins[1:])
zbins = [(0.2, 0.35), (0.35, 0.5), (0.5, 0.65)]
zcol = plt.cm.viridis(np.linspace(0, 0.85, 3))

fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})
ratios = []
for iz, (zlo, zhi) in enumerate(zbins):
    inz = (z >= zlo) & (z < zhi)
    med_b, med_c = [], []
    for i in range(len(mc)):
        inm = inz & (np.log10(M) >= mbins[i]) & (np.log10(M) < mbins[i + 1])
        if inm.sum() < 20:
            med_b.append(np.nan); med_c.append(np.nan); continue
        cb = np.median(c_buzz[inm])
        cc = child18_mass_concentration(np.median(M[inm]),
                                        np.median(z[inm]), halo_sample="stacked_nfw")
        med_b.append(cb); med_c.append(float(cc))
    med_b = np.array(med_b); med_c = np.array(med_c)
    ax.plot(10**mc, med_b, "o-", color=zcol[iz], label=f"Buzzard z {zlo}-{zhi}")
    ax.plot(10**mc, med_c, "--", color=zcol[iz], lw=1.2)
    axr.plot(10**mc, med_b / med_c, "o-", color=zcol[iz])
    ratios.append(med_b / med_c)

ratio_all = np.nanmedian(np.concatenate(ratios))
ax.plot([], [], "k--", label="Child18 (dashed)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylabel("concentration  c = R$_{200m}$ / r$_s$")
ax.legend(fontsize=8)
ax.set_title(f"Buzzard c(M) vs Child18 — Buzzard is more concentrated "
             f"(median ratio {ratio_all:.2f})")
axr.axhline(1.0, color="k", lw=0.7)
axr.axhline(ratio_all, color="crimson", lw=1.2, ls=":",
            label=f"median {ratio_all:.2f}")
axr.set_xscale("log")
axr.set_ylabel("c$_{Buzz}$ / c$_{Child18}$")
axr.set_xlabel("M$_{200m}$ [M$_\\odot$/h]")
axr.legend(fontsize=8)
fig.tight_layout()
for o in OUT:
    os.makedirs(os.path.dirname(o), exist_ok=True)
    fig.savefig(o, dpi=120, bbox_inches="tight")
    print("wrote", o)
print(f"median c_Buzz/c_Child18 = {ratio_all:.3f}  (justifies concentration_amplitude~1.25)")
