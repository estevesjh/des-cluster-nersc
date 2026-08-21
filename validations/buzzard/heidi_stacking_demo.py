"""What Heidi's mass-redshift-weighted stacking does, shown on Buzzard.

Heidi's `stacked_profile_weighted_by_mass_redshift` (muStarSigmaProfiles) is NOT a
plain average of the selected clusters. It builds a (lnM, z)-MATCHED reference:

  stacked(R) = sum_cells [ N_select(cell) * <profile_ALL(cell)>(R) ] / sum N_select

i.e. for each fine (lnM dm=0.1, z dz=0.05) cell it takes the MEAN profile of ALL
halos in that cell, weighted by how many SELECTED clusters fall in the cell.

- Naive stack     = mean( profile[selected] )         (what I've been doing)
- Weighted stack  = (M,z)-matched mean of ALL halos, reweighted to the selected
                    sample's (M,z) distribution.
- ratio = naive / weighted = the SELECTION bias on the profile AT FIXED mass &
  redshift. If richness (or mu_star) selection preferentially picks oriented /
  concentrated / projected halos beyond their mass, ratio != 1. For a pure
  mass-selected sample the two are identical (ratio == 1) -- shown as a control.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/src")
import fitsio
from fileLoc import FileLocs

_e2 = np.exp(np.linspace(np.log(0.0323), np.log(30.0), 16))
R_SIG = np.sqrt(_e2[:-1] * _e2[1:])


def heidi_weighted_stack(lnM_sel, z_sel, lnM_all, z_all, prof_all, dm=0.1, dz=0.05):
    """Exact port of stacked_profile_weighted_by_mass_redshift (muStarSigmaProfiles)."""
    m_bins = np.arange(lnM_all.min() - dm, lnM_all.max() + 2 * dm, dm)
    z_bins = np.arange(z_all.min() - dz, z_all.max() + 2 * dz, dz)
    nr = prof_all.shape[1]
    out = np.zeros(nr)
    wnorm = 0.0
    for iz in range(len(z_bins) - 1):
        zlo, zhi = z_bins[iz], z_bins[iz + 1]
        sel_z = (z_sel >= zlo) & (z_sel < zhi)
        all_z = (z_all >= zlo) & (z_all < zhi)
        if not sel_z.any():
            continue
        for iM in range(len(m_bins) - 1):
            mlo, mhi = m_bins[iM], m_bins[iM + 1]
            w = np.sum(sel_z & (lnM_sel >= mlo) & (lnM_sel < mhi))    # N_selected in cell
            if w == 0:
                continue
            cell = all_z & (lnM_all >= mlo) & (lnM_all < mhi)          # ALL halos in cell
            if cell.any():
                out += np.mean(prof_all[cell], axis=0) * w
                wnorm += w
    return out / wnorm


def main():
    fl = FileLocs(machine="nersc")
    p = fitsio.read(fl.profile_output_fname)
    h = fitsio.read(fl.halo_run_fname)
    good = ((p["pid"] == -1) & (p["cosi"] >= 0) & (p["cosi"] <= 1)
            & ((p["redshift"] < 0.33) | (p["redshift"] > 0.37)))
    lnM = np.log(p["Mvir"][good].astype(float))
    zz = p["redshift"][good]
    lam = h["LAMBDA_CHISQ"][good]
    ds = np.asarray(p["DeltaSigma"])[good]
    sig = np.asarray(p["Sigma"])[good]

    ZMIN, ZMAX, LAMMIN = 0.2, 0.33, 20.0
    inz = (zz >= ZMIN) & (zz < ZMAX)
    rich = inz & (lam > LAMMIN)                              # richness-selected
    print(f"z[{ZMIN},{ZMAX}): {inz.sum()} halos; richness lambda>{LAMMIN}: {rich.sum()}")

    # mass-matched CONTROL: same N, drawn to match richness sample's mass hist
    # (a pure mass selection -> ratio should be ~1)
    mass_ctrl = inz & (p["Mvir"][good] > np.median(p["Mvir"][good][rich]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, (obs, arr, name) in enumerate(
            [("ds", ds, r"$\Delta\Sigma(R)$"), ("sig", sig, r"$\Sigma(R)$")]):
        ax, axr = axes[0, col], axes[1, col]
        for lab, selmask, color in [("richness λ>20", rich, "crimson"),
                                    ("mass-selected (control)", mass_ctrl, "royalblue")]:
            naive = np.mean(arr[selmask], axis=0)
            weighted = heidi_weighted_stack(lnM[selmask], zz[selmask], lnM, zz, arr)
            ax.loglog(R_SIG, naive, "o-", color=color, ms=4, label=f"{lab}: naive")
            ax.loglog(R_SIG, weighted, "s--", color=color, ms=4, mfc="none",
                      label=f"{lab}: (M,z)-weighted")
            axr.semilogx(R_SIG, naive / weighted, "o-", color=color, ms=4, label=lab)
        ax.set_ylabel(f"{name} [M$_\\odot$/pc$^2$]")
        ax.set_title(f"{name}: naive vs Heidi (M,z)-matched stack "
                     f"(z∈[{ZMIN},{ZMAX}])", fontsize=11)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.2, which="both")
        axr.axhline(1, color="k", lw=1)
        axr.set_ylabel("naive / weighted"); axr.set_xlabel("R [Mpc]")
        axr.set_ylim(0.8, 1.6); axr.grid(alpha=0.3); axr.legend(fontsize=8)

    fig.suptitle("Heidi's mass-redshift-matched stack: isolates SELECTION bias "
                 "at fixed (M,z)  (ratio=1 for mass-selection)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ("/pscratch/sd/j/jesteves/github/des-cluster-nersc/validations/buzzard/"
           "heidi_stacking_demo.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    # print the DeltaSigma ratio at a few radii
    naive = np.mean(ds[rich], axis=0)
    weighted = heidi_weighted_stack(lnM[rich], zz[rich], lnM, zz, ds)
    print("\nDeltaSigma naive/weighted (richness λ>20), R[Mpc] -> ratio:")
    for R, rr in zip(R_SIG, naive / weighted):
        if 0.1 < R < 5:
            print(f"  R={R:5.2f}  ratio={rr:.3f}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
