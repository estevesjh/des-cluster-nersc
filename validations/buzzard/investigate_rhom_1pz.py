"""Buzzard validation 4a: rho(R200m)/rho_m(z=0) vs z from the 3D rho(r).

Fit the measured 3D density profile rho(r) (profile catalog, physical Msun/Mpc^3,
10 log-bins 0.05-2.5 physical Mpc; grid from estevesjh/muStarSigmaProfiles
radial_bins_3d_phys_mpc.py) with an NFW rho(r) = rho_s / [(r/rs)(1+r/rs)^2] in
bins of (Mvir log-spaced 1e13-1e15, 20 bins) x (z, dz=0.05). Then, per bin:
  1. R200m from spherical overdensity on the fit: mean enclosed = 200 rho_m(z),
  2. read the DIRECT measured rho(r) at R200m by log-log interpolation,
  3. plot rho_meas(R200m)/rho_m(z=0) vs z.

rho_s alone is mass-dependent; evaluating at R200m pins the density to the
overdensity boundary (~a fixed fraction of 200 rho_m(z)), collapsing the mass
dependence so only the (1+z) density evolution + the Buzzard box-seam jump remain.
RESULT: rho_meas(R200m)/rho_m(z=0) ~ (1+z)^3.4, mass-collapsed (c200m~3.3-5.0),
with a +10.5% step across the z=0.33 simulation-box seam. The pipeline builds the
1-halo with COMOVING rho_m0 frozen at z=0 -> misses this physical (1+z) growth.

Quality selection = MockDataVector.ipynb select_good (pid==-1, 0<=cosi<=1,
seam-excised) + Mvir>=1e13.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq

sys.path.insert(0, "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/src")
import fitsio
from fileLoc import FileLocs

# rho(r) 3D grid (physical Mpc), from muStarSigmaProfiles/radial_bins_3d_phys_mpc.py
_e = np.exp(np.linspace(np.log(0.05), np.log(2.5), 11))
R_RHO = np.sqrt(_e[:-1] * _e[1:])                       # 10 physical Mpc
OM, H = 0.286, 0.70
RHO_M0 = OM * 2.77533742639e+11 * H ** 2                # physical Msun/Mpc^3, z=0


def nfw_logrho(r, log_rhos, rs):
    x = r / rs
    return log_rhos - np.log10(x) - 2.0 * np.log10(1.0 + x)


def fit_nfw(r, rho):
    ok = np.isfinite(rho) & (rho > 0)
    if ok.sum() < 5:
        return np.nan, np.nan
    try:
        p, _ = curve_fit(nfw_logrho, r[ok], np.log10(rho[ok]),
                         p0=[np.log10(rho[ok][0] * 5), 0.3], maxfev=10000)
        return 10 ** p[0], p[1]                          # rho_s [Msun/Mpc^3], rs [Mpc]
    except Exception:
        return np.nan, np.nan


def _mu(x):                                              # NFW mass shape m(x)=ln(1+x)-x/(1+x)
    return np.log(1.0 + x) - x / (1.0 + x)


def nfw_rho_at_r200m(rho_s, rs, z):
    """rho_NFW(R200m) [phys Msun/Mpc^3], R200m from spherical overdensity:
    mean enclosed density = 200 * rho_m(z) (physical). Solve m(x)/x^3 = 200 rho_m(z)/(3 rho_s)."""
    if not (np.isfinite(rho_s) and np.isfinite(rs) and rho_s > 0 and rs > 0):
        return np.nan, np.nan
    rho_mz = RHO_M0 * (1.0 + z) ** 3                     # physical mean matter density at z
    target = 200.0 * rho_mz / (3.0 * rho_s)             # = m(x)/x^3 at x=c200m
    f = lambda x: _mu(x) / x ** 3 - target             # decreasing in x -> unique root
    if f(1e-3) < 0 or f(1e3) > 0:
        return np.nan, np.nan
    c = brentq(f, 1e-3, 1e3, maxiter=200)
    rho_r200 = rho_s / (c * (1.0 + c) ** 2)             # local NFW density at R200m
    return rho_r200, c


def main():
    d = fitsio.read(FileLocs(machine="nersc").profile_output_fname)
    sel = ((d["pid"] == -1) & (d["cosi"] >= 0) & (d["cosi"] <= 1)
           & ((d["redshift"] < 0.33) | (d["redshift"] > 0.37))
           & (d["Mvir"] >= 1e13))
    Mvir = d["Mvir"][sel].astype(float)
    z = d["redshift"][sel]
    rho = np.asarray(d["rho"])[sel]                      # (N,10) physical Msun/Mpc^3
    print(f"{sel.sum()} halos; rho grid (phys Mpc): {np.round(R_RHO,3)}")
    print(f"rho_m(z=0) = {RHO_M0:.3e} Msun/Mpc^3\n")

    mbins = np.logspace(13, 15, 21)                       # 1e13..1e15, 20 bins
    mc = np.sqrt(mbins[:-1] * mbins[1:])
    zedges = np.arange(0.10, 0.85 + 1e-9, 0.05)
    zc = 0.5 * (zedges[:-1] + zedges[1:])
    norm = plt.Normalize(13.0, 15.0)
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(8.5, 6))
    lgR = np.log10(R_RHO)
    print(f"{'Mvir bin center':>16}{'c200m':>8}{'R200m[Mpc]':>11}"
          f"{'rho_meas(R200m)/rho_m0 (1+z)^n':>32}")
    for k in range(len(mbins) - 1):
        ratio = np.full(len(zc), np.nan)
        cbin = np.full(len(zc), np.nan)
        r2bin = np.full(len(zc), np.nan)
        for i in range(len(zc)):
            m = ((Mvir >= mbins[k]) & (Mvir < mbins[k + 1])
                 & (z >= zedges[i]) & (z < zedges[i + 1]))
            if m.sum() < 30:
                continue
            stack = np.median(rho[m], axis=0)            # measured rho(r), phys Msun/Mpc^3
            rho_s, rs = fit_nfw(R_RHO, stack)
            _, c200 = nfw_rho_at_r200m(rho_s, rs, zc[i])  # fit->R200m (spherical overdensity)
            if not np.isfinite(c200):
                continue
            R200 = c200 * rs                              # physical Mpc
            gd = np.isfinite(stack) & (stack > 0)
            if gd.sum() < 3:
                continue
            # DIRECT measured density at R200m: log-log interpolation of rho(r)
            rho_meas = 10.0 ** np.interp(np.log10(R200), lgR[gd], np.log10(stack[gd]))
            ratio[i] = rho_meas / RHO_M0                  # rho_measured(R200m) / rho_m(z=0)
            cbin[i] = c200
            r2bin[i] = R200
        ok = np.isfinite(ratio)
        if ok.sum() < 3:
            continue
        n = np.polyfit(np.log(1 + zc[ok]), np.log(ratio[ok]), 1)[0]
        ax.plot(zc[ok], ratio[ok], "o-", ms=3, lw=1,
                color=cmap(norm(np.log10(mc[k]))))
        print(f"{mc[k]:16.2e}{np.nanmedian(cbin):8.2f}{np.nanmedian(r2bin):11.3f}"
              f"        (1+z)^{n:+.2f}")
    # reference: 200m boundary local density ~ g(c)*200*rho_m(z), pure physical (1+z)^3
    zg = np.linspace(zc[0], zc[-1], 50)
    cref = 5.0
    g = cref ** 2 / (3.0 * _mu(cref) * (1.0 + cref) ** 2)  # local/mean at R200m for c=5
    ax.plot(zg, g * 200.0 * (1.0 + zg) ** 3, "k--", lw=1.3,
            label=r"$g(c{=}5)\cdot200\,(1+z)^3$ (physical SO)")
    ax.axvspan(0.33, 0.37, color="0.85", zorder=0)        # Buzzard box seam
    ax.set_xlabel("redshift z")
    ax.set_ylabel(r"$\rho_{\rm meas}(R_{200m}) / \rho_m(z{=}0)$")
    ax.set_yscale("log")
    ax.set_title("Buzzard measured $\\rho(R_{200m})$ vs z "
                 "(direct $\\rho(r)$, interpolated; 20 mass bins)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label=r"log$_{10}$ M$_{vir}$ [M$_\odot$/h]")
    out = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/validations/buzzard/rho_R200m_measured_over_rhom0_vs_z.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("\nwrote", out)
    print("(1+z)^3 would mean the halo characteristic density tracks the PHYSICAL "
          "mean density; the pipeline's frozen z=0 (comoving) misses this.")


if __name__ == "__main__":
    main()
