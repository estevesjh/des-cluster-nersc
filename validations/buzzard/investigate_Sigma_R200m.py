"""Buzzard validation 4b: rho_m,0 x (1+z)^n from the PROJECTED surface density Sigma(R).

Projected analog of investigate_rhom_1pz.py. Instead of the 3D rho(r) we fit the
measured surface density Sigma(R) (profile catalog, physical Msun/pc^2, 15 log
bins 0.0323-30 physical Mpc; grid from muStarSigmaProfiles/radial_bins_phys_mpc.py)
with the Wright & Brainerd (2000) projected-NFW Sigma(R) = 2 rho_s rs f(R/rs).

Steps (mirroring the rho(r) analysis):
  1. subtract the large-R background plateau (projected mean density + 2-halo),
  2. fit the projected NFW to the 1-halo Sigma -> rho_s, rs,
  3. R200m from spherical overdensity on the fit (mean enclosed = 200 rho_m(z)),
  4. read the DIRECT measured Sigma_1h(R200m) by log-log interpolation,
  5. plot the dimensionless Sigma(R200m)/(rho_m(z=0) R200m) vs z, per (Mvir, z) bin.

Quality selection = MockDataVector.ipynb select_good (pid==-1, 0<=cosi<=1,
seam-excised) + Mvir>=1e13. rho is the sim's particle-mass surface-density.
"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq
import sys

warnings.filterwarnings("ignore", category=RuntimeWarning)   # arctanh edge at x->1
sys.path.insert(0, "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/src")
import fitsio
from fileLoc import FileLocs

# Sigma(R) projected grid (physical Mpc), muStarSigmaProfiles/radial_bins_phys_mpc.py
_e = np.exp(np.linspace(np.log(0.0323), np.log(30.0), 16))
R_SIG = np.sqrt(_e[:-1] * _e[1:])                       # 15 physical Mpc
FIT_RMAX = 2.5                                          # fit 1-halo inside this (phys Mpc)
BG_RMIN = 5.0                                           # background plateau beyond this
OM, H = 0.286, 0.70
RHO_M0 = OM * 2.77533742639e+11 * H ** 2                # physical Msun/Mpc^3, z=0


def _mu(x):
    return np.log(1.0 + x) - x / (1.0 + x)


def sigma_nfw(R, rho_s, rs):
    """Wright & Brainerd (2000) projected NFW Sigma(R) [Msun/Mpc^2]."""
    x = np.atleast_1d(R / rs).astype(float)
    out = np.empty_like(x)
    lo, hi = x < 1 - 1e-6, x > 1 + 1e-6
    mid = ~(lo | hi)
    xl = x[lo]
    out[lo] = (1.0 - 2.0 / np.sqrt(1 - xl ** 2)
               * np.arctanh(np.sqrt((1 - xl) / (1 + xl)))) / (xl ** 2 - 1)
    xh = x[hi]
    out[hi] = (1.0 - 2.0 / np.sqrt(xh ** 2 - 1)
               * np.arctan(np.sqrt((xh - 1) / (1 + xh)))) / (xh ** 2 - 1)
    out[mid] = 1.0 / 3.0
    return 2.0 * rho_s * rs * out                       # Msun/Mpc^2


def fit_sigma(R, sig_pc2):
    """Fit projected NFW to background-subtracted 1-halo Sigma. Returns rho_s, rs."""
    bg = np.nanmedian(sig_pc2[R > BG_RMIN])             # plateau: proj mean-density+2-halo
    sig1h = (sig_pc2 - bg) * 1e12                       # Msun/Mpc^2, 1-halo only
    ok = (R < FIT_RMAX) & np.isfinite(sig1h) & (sig1h > 0)
    if ok.sum() < 5:
        return np.nan, np.nan, bg
    model = lambda r, lg, rs: np.log10(sigma_nfw(r, 10 ** lg, rs))
    try:
        p, _ = curve_fit(model, R[ok], np.log10(sig1h[ok]),
                         p0=[np.log10(RHO_M0 * 1e4), 0.3], maxfev=10000)
        return 10 ** p[0], p[1], bg
    except Exception:
        return np.nan, np.nan, bg


def c200m_from_fit(rho_s, rs, z):
    """c200m via spherical overdensity: m(x)/x^3 = 200 rho_m(z)/(3 rho_s)."""
    if not (np.isfinite(rho_s) and np.isfinite(rs) and rho_s > 0 and rs > 0):
        return np.nan
    target = 200.0 * RHO_M0 * (1 + z) ** 3 / (3.0 * rho_s)
    f = lambda x: _mu(x) / x ** 3 - target
    if f(1e-3) < 0 or f(1e3) > 0:
        return np.nan
    return brentq(f, 1e-3, 1e3, maxiter=200)


def main():
    d = fitsio.read(FileLocs(machine="nersc").profile_output_fname)
    sel = ((d["pid"] == -1) & (d["cosi"] >= 0) & (d["cosi"] <= 1)
           & ((d["redshift"] < 0.33) | (d["redshift"] > 0.37))
           & (d["Mvir"] >= 1e13))
    Mvir = d["Mvir"][sel].astype(float)
    z = d["redshift"][sel]
    sig = np.asarray(d["Sigma"])[sel]                   # (N,15) physical Msun/pc^2
    print(f"{sel.sum()} halos; Sigma grid (phys Mpc): {np.round(R_SIG,3)}")
    print(f"rho_m(z=0) = {RHO_M0:.3e} Msun/Mpc^3\n")

    mbins = np.logspace(13, 15, 21)
    mc = np.sqrt(mbins[:-1] * mbins[1:])
    zedges = np.arange(0.10, 0.85 + 1e-9, 0.05)
    zc = 0.5 * (zedges[:-1] + zedges[1:])
    norm = plt.Normalize(13.0, 15.0)
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(8.5, 6))
    lgR = np.log10(R_SIG)
    # projected-NFW Sigma(R200m)/(rho_m(z) R200m) for c=5 (pure physical-SO reference)
    cref = 5.0
    g2 = sigma_nfw(cref, 200.0 * cref ** 3 / (3.0 * _mu(cref)), 1.0)[0] / cref
    lo, hi = [], []                                     # for the seam-step summary
    print(f"{'Mvir bin center':>16}{'c200m':>8}{'R200m[Mpc]':>11}"
          f"{'Sig_meas(R200m)/(rho_m0 R200m) (1+z)^n':>40}")
    for k in range(len(mbins) - 1):
        ratio = np.full(len(zc), np.nan)
        cbin = np.full(len(zc), np.nan)
        r2bin = np.full(len(zc), np.nan)
        for i in range(len(zc)):
            m = ((Mvir >= mbins[k]) & (Mvir < mbins[k + 1])
                 & (z >= zedges[i]) & (z < zedges[i + 1]))
            if m.sum() < 30:
                continue
            stack = np.median(sig[m], axis=0)           # measured Sigma(R), Msun/pc^2
            rho_s, rs, bg = fit_sigma(R_SIG, stack)
            c200 = c200m_from_fit(rho_s, rs, zc[i])
            if not np.isfinite(c200):
                continue
            R200 = c200 * rs
            sig1h = (stack - bg) * 1e12                  # Msun/Mpc^2 1-halo
            gd = (R_SIG < FIT_RMAX) & np.isfinite(sig1h) & (sig1h > 0)
            if gd.sum() < 3:
                continue
            s_meas = 10.0 ** np.interp(np.log10(R200), lgR[gd], np.log10(sig1h[gd]))
            ratio[i] = s_meas / (RHO_M0 * R200)         # dimensionless
            cbin[i] = c200
            r2bin[i] = R200
            if mc[k] >= 1.4e13:                         # lowest-mass bg-subtraction noisy
                frac = ratio[i] / (g2 * (1 + zc[i]) ** 3)
                (lo if zc[i] < 0.35 else hi).append(frac)
        ok = np.isfinite(ratio)
        if ok.sum() < 3:
            continue
        n = np.polyfit(np.log(1 + zc[ok]), np.log(ratio[ok]), 1)[0]
        ax.plot(zc[ok], ratio[ok], "o-", ms=3, lw=1,
                color=cmap(norm(np.log10(mc[k]))))
        print(f"{mc[k]:16.2e}{np.nanmedian(cbin):8.2f}{np.nanmedian(r2bin):11.3f}"
              f"                (1+z)^{n:+.2f}")

    zg = np.linspace(zc[0], zc[-1], 50)
    ax.plot(zg, g2 * (1 + zg) ** 3, "k--", lw=1.3,
            label=r"projected NFW $c{=}5$, $\propto(1+z)^3$")
    # seam-step summary vs the (1+z)^3 reference (Mvir>=1.4e13)
    lo, hi = np.array(lo), np.array(hi)
    print(f"\nprojected-SO reference g2(c=5) = {g2:.1f}")
    print(f"seam step:  z<0.33 {np.median(lo):.3f} / (1+z)^3  ,  "
          f"z>0.37 {np.median(hi):.3f}  ->  "
          f"+{100*(np.median(hi)/np.median(lo)-1):.1f}%")

    ax.axvspan(0.33, 0.37, color="0.85", zorder=0)        # Buzzard box seam
    ax.set_xlabel("redshift z")
    ax.set_ylabel(r"$\Sigma_{\rm meas}(R_{200m}) / (\rho_m(z{=}0)\,R_{200m})$")
    ax.set_yscale("log")
    ax.set_title("Buzzard measured $\\Sigma(R_{200m})$ vs z "
                 "(projected-NFW fit, bg-subtracted; 20 mass bins)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label=r"log$_{10}$ M$_{vir}$ [M$_\odot$/h]")
    out = ("/pscratch/sd/j/jesteves/github/des-cluster-nersc/validations/buzzard/"
           "Sigma_R200m_over_rhom0_vs_z.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
