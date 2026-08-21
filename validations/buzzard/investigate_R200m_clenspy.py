"""Buzzard density at R200m vs z -- rho(r), Sigma(R), DeltaSigma(R) -- CLensPy fits.

Re-makes the (1+z) + box-seam diagnostic with the robust recipe:
  - mass bins Delta log10 M = 0.1 (1e13-1e15), require >= 100 clusters per (M,z) bin,
  - z bins dz=0.05, seam-excised [0.33,0.37],
  - fit MAX( NFW_1h(M200,c200) [CLensPy] , b_cls * 2halo(R,z) ) over R>0.2 Mpc
    (2h template: CLensPy TwoHaloTerm from a CAMB P(k,z); Sigma also gets the
    measured uniform sheet as a fixed constant),
  - R200m from spherical overdensity on the fitted (rho_s, rs): mean = 200 rho_m(z),
  - evaluate the DIRECT measured profile (2h/sheet-subtracted) at R200m (log-log
    interp), normalize by rho_m(z=0) [rho] or rho_m(z=0)*R200m [Sigma, DeltaSigma].

Reduced chi^2 uses per-radius error = standard error of the median over the halos
in the bin (1.2533*std/sqrt(N)); fit in log10 space, R>0.2.

Output: three panels rho/Sigma/DeltaSigma, ratio(R200m) vs z, colored by mass, with
the (1+z)^3 reference and z=0.33 box seam. Prints (1+z)^n, median chi2/dof, seam step.
"""
import warnings
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq
from astropy.cosmology import FlatLambdaCDM
import sys
sys.path.insert(0, "/pscratch/sd/j/jesteves/github/CLensPy/src")
sys.path.insert(0, "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/src")
import fitsio
from fileLoc import FileLocs
from clenspy.halo import NfwProfile, TwoHaloTerm
from clenspy.cosmology import PkGrid

warnings.filterwarnings("ignore")
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.286, Ob0=0.047)
COSMO.sigma8, COSMO.n_s = 0.82, 0.96
RHOC0 = COSMO.critical_density(0).to_value("Msun/Mpc^3")
RHO_M0 = RHOC0 * COSMO.Om0                              # physical z=0, Msun/Mpc^3
RMIN_FIT = 0.2
NMIN = 100

_e3 = np.exp(np.linspace(np.log(0.05), np.log(2.5), 11))
R_RHO = np.sqrt(_e3[:-1] * _e3[1:])
_e2 = np.exp(np.linspace(np.log(0.0323), np.log(30.0), 16))
R_SIG = np.sqrt(_e2[:-1] * _e2[1:])
RHOM_Z = {}


def rhom(z):
    if z not in RHOM_Z:
        RHOM_Z[z] = RHOC0 * (1 + z) ** 3 * COSMO.Om0
    return RHOM_Z[z]


def _mu(x):
    return np.log(1 + x) - x / (1 + x)


def r200m_phys(rho_s, rs, z):
    if not (np.isfinite(rho_s) and np.isfinite(rs) and rho_s > 0 and rs > 0):
        return np.nan, np.nan
    target = 200.0 * rhom(z) / (3.0 * rho_s)
    f = lambda x: _mu(x) / x ** 3 - target
    if f(1e-3) < 0 or f(1e3) > 0:
        return np.nan, np.nan
    c = brentq(f, 1e-3, 1e3, maxiter=200)
    return c * rs, c


def build_twohalo(zcs):
    kk = np.logspace(-3, 1, 120)
    zg = np.linspace(0.10, 0.85, 20)
    pk = PkGrid(cosmo=COSMO, backend="camb")
    Pk = np.array([pk(kk, zz) for zz in zg])
    th = TwoHaloTerm(kk, Pk, zvec=zg)
    th.deltasigma(R_SIG, zg[len(zg) // 2])
    T = {}
    for zc in zcs:
        rm = rhom(zc)
        T[zc] = dict(rho=rm * np.maximum(th.xi(R_RHO, zc), 0.0),
                     sig=rm * th.sigma(R_SIG, zc) / 1e12,
                     ds=rm * th.deltasigma(R_SIG, zc) / 1e12)
    return T


def _obs_model(n, kind, r):
    return (n.density(r) if kind == "rho"
            else n.sigma(r) / 1e12 if kind == "sigma" else n.deltasigma(r) / 1e12)


def fit_max(R, y, yerr, kind, t2h, sheet):
    """Fit MAX(NFW_1h, b*2h)+sheet, R>0.2, log-space. Returns dict incl chi2/dof."""
    ok = (R > RMIN_FIT) & np.isfinite(y) & (y > 0)
    if ok.sum() < 4:
        return None
    t, Rk = t2h[ok], R[ok]

    def model(r, logM, c, b):
        one = _obs_model(NfwProfile(10 ** logM, c, cosmo=COSMO), kind, r)
        return np.log10(np.maximum(one, b * t) + sheet)
    try:
        p, _ = curve_fit(model, Rk, np.log10(y[ok]),
                         p0=[14.3, 6.0, 3.0], bounds=([12, 0.5, 0.0], [16, 25, 60]),
                         maxfev=8000)
    except Exception:
        return None
    n = NfwProfile(10 ** p[0], p[1], cosmo=COSMO)
    # reduced chi2 in log-space with SEM errors
    ymod = np.maximum(_obs_model(n, kind, Rk), p[2] * t) + sheet
    slog = (yerr[ok] / y[ok]) / np.log(10.0)
    dof = max(ok.sum() - 3, 1)
    chi2 = np.sum(((np.log10(y[ok]) - np.log10(ymod)) / slog) ** 2) / dof
    return dict(M=10 ** p[0], c=p[1], b=p[2], rho_s=float(n.rho_s),
                rs=float(n.rs), chi2=chi2)


def main():
    t0 = time.time()
    d = fitsio.read(FileLocs(machine="nersc").profile_output_fname)
    sel = ((d["pid"] == -1) & (d["cosi"] >= 0) & (d["cosi"] <= 1)
           & ((d["redshift"] < 0.33) | (d["redshift"] > 0.37)) & (d["Mvir"] >= 1e13))
    Mvir = d["Mvir"][sel].astype(float)
    z = d["redshift"][sel]
    prof = dict(rho=np.asarray(d["rho"])[sel], sigma=np.asarray(d["Sigma"])[sel],
                ds=np.asarray(d["DeltaSigma"])[sel])

    mbins = 10 ** np.round(np.arange(13.0, 15.0 + 1e-9, 0.1), 3)     # dlogM=0.1
    mc = np.sqrt(mbins[:-1] * mbins[1:])
    zedges = np.arange(0.10, 0.85 + 1e-9, 0.05)
    zc = 0.5 * (zedges[:-1] + zedges[1:])
    T = build_twohalo(list(zc))
    print(f"P(k)+2h ready ({time.time()-t0:.0f}s); {len(mbins)-1} mass bins, "
          f">= {NMIN} clusters/bin\n")

    OBS = [("rho", R_RHO, "rho", r"$\rho_{\rm meas}(R_{200m})/\rho_m(0)$"),
           ("sigma", R_SIG, "sig", r"$\Sigma_{\rm meas}(R_{200m})/(\rho_m(0)R_{200m})$"),
           ("ds", R_SIG, "ds", r"$\Delta\Sigma_{\rm meas}(R_{200m})/(\rho_m(0)R_{200m})$")]
    norm = plt.Normalize(13.0, 15.0)
    cmap = plt.cm.viridis
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    n5 = NfwProfile(1e14, 5.0, cosmo=COSMO)
    for pi, (kind, R, tk, ylab) in enumerate(OBS):
        ax = axes[pi]
        lgR = np.log10(R)
        # (1+z)^3 reference amplitude for c=5 (dimensionless, at z=0)
        if kind == "rho":
            gref = 200.0 * 5.0 ** 2 / (3 * _mu(5.0) * (1 + 5.0) ** 2)
        elif kind == "sigma":
            gref = float(n5.sigma(n5.r200)) / (RHO_M0 * float(n5.r200))
        else:
            gref = float(n5.deltasigma(n5.r200)) / (RHO_M0 * float(n5.r200))
        lo, hi, chis, nfit = [], [], [], 0
        for k in range(len(mbins) - 1):
            ratio = np.full(len(zc), np.nan)
            for i in range(len(zc)):
                m = ((Mvir >= mbins[k]) & (Mvir < mbins[k + 1])
                     & (z >= zedges[i]) & (z < zedges[i + 1]))
                if m.sum() < NMIN:
                    continue
                arr = prof[kind][m]
                stack = np.median(arr, axis=0)
                sem = 1.2533 * np.std(arr, axis=0) / np.sqrt(arr.shape[0])
                t2h = T[zc[i]][tk]
                sheet = np.nanmedian(stack[R > 5.0]) if kind == "sigma" else 0.0
                r = fit_max(R, stack, sem, kind, t2h, sheet)
                if r is None:
                    continue
                nfit += 1
                chis.append(r["chi2"])
                R200, _c = r200m_phys(r["rho_s"], r["rs"], zc[i])
                if not np.isfinite(R200) or R200 <= R[0] or R200 >= R[-1]:
                    continue
                sub = stack - (sheet + r["b"] * t2h) if kind != "rho" else stack
                gd = (R > RMIN_FIT) & np.isfinite(sub) & (sub > 0)
                if gd.sum() < 3:
                    continue
                val = 10 ** np.interp(np.log10(R200), lgR[gd], np.log10(sub[gd]))
                if kind == "rho":
                    ratio[i] = val / RHO_M0
                else:
                    ratio[i] = val * 1e12 / (RHO_M0 * R200)
                (lo if zc[i] < 0.35 else hi).append(ratio[i] / (gref * (1 + zc[i]) ** 3))
            ok = np.isfinite(ratio)
            if ok.sum() < 3:
                continue
            ax.plot(zc[ok], ratio[ok], "o-", ms=3, lw=1, color=cmap(norm(np.log10(mc[k]))))
        zg = np.linspace(zc[0], zc[-1], 50)
        ax.plot(zg, gref * (1 + zg) ** 3, "k--", lw=1.3, label=r"$c{=}5$, $\propto(1+z)^3$")
        ax.axvspan(0.33, 0.37, color="0.85", zorder=0)
        ax.set_yscale("log"); ax.set_xlabel("redshift z")
        ax.set_ylabel(ylab, fontsize=11); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        lo, hi = np.array(lo), np.array(hi)
        step = 100 * (np.median(hi) / np.median(lo) - 1) if lo.size and hi.size else np.nan
        medchi = np.median(chis) if chis else np.nan
        ax.set_title(f"{kind}:  seam {step:+.0f}%   median $\\chi^2$/dof={medchi:.0f}",
                     fontsize=12)
        print(f"[{kind:5s}] {nfit} fits;  median chi2/dof={medchi:6.1f};  "
              f"seam z<0.33 {np.median(lo):.3f} / z>0.37 {np.median(hi):.3f} "
              f"= {step:+.1f}%")

    fig.suptitle("Buzzard density at $R_{200m}$ vs z -- CLensPy MAX(NFW,2h), "
                 "dlogM=0.1, >=100 clusters/bin", fontsize=13)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 label=r"log$_{10}$ M$_{vir}$ [M$_\odot$/h]", fraction=0.02, pad=0.01)
    out = ("/pscratch/sd/j/jesteves/github/des-cluster-nersc/validations/buzzard/"
           "R200m_1pz_clenspy_all.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
