"""Buzzard NFW+2halo fits + residuals for rho(r), Sigma(R), DeltaSigma(R), CLensPy.

Recipe (per J. Esteves):
  - trust only R > 0.2 Mpc (inner scales unreliable: resolution / miscentering
    smoothing -> the DeltaSigma turnover that faked a low concentration),
  - model = MAX( NFW_1h(M200,c200) , b_cls * 2halo(R,z) ). The 2-halo template
    depends only on (R,z) [CLensPy TwoHaloTerm from a CAMB P(k,z)]; the only added
    free parameter is the cluster bias amplitude b_cls (one per bin).
  - Sigma also carries a uniform mean-density sheet (the large-R plateau), added
    as a fixed constant (measured at R>5 Mpc); DeltaSigma is differential so the
    sheet cancels.

Fits CLensPy NfwProfile(M200,c200) [M200m @ z=0 comoving] independently to each
observable and shows fit (1h, 2h, MAX) + fractional residual. Answers whether the
R>0.2 + MAX(NFW,2h) fit recovers a consistent concentration across rho/Sigma/DS.
"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
COSMO.sigma8, COSMO.n_s = 0.82, 0.96                    # Buzzard v1.9.8
RMIN_FIT = 0.2                                          # trust only R > 0.2 Mpc

_e3 = np.exp(np.linspace(np.log(0.05), np.log(2.5), 11))
R_RHO = np.sqrt(_e3[:-1] * _e3[1:])
_e2 = np.exp(np.linspace(np.log(0.0323), np.log(30.0), 16))
R_SIG = np.sqrt(_e2[:-1] * _e2[1:])


def build_twohalo(zc_list):
    """CAMB P(k,z) -> TwoHaloTerm; cache the rho_m(z)-scaled 2h templates per z."""
    kk = np.logspace(-3, 1, 120)
    zg = np.linspace(0.10, 0.85, 20)
    pk = PkGrid(cosmo=COSMO, backend="camb")
    Pk = np.array([pk(kk, zz) for zz in zg])            # (nz, nk)
    th = TwoHaloTerm(kk, Pk, zvec=zg)
    th.deltasigma(R_SIG, zg[len(zg) // 2])              # trigger interpolator build
    tmpl = {}
    for zc in zc_list:
        rhom = COSMO.critical_density(zc).to_value("Msun/Mpc^3") * COSMO.Om0
        tmpl[zc] = dict(
            ds=rhom * th.deltasigma(R_SIG, zc) / 1e12,      # Msun/pc^2, bias=1
            sig=rhom * th.sigma(R_SIG, zc) / 1e12,
            rho=rhom * np.maximum(th.xi(R_RHO, zc), 0.0),   # rho_2h ~ rho_m*b*xi
            rhom=rhom)
    return tmpl


def fit_max(R, y, kind, t2h, sheet=0.0):
    """Fit MAX(NFW_1h, b*2h)+sheet over R>RMIN_FIT. Returns M200, c200, b."""
    ok = (R > RMIN_FIT) & np.isfinite(y) & (y > 0)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan
    t = t2h[ok]

    def model(r, logM, c, b):
        n = NfwProfile(10 ** logM, c, cosmo=COSMO)
        one = (n.density(r) if kind == "rho"
               else n.sigma(r) / 1e12 if kind == "sigma"
               else n.deltasigma(r) / 1e12)
        return np.log10(np.maximum(one, b * t) + sheet)
    try:
        p, _ = curve_fit(model, R[ok], np.log10(y[ok]),
                         p0=[14.0, 5.0, 3.0], bounds=([12, 0.5, 0.0], [16, 20, 50]),
                         maxfev=30000)
        return 10 ** p[0], p[1], p[2]
    except Exception:
        return np.nan, np.nan, np.nan


def curve(kind, M, c, b, R, t2h_R, sheet=0.0):
    n = NfwProfile(M, c, cosmo=COSMO)
    one = (n.density(R) if kind == "rho"
           else n.sigma(R) / 1e12 if kind == "sigma"
           else n.deltasigma(R) / 1e12)
    return np.maximum(one, b * t2h_R) + sheet, one, b * t2h_R + sheet


def main():
    d = fitsio.read(FileLocs(machine="nersc").profile_output_fname)
    sel = ((d["pid"] == -1) & (d["cosi"] >= 0) & (d["cosi"] <= 1)
           & ((d["redshift"] < 0.33) | (d["redshift"] > 0.37)) & (d["Mvir"] >= 1e13))
    Mvir = d["Mvir"][sel].astype(float)
    z = d["redshift"][sel]
    rho = np.asarray(d["rho"])[sel]
    sig = np.asarray(d["Sigma"])[sel]
    dsig = np.asarray(d["DeltaSigma"])[sel]

    bins = [(1.0e14, 1.4e14, 0.42, 0.48, r"M$_{vir}\sim$1.2e14, z$\sim$0.45"),
            (3.2e14, 4.5e14, 0.42, 0.48, r"M$_{vir}\sim$3.8e14, z$\sim$0.45")]
    zcs = sorted({0.5 * (b[2] + b[3]) for b in bins})
    tmpl = build_twohalo(zcs)

    obs = [("rho", R_RHO, rho, "rho", r"$\rho(r)$  [M$_\odot$/Mpc$^3$]"),
           ("sigma", R_SIG, sig, "sig", r"$\Sigma(R)$  [M$_\odot$/pc$^2$]"),
           ("ds", R_SIG, dsig, "ds", r"$\Delta\Sigma(R)$  [M$_\odot$/pc$^2$]")]

    nb = len(bins)
    fig, axes = plt.subplots(2 * nb, 3, figsize=(15, 5.2 * nb),
                             gridspec_kw={"height_ratios": [3, 1] * nb})
    print(f"{'bin':>26}{'obs':>7}{'M200[Msun]':>13}{'c200':>7}{'b_cls':>7}")
    for bi, (mlo, mhi, zlo, zhi, lab) in enumerate(bins):
        m = (Mvir >= mlo) & (Mvir < mhi) & (z >= zlo) & (z < zhi)
        zc = 0.5 * (zlo + zhi)
        for oi, (kind, R, arr, tkey, ylab) in enumerate(obs):
            ax, axr = axes[2 * bi, oi], axes[2 * bi + 1, oi]
            stack = np.median(arr[m], axis=0)
            t2h = tmpl[zc][tkey]
            sheet = np.nanmedian(stack[R > 5.0]) if kind == "sigma" else 0.0
            M, c, b = fit_max(R, stack, kind, t2h, sheet)
            print(f"{lab:>26}{kind:>7}{M:13.3e}{c:7.2f}{b:7.2f}")
            tot, one, two = curve(kind, M, c, b, R, t2h, sheet)
            inr = R > RMIN_FIT
            ax.loglog(R[inr], stack[inr], "o", color="k", ms=5, label="Buzzard")
            ax.loglog(R[~inr], stack[~inr], "o", mfc="none", mec="0.6", ms=5,
                      label="R<0.2 (untrusted)")
            ax.loglog(R, one, "--", color="royalblue", lw=1.4, label=f"1h NFW c={c:.1f}")
            ax.loglog(R, two, ":", color="seagreen", lw=1.4, label=f"2h (b={b:.1f})")
            ax.loglog(R, tot, "-", color="crimson", lw=2, label="MAX(1h,2h)")
            ax.axvline(RMIN_FIT, color="0.6", ls=":", lw=1)
            ax.set_ylim(max(stack.min() * 0.3, 1e-1), stack.max() * 3)
            ax.set_ylabel(ylab, fontsize=10)
            if oi == 1:
                ax.set_title(lab, fontsize=12)
            ax.legend(fontsize=7.5, loc="lower left")
            ax.grid(alpha=0.2, which="both")
            resid = stack / tot - 1.0
            axr.plot(R[inr], resid[inr], "o-", color="crimson", ms=4)
            axr.plot(R[~inr], resid[~inr], "o", mfc="none", mec="0.6", ms=4)
            axr.axhline(0, color="k", lw=1)
            axr.axvline(RMIN_FIT, color="0.6", ls=":", lw=1)
            axr.set_xscale("log"); axr.set_ylim(-0.4, 0.4)
            axr.set_ylabel("data/model-1", fontsize=9); axr.set_xlabel("R [Mpc]", fontsize=10)
            axr.grid(alpha=0.2)

    fig.suptitle("Buzzard truth: CLensPy MAX(NFW 1h, b·2h) fits + residuals "
                 f"(R>{RMIN_FIT} Mpc)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = ("/pscratch/sd/j/jesteves/github/des-cluster-nersc/validations/buzzard/"
           "profile_fits_residuals_clenspy.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
