"""Does richness (lambda_obs) selection change the fitted concentration?

Uses the C19 HOD forward-model richness lambda_obs (Costanzi: HOD mean + intrinsic
Poisson/lognormal scatter + projection EMG), which is a function of (M,z) + random
draws ONLY -- NO correlation with the halo profile / orientation / LSS (unlike the
redMaPPer LAMBDA_CHISQ, deliberately not used). For each (lambda_obs, z) bin we fit
CLensPy MAX(NFW 1h, b*2h) over R>0.2 to:
  (1) the naive lambda_obs-selected stack               -> c_sel
  (2) the (logM,z)-matched reference stack (Heidi's)     -> c_matched   [null test]
  (3) a narrow true-mass bin at the bin's mean logM      -> c_massbin   [mass-mixing]
c_sel vs c_matched isolates any *selection* bias on concentration (expected ~none,
since lambda_obs has no LSS info); c_sel vs c_massbin shows the *mass-mixing*
dilution (a richness bin spans a mass range, broadening the stacked profile).

================================ UNITS LEDGER ================================
 profile 'rho'         : physical Msun / Mpc^3     on R_RHO [physical Mpc]
 profile 'Sigma','DeltaSigma' : physical Msun / pc^2  on R_SIG [physical Mpc]
 CLensPy NfwProfile(m200[Msun], c200):
     .density   -> Msun/Mpc^3      (compare to rho directly)
     .sigma/.deltasigma -> Msun/Mpc^2  -> divide by 1e12 -> Msun/pc^2
 RHO_M0 = rho_crit(0)[Msun/Mpc^3] * Om0            (physical, z=0)
 2h template = rho_m(z_c)[Msun/Mpc^3] * TwoHalo.{sigma,ds}(R,z)[Mpc] / 1e12
             -> Msun/pc^2   (so it adds to the measured Sigma/DeltaSigma)
 Fits: rho in Msun/Mpc^3 ; Sigma & DeltaSigma in Msun/pc^2 (data & model matched)
=============================================================================
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
from costanzi_selection import (sample_lambda_true, sample_lambda_obs,
                                load_prj_posterior_mean)

warnings.filterwarnings("ignore")
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.286, Ob0=0.046)
COSMO.sigma8, COSMO.n_s = 0.82, 0.96
RHO_M0 = COSMO.critical_density(0).to_value("Msun/Mpc^3") * COSMO.Om0   # phys z=0
RMIN_FIT = 0.2                                                          # phys Mpc
PRJ_FILE = ("/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/data/"
            "prj_params_DESY3_lss_lin_dep_getdist_v1.txt")
LBDBINS = np.array([20, 30, 45, 60, 500])
ZLO, ZHI = 0.20, 0.33                                                   # one clean z-bin

_e3 = np.exp(np.linspace(np.log(0.05), np.log(2.5), 11))
R_RHO = np.sqrt(_e3[:-1] * _e3[1:])                                     # phys Mpc
_e2 = np.exp(np.linspace(np.log(0.0323), np.log(30.0), 16))
R_SIG = np.sqrt(_e2[:-1] * _e2[1:])                                     # phys Mpc


def rhom_z(z):
    return COSMO.critical_density(0).to_value("Msun/Mpc^3") * (1 + z) ** 3 * COSMO.Om0


def build_2h(zc):
    kk = np.logspace(-3, 1, 120)                          # 1/Mpc
    zg = np.linspace(0.10, 0.85, 20)
    pk = PkGrid(cosmo=COSMO, backend="camb")
    Pk = np.array([pk(kk, zz) for zz in zg])
    th = TwoHaloTerm(kk, Pk, zvec=zg)
    th.deltasigma(R_SIG, zg[10])
    rm = rhom_z(zc)                                       # Msun/Mpc^3
    return dict(rho=rm * np.maximum(th.xi(R_RHO, zc), 0.0),          # Msun/Mpc^3 (~rho_m*b*xi)
                sig=rm * th.sigma(R_SIG, zc) / 1e12,                 # Msun/pc^2
                ds=rm * th.deltasigma(R_SIG, zc) / 1e12)             # Msun/pc^2


def mass_match_stack(lgM_sel, z_sel, lgM_all, z_all, prof_all, dm=0.1, dz=0.05):
    """Heidi's (log10 M, z)-matched stack (units pass through prof_all unchanged)."""
    mb = np.arange(lgM_all.min() - dm, lgM_all.max() + 2 * dm, dm)
    zb = np.arange(z_all.min() - dz, z_all.max() + 2 * dz, dz)
    out = np.zeros(prof_all.shape[1]); wn = 0.0
    for iz in range(len(zb) - 1):
        sz = (z_sel >= zb[iz]) & (z_sel < zb[iz + 1])
        if not sz.any():
            continue
        az = (z_all >= zb[iz]) & (z_all < zb[iz + 1])
        for iM in range(len(mb) - 1):
            w = np.sum(sz & (lgM_sel >= mb[iM]) & (lgM_sel < mb[iM + 1]))
            if w == 0:
                continue
            cell = az & (lgM_all >= mb[iM]) & (lgM_all < mb[iM + 1])
            if cell.any():
                out += np.mean(prof_all[cell], axis=0) * w; wn += w
    return out / wn


def _model_val(n, kind, r):
    return (n.density(r) if kind == "rho"
            else n.sigma(r) / 1e12 if kind == "sigma" else n.deltasigma(r) / 1e12)


def fit_c(R, y, kind, t2h, sheet=0.0):
    """Fit MAX(NFW_1h, b*2h)+sheet, R>0.2. Returns (M200, c200)."""
    ok = (R > RMIN_FIT) & np.isfinite(y) & (y > 0)
    if ok.sum() < 4:
        return np.nan, np.nan
    t = t2h[ok]

    def model(r, logM, c, b):
        return np.log10(np.maximum(_model_val(NfwProfile(10 ** logM, c, cosmo=COSMO),
                                              kind, r), b * t) + sheet)
    try:
        p, _ = curve_fit(model, R[ok], np.log10(y[ok]), p0=[14.3, 6.0, 3.0],
                         bounds=([12, 0.5, 0.0], [16, 25, 60]), maxfev=8000)
        return 10 ** p[0], p[1]
    except Exception:
        return np.nan, np.nan


def main():
    fl = FileLocs(machine="nersc")
    p = fitsio.read(fl.profile_output_fname)
    good = ((p["pid"] == -1) & (p["cosi"] >= 0) & (p["cosi"] <= 1)
            & ((p["redshift"] < 0.33) | (p["redshift"] > 0.37)) & (p["Mvir"] >= 1e13))
    Mvir = p["Mvir"][good].astype(float)
    lgM = np.log10(Mvir)
    zz = p["redshift"][good]
    prof = dict(rho=np.asarray(p["rho"])[good], sigma=np.asarray(p["Sigma"])[good],
                ds=np.asarray(p["DeltaSigma"])[good])

    # --- assign C19 HOD lambda_obs (no LSS correlation) ---
    rng = np.random.default_rng(42)
    prj = load_prj_posterior_mean(PRJ_FILE)
    ltrue = sample_lambda_true(Mvir, zz, rng=rng).astype(float)
    lobs, *_ = sample_lambda_obs(ltrue, zz, prj, rng=rng)
    print(f"assigned lambda_obs: <lobs>={lobs.mean():.1f}, frac>20={np.mean(lobs>20):.3f}")

    zc = 0.5 * (ZLO + ZHI)
    T = build_2h(zc)
    inz = (zz >= ZLO) & (zz < ZHI)
    tkey = {"rho": "rho", "sigma": "sig", "ds": "ds"}

    OBS = [("ds", R_SIG, r"$\Delta\Sigma$"), ("rho", R_RHO, r"$\rho$")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    lam_c = 0.5 * (LBDBINS[:-1] + np.minimum(LBDBINS[1:], 120))
    print(f"\nz in [{ZLO},{ZHI}]   (fit R>{RMIN_FIT} Mpc, MAX(NFW,2h))")
    for ax, (kind, R, name) in zip(axes, OBS):
        t2h = T[tkey[kind]]
        rows = {"c_sel": [], "c_matched": [], "c_massbin": [], "N": [], "logMbar": []}
        for il in range(len(LBDBINS) - 1):
            sel = inz & (lobs >= LBDBINS[il]) & (lobs < LBDBINS[il + 1])
            if sel.sum() < 50:
                for k in rows:
                    rows[k].append(np.nan)
                continue
            lgMbar = np.median(lgM[sel])
            # (1) naive lambda_obs-selected stack
            stack_sel = np.mean(prof[kind][sel], axis=0)
            # (2) (logM,z)-matched reference for the SAME selected sample
            stack_mat = mass_match_stack(lgM[sel], zz[sel], lgM, zz, prof[kind])
            # (3) narrow true-mass bin (+-0.05 dex) at the selected sample's median logM
            mb = inz & (np.abs(lgM - lgMbar) < 0.05)
            stack_mb = np.mean(prof[kind][mb], axis=0)
            sheet = (np.nanmedian(stack_sel[R > 5.0]) if kind == "sigma" else 0.0)
            _, c_sel = fit_c(R, stack_sel, kind, t2h, sheet)
            _, c_mat = fit_c(R, stack_mat, kind, t2h,
                             np.nanmedian(stack_mat[R > 5.0]) if kind == "sigma" else 0.0)
            _, c_mb = fit_c(R, stack_mb, kind, t2h,
                            np.nanmedian(stack_mb[R > 5.0]) if kind == "sigma" else 0.0)
            rows["c_sel"].append(c_sel); rows["c_matched"].append(c_mat)
            rows["c_massbin"].append(c_mb); rows["N"].append(int(sel.sum()))
            rows["logMbar"].append(lgMbar)
            print(f"  [{name:>10}] lam[{LBDBINS[il]:>3},{LBDBINS[il+1]:>3}) N={sel.sum():6d} "
                  f"<logM>={lgMbar:.2f}  c_sel={c_sel:5.2f}  c_matched={c_mat:5.2f}  "
                  f"c_massbin={c_mb:5.2f}")
        ax.plot(lam_c, rows["c_sel"], "o-", color="crimson", ms=7, label="λ_obs-selected")
        ax.plot(lam_c, rows["c_matched"], "s--", color="royalblue", ms=6,
                label="(logM,z)-matched")
        ax.plot(lam_c, rows["c_massbin"], "^:", color="seagreen", ms=7,
                label="narrow true-mass bin")
        ax.set_xlabel(r"$\lambda_{\rm obs}$ bin center")
        ax.set_ylabel("fitted $c_{200}$")
        ax.set_title(f"{name}: concentration vs richness  (z∈[{ZLO},{ZHI}])", fontsize=12)
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.suptitle("Does lambda_obs (HOD, no LSS) selection change concentration? "
                 "CLensPy MAX(NFW,2h), R>0.2", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ("/pscratch/sd/j/jesteves/github/des-cluster-nersc/validations/buzzard/"
           "richness_vs_mass_concentration.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
