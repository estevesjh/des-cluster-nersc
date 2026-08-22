"""Buzzard DeltaSigma vs theory: FIXED concentration + physical (1+z)^3 density.

Re-make of buzzard_vs_theory with the report's conclusions:
  - FIXED concentration c = 5 (physical M200m; = Child18 x 1.25), single value, NO z-split;
  - physical mean density rho_m(z) = rho_m0 (1+z)^3 in the NFW (the "(1+z)^3" the comoving
    pipeline omits). A comoving (frozen z=0) curve is overlaid dashed for contrast.

Data: the lambda_obs DeltaSigma data vector (Heidi matched stack, pipeline units)
      data/mock/dv_buzzard_deltasigma_heidi.npz  (DeltaSigma little-h, radii Mpc/h, JK cov).
Theory: physical NFW (Wright&Brainerd) DeltaSigma at fixed c + CLensPy 2-halo; per bin
        fit only (M200, b_cls) over R>0.4 Mpc/h. chi^2 uses the JK block covariance.

UNITS: R_phys[Mpc] = R_data[Mpc/h] / h ; DeltaSigma_littleh = DeltaSigma_phys * (1/h).
"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.cosmology import FlatLambdaCDM
import sys
sys.path.insert(0, "/pscratch/sd/j/jesteves/github/CLensPy/src")
from clenspy.halo import TwoHaloTerm
from clenspy.cosmology import PkGrid

warnings.filterwarnings("ignore")
COSMO = FlatLambdaCDM(H0=70, Om0=0.286, Ob0=0.046)
COSMO.sigma8, COSMO.n_s = 0.82, 0.96
H = 0.70
RHO_M0 = COSMO.critical_density(0).to_value("Msun/Mpc^3") * COSMO.Om0   # physical z=0
C_FIX = 5.0                                                # fixed physical concentration
RMIN = 0.4                                                 # trusted radii [Mpc/h]
DV = "/pscratch/sd/j/jesteves/github/des-cluster-nersc/data/mock/dv_buzzard_deltasigma_heidi.npz"
LBD = [20, 30, 45, 60, 500]
ZC = [0.265, 0.435, 0.575]                                # z-bin centers (z-major)


def _mu(x):
    return np.log(1 + x) - x / (1 + x)


def nfw_ds_phys(Rp, M, c, z, physical=True):
    """Wright&Brainerd NFW DeltaSigma [Msun/Mpc^2]; M [Msun], Rp [physical Mpc].
    physical=True uses rho_m(z)=rho_m0(1+z)^3 (the (1+z)^3 density); False = comoving z=0."""
    rho_ref = RHO_M0 * (1 + z) ** 3 if physical else RHO_M0
    R200 = (3 * M / (4 * np.pi * 200 * rho_ref)) ** (1.0 / 3.0)
    rs = R200 / c
    rho_s = (200.0 / 3.0) * rho_ref * c ** 3 / _mu(c)
    x = np.atleast_1d(Rp / rs).astype(float)
    lo, hi = x < 1 - 1e-6, x > 1 + 1e-6
    mid = ~(lo | hi)
    f = np.empty_like(x); h_ = np.empty_like(x)
    xl = x[lo]; s = np.sqrt(1 - xl ** 2); at = np.arctanh(np.sqrt((1 - xl) / (1 + xl)))
    f[lo] = (1 - 2 / s * at) / (xl ** 2 - 1); h_[lo] = 2 / s * at + np.log(xl / 2)
    xh = x[hi]; s2 = np.sqrt(xh ** 2 - 1); at2 = np.arctan(np.sqrt((xh - 1) / (1 + xh)))
    f[hi] = (1 - 2 / s2 * at2) / (xh ** 2 - 1); h_[hi] = 2 / s2 * at2 + np.log(xh / 2)
    f[mid] = 1.0 / 3.0; h_[mid] = 1.0 + np.log(0.5)
    return rho_s * rs * ((4.0 / x ** 2) * h_ - 2.0 * f)


def main():
    d = np.load(DV, allow_pickle=True)
    R = d["radii"]                                         # Mpc/h
    data = d["data_Shear"].reshape(12, 10)                 # DeltaSigma little-h
    cov = d["cov_Shear"]
    Rp = R / H                                             # physical Mpc
    th2h = {}
    kk = np.logspace(-3, 1, 120); zg = np.linspace(0.10, 0.85, 20)
    pk = PkGrid(cosmo=COSMO, backend="camb")
    TH = TwoHaloTerm(kk, np.array([pk(kk, zz) for zz in zg]), zvec=zg)
    for z in ZC:
        rm = RHO_M0 * (1 + z) ** 3
        th2h[z] = rm * TH.deltasigma(Rp, z) / 1e12 * (1.0 / H)   # little-h Msun/pc^2, bias=1

    keep = R > RMIN
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    chi2_tot, ndof = 0.0, 0
    print(f"fixed c={C_FIX} (physical), (1+z)^3 density, fit (M,b) per bin, R>{RMIN} Mpc/h")
    for b in range(12):
        iz, il = b // 4, b % 4
        ax = axes[iz, il]
        z = ZC[iz]
        y = data[b]
        C = cov[b * 10:(b + 1) * 10, b * 10:(b + 1) * 10]
        Ck = C[np.ix_(np.where(keep)[0], np.where(keep)[0])]
        iCk = np.linalg.inv(Ck)
        t2 = th2h[z]

        def model(p, physical=True):
            M, bb = 10 ** p[0], p[1]
            return nfw_ds_phys(Rp, M, C_FIX, z, physical) / 1e12 * (1 / H) + bb * t2

        def chi2(p):
            r = (y - model(p))[keep]
            return r @ iCk @ r
        res = minimize(chi2, x0=[14.5, 3.0], method="Nelder-Mead",
                       options=dict(xatol=1e-3, fatol=1e-3, maxiter=2000))
        M, bb = 10 ** res.x[0], res.x[1]
        thy = model(res.x); thy_com = model(res.x, physical=False)
        c2 = res.fun; chi2_tot += c2; ndof += keep.sum() - 2
        thd = np.median((thy / y)[keep])

        ax.errorbar(R[keep], y[keep], yerr=np.sqrt(np.diag(C))[keep], fmt="o",
                    color="k", ms=4, capsize=2, label="Buzzard ΔΣ (JK)")
        ax.errorbar(R[~keep], y[~keep], yerr=np.sqrt(np.diag(C))[~keep], fmt="o",
                    mfc="none", mec="0.6", ms=4)
        ax.loglog(R, thy, "-", color="crimson", lw=2,
                  label="NFW c=5 + (1+z)³ + 2h" if b == 0 else None)
        ax.loglog(R, thy_com, "--", color="royalblue", lw=1.3,
                  label="same, comoving (no (1+z)³)" if b == 0 else None)
        ax.axvspan(R[0] * 0.8, RMIN, color="0.9", zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.text(0.04, 0.06, f"z {ZC[iz]:.2f}\nλ[{LBD[il]},{LBD[il+1] if LBD[il+1]<500 else 999})\n"
                f"logM={np.log10(M):.2f}\nth/dat={thd:.2f}", transform=ax.transAxes,
                fontsize=8, va="bottom")
        if iz == 2:
            ax.set_xlabel("R [Mpc/h]")
        if il == 0:
            ax.set_ylabel(r"$\Delta\Sigma$ [h M$_\odot$/pc$^2$]")
        if b == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Buzzard ΔΣ vs theory — FIXED c={C_FIX} + physical (1+z)³ density "
                 f"(fit M,b; R>{RMIN} Mpc/h)   χ²={chi2_tot:.0f}/{ndof}={chi2_tot/ndof:.2f}/dof",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ("/pscratch/sd/j/jesteves/github/des-cluster-nersc/chains/"
           "buzzard_vs_theory_fixedc_1pz3.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"chi2 total = {chi2_tot:.0f}/{ndof} = {chi2_tot/ndof:.2f}/dof")
    print("wrote", out)


if __name__ == "__main__":
    main()
