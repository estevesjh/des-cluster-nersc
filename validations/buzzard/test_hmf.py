"""Buzzard validation 1/4: halo mass function (HMF).

Compare the pipeline Tinker HMF (MfTinker / halo_model, at Buzzard fiducial
cosmology) to the Buzzard halo-catalog binned dn/dlogM, over the DES-Y3 Buzzard
sample volume (sky fraction x seam-excised comoving shell). Plot + check the
ratio is within the (Poisson) covariance.

Pipeline HMF: run any Buzzard sampler=test and point --db at its save_dir; this
reads mass_function/{m_h, z, dndlnmh} (dn/dlnM [ (Mpc/h)^-3 ] on a (z, m_h) grid)
and volume-averages it over z in [0.2,0.65] \ [0.33,0.37], dV/dz weighted.

Buzzard: host halos (pid==-1) from the profile catalog, same selection as the DV.

*** CAVEAT (found 2026-08): the available FileLocs catalogs are NOT volume-complete
halo catalogs -- the profile catalog (~580k hosts >=1e13) and the halo_run/redMaPPer
catalog (~597k) are the CLUSTER/redMaPPer-matched sample, not the full rockstar
lightcone. Against them Buzzard sits ~4-5x below Tinker at 1e13 growing to ~15-50x at
1e14.5 (mass-growing deficit; units ×h vs not shift it only slightly). That is a
SELECTION/completeness artifact, not a physics HMF mismatch. A clean HMF test needs
the full rockstar halo catalog + a confirmed mass definition (M200m vs Mvir; M200m~1.05
Mvir here) and mass units (M_sun vs M_sun/h). Until that catalog is located, this test
is infrastructure only; the plot shows the (uncorrected) comparison. ***
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

# Buzzard fiducial cosmology (DeRose+2019)
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.286, Ob0=0.047, Tcmb0=2.725)
H = COSMO.h
SKY_AREA_DEG2 = 4143.0                 # DES Y3 Buzzard area
SKY_FRAC = SKY_AREA_DEG2 / 41252.96
ZMIN, ZMAX = 0.2, 0.65
SEAM_LO, SEAM_HI = 0.33, 0.37
DLOGM = 0.1
MASS_EDGES = np.arange(13.0, 15.6, DLOGM)


def comoving_volume_shell(zlo, zhi):
    """Full-sky comoving volume between zlo,zhi in (Mpc/h)^3."""
    v = (COSMO.comoving_volume(zhi).to("Mpc3").value
         - COSMO.comoving_volume(zlo).to("Mpc3").value)
    return v * H ** 3


def sample_volume():
    return SKY_FRAC * (comoving_volume_shell(ZMIN, ZMAX)
                       - comoving_volume_shell(SEAM_LO, SEAM_HI))


def buzzard_dndlogM():
    """Binned Buzzard dn/dlogM [(Mpc/h)^-3 dex^-1] + Poisson error + centers."""
    import sys
    import fitsio
    sys.path.insert(0, "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/src")
    from fileLoc import FileLocs
    d = fitsio.read(FileLocs(machine="nersc").profile_output_fname)
    sel = ((d["pid"] == -1) & (d["cosi"] >= 0) & (d["cosi"] <= 1)
           & ((d["redshift"] < SEAM_LO) | (d["redshift"] > SEAM_HI))
           & (d["redshift"] >= ZMIN) & (d["redshift"] <= ZMAX))
    logM_h = np.log10(np.asarray(d["Mvir"][sel]) * H)     # log10(M h/Msun)
    V = sample_volume()
    N, _ = np.histogram(logM_h, bins=MASS_EDGES)
    centers = 0.5 * (MASS_EDGES[:-1] + MASS_EDGES[1:])
    return centers, N / V / DLOGM, np.sqrt(N) / V / DLOGM, N, V


def pipeline_dndlogM(db, centers):
    """Volume-averaged pipeline dn/dlogM at the bin centers [(Mpc/h)^-3 dex^-1]."""
    d = os.path.join(db, "mass_function")
    m = np.loadtxt(d + "/m_h.txt")                 # (Nm,) Msun/h
    z = np.loadtxt(d + "/z.txt")                    # (Nz,)
    dndlnm = np.loadtxt(d + "/dndlnmh.txt")         # (Nz, Nm) dn/dlnM
    # dV/dz weights over the seam-excised sample z-range
    inz = (z >= ZMIN) & (z <= ZMAX) & ((z < SEAM_LO) | (z > SEAM_HI))
    dVdz = (COSMO.differential_comoving_volume(z).to("Mpc3/sr").value
            * H ** 3)                               # (Mpc/h)^3 / sr  (sky-frac cancels in the avg)
    w = np.where(inz, dVdz, 0.0)
    # volume-average dn/dlnM over z, then -> dn/dlogM
    dndlnm_avg = (w[:, None] * dndlnm).sum(0) / w.sum()          # (Nm,)
    dndlogM = dndlnm_avg * np.log(10.0)
    # interpolate (log-log) to the Buzzard bin centers
    return np.exp(np.interp(centers, np.log10(m), np.log(dndlogM)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="sampler=test save_dir (Buzzard fiducial)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "hmf_buzzard.png"))
    args = ap.parse_args()

    centers, buzz, buzz_err, N, V = buzzard_dndlogM()
    pipe = pipeline_dndlogM(args.db, centers)
    print(f"sample volume = {V:.3e} (Mpc/h)^3 ; N halos = {int(N.sum())}")
    good = N > 0
    ratio = buzz[good] / pipe[good]
    # chi2 within Poisson covariance (log-space to handle the dynamic range)
    chi2 = np.sum(((buzz[good] - pipe[good]) / buzz_err[good]) ** 2)
    print(f"chi2(Buzzard vs pipeline Tinker) = {chi2:.1f}/{good.sum()} "
          f"= {chi2/good.sum():.2f}/dof")
    for c, b, p, e in zip(centers[good], buzz[good], pipe[good], buzz_err[good]):
        print(f"  logM={c:.2f}  buzz={b:.3e}  pipe={p:.3e}  ratio={b/p:.3f}  "
              f"({(b-p)/e:+.1f} sigma)")

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.4, 1]})
    ax.errorbar(10 ** centers[good], buzz[good], yerr=buzz_err[good], fmt="o",
                color="k", ms=5, capsize=3, label="Buzzard (Poisson)")
    ax.plot(10 ** centers[good], pipe[good], "-", color="crimson", lw=2,
            label="pipeline Tinker HMF")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_ylabel(r"dn/dlogM  [(Mpc/h)$^{-3}$ dex$^{-1}$]")
    ax.legend(); ax.set_title(f"Buzzard HMF vs pipeline Tinker  "
                              f"($\\chi^2$={chi2:.0f}/{good.sum()}={chi2/good.sum():.2f}/dof)")
    axr.errorbar(10 ** centers[good], ratio, yerr=buzz_err[good] / pipe[good],
                 fmt="o", color="k", ms=5, capsize=3)
    axr.axhline(1.0, color="crimson", lw=1)
    axr.set_ylabel("Buzzard / pipeline"); axr.set_xscale("log")
    axr.set_xlabel(r"M$_{vir}$ [M$_\odot$/h]"); axr.set_ylim(0.5, 1.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
