"""Validate the DeltaSigma jackknife covariance: fractional error vs N_patches.

Concern: is K_JK=50 enough, and are the patches large enough to be quasi-independent
(patch size must exceed the measurement's max scale, ~5 Mpc/h, with buffer -> ~20
Mpc/h)? Scan N_patches from 10 up to N(20 Mpc/h) and plot the JK fractional error of
DeltaSigma (at R~1 Mpc, unit-independent) per (lambda_obs, z) bin.

Expected: a plateau where the JK is converged and patches are independent; a DROP at
large N (patches < ~20 Mpc/h -> patches share correlated LSS -> JK UNDER-estimates).
Uses the naive lambda_obs stack (fast; frac-err-vs-N behavior governs the cov, and
for lambda_obs the mass-matched stack = naive to ~1-3%).
"""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/src")
import fitsio
from fileLoc import FileLocs
from astropy.cosmology import FlatLambdaCDM
from costanzi_selection import (sample_lambda_true, sample_lambda_obs,
                                load_prj_posterior_mean)
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
COSMO = FlatLambdaCDM(H0=70, Om0=0.286)
H = 0.70
LBDBINS = np.array([20, 30, 45, 60, 500])
ZMIN_LIST = np.array([0.20, 0.37, 0.50]); ZMAX_LIST = np.array([0.33, 0.50, 0.65])
PRJ_FILE = ("/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/data/"
            "prj_params_DESY3_lss_lin_dep_getdist_v1.txt")
_e2 = np.exp(np.linspace(np.log(0.0323), np.log(30.0), 16))
R_SIG = np.sqrt(_e2[:-1] * _e2[1:])
RI = int(np.argmin(np.abs(R_SIG - 1.0)))                  # radius ~1 Mpc for the frac-err


def main():
    fl = FileLocs(machine="nersc")
    p = fitsio.read(fl.profile_output_fname); h = fitsio.read(fl.halo_run_fname)
    good = ((p["pid"] == -1) & (p["cosi"] >= 0) & (p["cosi"] <= 1)
            & ((p["redshift"] < 0.33) | (p["redshift"] > 0.37))
            & (p["redshift"] >= 0.2) & (p["redshift"] <= 0.65)
            & (np.log10(p["Mvir"]) >= 13.0))
    Mvir = p["Mvir"][good].astype(float); z = p["redshift"][good].astype(float)
    DS = np.asarray(p["DeltaSigma"])[good][:, RI]         # (N,) at R~1 Mpc
    ra, dec = h["RA"][good].astype(float), h["DEC"][good].astype(float)

    rng = np.random.default_rng(42)
    prj = load_prj_posterior_mean(PRJ_FILE)
    lobs, *_ = sample_lambda_obs(sample_lambda_true(Mvir, z, rng=rng).astype(float),
                                 z, prj, rng=rng)

    # analysis bin index (lambda-major il*3+iz), and per-bin masks
    nl, nz = len(LBDBINS) - 1, len(ZMIN_LIST)
    binid = np.full(z.size, -1)
    labels = []
    for il in range(nl):
        for iz in range(nz):
            m = ((lobs >= LBDBINS[il]) & (lobs < LBDBINS[il + 1])
                 & (z >= ZMIN_LIST[iz]) & (z < ZMAX_LIST[iz]))
            binid[m] = il * nz + iz
            labels.append(f"λ[{int(LBDBINS[il])},{int(LBDBINS[il+1]) if LBDBINS[il+1]<500 else 999}) "
                          f"z[{ZMIN_LIST[iz]:.2f},{ZMAX_LIST[iz]:.2f})")

    # footprint area -> patch size L(N) [Mpc/h]
    cs = 0.2
    cells = len(set(zip((ra / cs).astype(int).tolist(), (dec / cs).astype(int).tolist())))
    Om = cells * cs * cs * np.mean(np.cos(np.radians(dec))) * (np.pi / 180) ** 2
    Dc_h = COSMO.comoving_distance(np.median(z)).to_value("Mpc") * H
    A = Om * Dc_h ** 2                                    # (Mpc/h)^2
    Nmax = int(A / 20.0 ** 2)                             # patches ~ 20 Mpc/h
    print(f"area={A:.2e} (Mpc/h)^2 ; N(20 Mpc/h)={Nmax}")

    Ns = np.array([10, 50, 100, 200, 500, 1000])
    fracerr = np.full((len(Ns), nl * nz), np.nan)
    for ii, Nc in enumerate(Ns):
        lab = MiniBatchKMeans(n_clusters=int(Nc), random_state=0, n_init=3,
                              batch_size=10000).fit_predict(np.column_stack([ra, dec]))
        for b in range(nl * nz):
            m = binid == b
            pk = lab[m]; ds = DS[m]
            psum = np.zeros(Nc); pcnt = np.zeros(Nc)
            np.add.at(psum, pk, ds); np.add.at(pcnt, pk, 1.0)
            tot_s, tot_c = ds.sum(), ds.size
            occ = pcnt > 0                                # patches with halos in this bin
            loo = (tot_s - psum[occ]) / (tot_c - pcnt[occ])   # leave-one-out means
            K = occ.sum()
            var = (K - 1) / K * np.sum((loo - loo.mean()) ** 2)
            fracerr[ii, b] = np.sqrt(var) / (tot_s / tot_c)
        print(f"  N={Nc:5d} (L={np.sqrt(A/Nc):5.1f} Mpc/h)  "
              f"mean fracerr={np.nanmean(fracerr[ii]):.4f}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.viridis
    for b in range(nl * nz):
        il = b // nz
        ax.plot(Ns, fracerr[:, b], "o-", ms=4, lw=1.2, color=cmap(il / (nl - 1)),
                label=labels[b] if b % nz == 0 else None)
    ax.axvline(50, color="crimson", ls="--", lw=1.5, label="K=50 (current)")
    ax.axvline(Nmax, color="k", ls=":", lw=1.5, label=f"20 Mpc/h ({Nmax} patches)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N patches"); ax.set_ylabel(r"JK fractional error of $\Delta\Sigma$ (R~1 Mpc)")
    ax.set_title("JK covariance validation: DeltaSigma frac-err vs N_patches "
                 "(color = richness bin)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, which="both")
    secx = ax.secondary_xaxis("top", functions=(lambda n: np.sqrt(A / np.clip(n, 1, None)),
                                                 lambda L: A / np.clip(L, 1e-6, None) ** 2))
    secx.set_xlabel("patch size L [Mpc/h]")
    out = "validations/buzzard/jk_convergence_deltasigma.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
