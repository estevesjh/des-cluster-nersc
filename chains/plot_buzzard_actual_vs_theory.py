"""Actual Buzzard DV (gamma_t_obs_C1 / scinv_bin -> physical DeltaSigma) vs the
pipeline theory at the Buzzard cosmology (cp_camb). Diagnoses the theory-vs-data
offset: a ~h^2 amplitude gap in the 1-halo regime + a large-R 2-halo tilt.

Data:   MockDataVector_scinv.npz, gamma_t_obs_C1 (physical Msun/pc^2 after /scinv),
        radii_phys_mpc (physical Mpc). Comoving radii = R_phys*(1+z_rep).
Theory: dvtest_buzz_cpcamb (pipeline units, geomspace(0.2,5,10)).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
CH = "/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc"
MC = "/pscratch/sd/j/jesteves/github/mock_cluster_buzzard/output/MockDataVector_scinv.npz"
N_BINS, N_R = 12, 10
Rpipe = np.geomspace(0.2, 5.0, N_R)
LBD = ["20-30", "30-45", "45-60", ">60"]; ZED = ["0.2-0.33", "0.37-0.5", "0.5-0.65"]
zrep = np.array([0.27, 0.44, 0.573])

col = lambda d, s: np.loadtxt(f"{CH}/{d}/{s}/vals.txt", comments="#").ravel()
T = "dvtest_buzz_cpcamb"
nc = col(T, "numcountssel")
th = (col(T, "shear1hmissel") / np.repeat(nc, N_R) + col(T, "shear_prj")).reshape(N_BINS, N_R)

d = np.load(MC)
Rphys = d["radii_phys_mpc"]
DS = np.transpose(d["gamma_t_obs_C1"] / d["scinv_bin"][:, :, None], (1, 0, 2)).reshape(N_BINS, -1)

def ib(Rs, y, Rd):
    return np.exp(np.interp(np.log(Rd), np.log(Rs), np.log(y)))

# data on the pipeline grid, comoving radii R_phys*(1+z_rep)
data_grid = np.zeros((N_BINS, N_R))
for b in range(N_BINS):
    data_grid[b] = ib(Rphys * (1 + zrep[b // 4]), DS[b], Rpipe)
ratio = th / data_grid

fig = plt.figure(figsize=(16, 12)); gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.32)
for b in range(N_BINS):
    a = fig.add_subplot(gs[b // 4, b % 4])
    a.plot(Rpipe, data_grid[b], "o-", ms=4, lw=1.3, color="#c94f4f", label="Buzzard data (ΔΣ=γt/scinv)")
    a.plot(Rpipe, th[b], "s-", ms=4, lw=1.3, color="#3d7dc9", label="theory (cp_camb, Buzzard cosmo)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_title(f"λ {LBD[b % 4]}, z {ZED[b // 4]}", fontsize=8)
    if b % 4 == 0: a.set_ylabel(r"$\Delta\Sigma$", fontsize=9)
    if b == 0: a.legend(fontsize=6.5)

axr = fig.add_subplot(gs[3, :2])
for b in range(N_BINS): axr.plot(Rpipe, ratio[b], "-", lw=0.6, alpha=0.4)
axr.plot(Rpipe, np.median(ratio, 0), "k-", lw=2.4, label="median")
axr.axhline(0.49, color="orange", ls="--", lw=1.5, label=r"$h^2=0.49$")
axr.axhline(1.0, color="0.5", lw=1); axr.set_xscale("log")
axr.set_xlabel("R [Mpc/h]"); axr.set_ylabel("theory / data"); axr.legend(fontsize=8)
axr.set_title("ratio: flat ~$h^2$ at small R (units), rises at large R (2-halo)", fontsize=10)

axc = fig.add_subplot(gs[3, 2:])
b0 = 0
axc.plot(Rpipe, (col(T, "shear1hmissel") / np.repeat(nc, N_R)).reshape(N_BINS, N_R)[b0], "b-o", ms=3, label="theory 1-halo")
axc.plot(Rpipe, col(T, "shear_prj").reshape(N_BINS, N_R)[b0], "g-s", ms=3, label="theory 2-halo")
axc.plot(Rpipe, data_grid[b0], "r-^", ms=3, label="data total")
axc.set_xscale("log"); axc.set_yscale("log")
axc.set_xlabel("R [Mpc/h]"); axc.set_ylabel(r"$\Delta\Sigma$")
axc.set_title(f"term decomposition, λ{LBD[0]} z{ZED[0]}", fontsize=10); axc.legend(fontsize=8)

fig.suptitle("Actual Buzzard DV (gamma_t_obs_C1/scinv) vs pipeline theory at Buzzard cosmology", fontsize=13, y=0.995)
out = f"{_HERE}/buzzard_actual_vs_theory.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
print("median theory/data by R:")
for i, R in enumerate(Rpipe):
    print(f"  R={R:5.2f}  {np.median(ratio[:,i]):.3f}")
print(f"wrote {out}")
