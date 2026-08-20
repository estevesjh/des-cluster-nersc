#!/usr/bin/env python
"""Remake of chains/buzzard_realcov_nc.png with the corrected inputs
(owner request, 2026-08-20).

What changed vs the original:
  * DATA: refreshed NC from Tan Xing's CURRENT shipped npz
    (xtang126/mock_cluster_buzzard output/MockDataVector.npz) on her
    seam-excluding observed bins [0.20,0.33) [0.37,0.50) [0.50,0.65)
    -- the old dv_buzzard_* NC was a stale revision (x1.186 high).
  * MODEL: the pipeline at the FIDUCIAL point (widePlanck values) under
    the corrected Buzzard config (mock_mcmc_cp_camb_buzzard.ini):
    seam-excluding zob edges + true-z seam excision + cp_camb nz=400 +
    z_halo=0.4 + Child18 projection concentration. This is a PRE-FLIGHT
    check (no chain): the original plot showed the BEST-FIT model, so
    ratios are not directly comparable -- here a good model shows
    data/model ~ 1 WITHOUT parameter shifts.
  * SIGMA: Matteo's realistic NC covariance (dv_buzzard_realcov.npz
    invcov_NC), same as the original.

Inputs: the fiducial dump written by

    cosmosis cosmosis-models/mock_mcmc_cp_camb_buzzard.ini \
        -p runtime.sampler=test test.save_dir=<dump> test.save=T [...]

Run:  python chains/plot_buzzard_nc_zfix.py <dump_dir>
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MOCK_NPZ = os.path.expanduser(
    "~/Documents/Dev/github/mock_cluster_buzzard/output/MockDataVector.npz")
REALCOV = os.path.join(ROOT, "data", "mock", "dv_buzzard_realcov.npz")
OUT = os.path.join(HERE, "buzzard_realcov_nc_zfix.png")

LAM_LABELS = ["20-30", "30-45", "45-60", ">60"]


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else None
    assert dump and os.path.isdir(dump), "usage: plot_buzzard_nc_zfix.py <dump>"

    model = np.loadtxt(os.path.join(dump, "numcountssel", "vals.txt"))
    d = np.load(MOCK_NPZ)
    data = d["NC"].T.reshape(-1)                      # bin-major (z, lam)
    z_lo, z_hi = d["z_bin_min"], d["z_bin_max"]
    zlabels = [f"{a:.2g}-{b:.2g}" for a, b in zip(z_lo, z_hi)]

    rc = np.load(REALCOV)
    sig = np.sqrt(np.diag(np.linalg.inv(np.asarray(rc["invcov_NC"], float))))
    invcov = np.asarray(rc["invcov_NC"], float)

    delta = data - model
    chi2 = float(delta @ invcov @ delta)
    pulls = delta / sig

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.5, 8), gridspec_kw={"height_ratios": [2.2, 1]})

    x = np.arange(12)
    ax1.errorbar(x - 0.08, data, yerr=sig, fmt="o", color="#c94040",
                 label=r"Buzzard data (xtang126 current npz, $\pm$Matteo $\sigma$)")
    ax1.plot(x + 0.08, model, "s", ms=9, color="#33608c",
             label="fiducial model, corrected Buzzard config (pre-flight)")
    for i in range(12):
        ax1.annotate(f"{data[i]/model[i]:.2f}",
                     (x[i], max(data[i], model[i]) * 1.12),
                     ha="center", fontsize=9)
    ax1.set_yscale("log")
    ax1.set_ylabel("number counts")
    ax1.set_title("Buzzard NC: current data vs FIDUCIAL model, corrected "
                  f"config (pre-flight, not best-fit)   "
                  f"$\\chi^2$={chi2:.0f}/12 = {chi2/12:.1f}/dof")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{LAM_LABELS[i % 4]}\n{zlabels[i // 4]}"
                         for i in range(12)], fontsize=9)
    ax1.legend()

    colors = ["#c94040" if abs(p) > 1 else "#33608c" for p in pulls]
    ax2.axhspan(-1, 1, color="0.85", zorder=0)
    ax2.bar(x, pulls, color=colors)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel(r"(data$-$model)/$\sigma$")
    ax2.set_title(r"NC pull per bin (grey = $\pm1\sigma$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{LAM_LABELS[i % 4]}\n{zlabels[i // 4]}"
                         for i in range(12)], fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)
    print("chi2 =", round(chi2, 1), "/ 12")
    print("data/model:", " ".join(f"{v:.2f}" for v in data / model))


if __name__ == "__main__":
    main()
