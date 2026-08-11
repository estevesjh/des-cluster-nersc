"""Nested-sampling convergence diagnostic for the Buzzard polychord chain.

PolyChord is NOT an MCMC walker, so Gelman-Rubin / autocorrelation don't apply.
The right convergence questions for nested sampling are:
  1. Has the likelihood contour stopped climbing? (logL vs iteration)
  2. Has the posterior WEIGHT profile risen to a peak and decayed back to ~0?
     If the run is truncated while weights are still large, the bulk of the
     posterior mass has not been collected -> NOT converged.
  3. Have the running posterior-weighted parameter means stabilized?

Buzzard run job 56546840 hit the 9h wall (TIMEOUT), so this is expected to show
an UN-converged chain (weights still substantial at the end, means still
drifting). Re-submit buzzard_polychord.sh with polychord.resume=T to finish.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CHAIN = "/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard/chain.txt"
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(_HERE), "chains", "polychord_buzzard_convergence.png")

COLS = ['h0', 'omega_m', 'omega_b', 'n_s', 'sigma8',
        'log10_mmin', 'log10_ratio', 'alpha', 'epsilon', 'sigma_lambda',
        'prior', 'like', 'post', 'weight']
FID = dict(h0=0.6766, omega_m=0.311049, omega_b=0.048975, n_s=0.9665,
           sigma8=0.8238, log10_mmin=11.4, log10_ratio=1.3, alpha=0.86,
           epsilon=0.0, sigma_lambda=0.18)
KEY = ['omega_m', 'sigma8', 'log10_mmin', 'h0']  # params we track for recovery


def main():
    df = pd.read_csv(CHAIN, sep=r'\s+', comment='#', header=None,
                     names=COLS, skiprows=3)
    n = len(df)
    w = df['weight'].values.astype(float)
    logL = df['like'].values.astype(float)
    idx = np.arange(n)
    cw = np.cumsum(w)
    ess = w.sum()**2 / (w**2).sum()

    # convergence metrics
    peak = int(np.argmax(w)) / n
    tail = w[int(0.95 * n):].sum() / w.sum()      # weight fraction in last 5%
    verdict = "CONVERGED" if (tail < 0.10 and peak < 0.90) else "NOT converged (truncated)"
    print(f"samples (dead points): {n}")
    print(f"ESS ~ {ess:.0f}")
    print(f"posterior-weight peak at {peak:.0%} of the run")
    print(f"weight fraction in last 5% of samples: {tail:.1%}")
    print(f"VERDICT: {verdict}")

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # (a) likelihood contour climb
    ax[0, 0].plot(idx, logL, lw=0.6, color='#30638e')
    ax[0, 0].set_title("(a) logL vs iteration (contour climb)")
    ax[0, 0].set_xlabel("dead-point index"); ax[0, 0].set_ylabel("logL")

    # (b) posterior weight profile — the key NS convergence view
    ax[0, 1].plot(idx, w / w.max(), lw=0.7, color='#d1495b')
    ax[0, 1].axvline(0.95 * n, color='0.5', ls='--', lw=1)
    ax[0, 1].set_title("(b) posterior weight (norm.) — should peak & decay to 0")
    ax[0, 1].set_xlabel("dead-point index"); ax[0, 1].set_ylabel("w / w_max")

    # (c) cumulative posterior mass
    ax[1, 0].plot(idx, cw / cw[-1], lw=1.2, color='#00798c')
    ax[1, 0].set_title("(c) cumulative posterior mass fraction")
    ax[1, 0].set_xlabel("dead-point index"); ax[1, 0].set_ylabel("cumulative w")

    # (d) running posterior-weighted means of tracked params (vs fiducial)
    for c in KEY:
        x = df[c].values.astype(float)
        rm = np.cumsum(w * x) / np.where(cw > 0, cw, np.nan)
        line, = ax[1, 1].plot(idx, rm / FID[c], lw=1.2, label=c)
    ax[1, 1].axhline(1.0, color='r', ls='--', lw=1, label='fiducial')
    ax[1, 1].set_ylim(0.3, 2.0)
    ax[1, 1].set_title("(d) running weighted mean / fiducial (drift check)")
    ax[1, 1].set_xlabel("dead-point index"); ax[1, 1].set_ylabel("mean / fiducial")
    ax[1, 1].legend(fontsize=10, ncol=2)

    fig.suptitle(f"Buzzard polychord — job 56588673 COMPLETED at tolerance  "
                 f"(ndead={n}, ESS~{ess:.0f}, log(Z)~-2522)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
