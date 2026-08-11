"""Buzzard polychord posterior corner plots (cosmology + HOD).

Converged chain (job 56588673, ndead=26212). Overlays fiducial truth lines so
the Buzzard recovery bias is visible directly. Uses the same ChainConsumer
conventions as chains/Plot_Polychord_widePlanck_Chains.ipynb.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

CHAIN = "/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/buzzard/chain.txt"
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(_HERE), "chains")

COLS = ['h0', 'omega_m', 'omega_b', 'n_s', 'sigma8',
        'log10_mmin', 'log10_ratio', 'alpha', 'epsilon', 'sigma_lambda',
        'prior', 'like', 'post', 'weight']
FID = dict(h0=0.6766, omega_m=0.311049, omega_b=0.048975, n_s=0.9665,
           sigma8=0.8238, log10_mmin=11.4, log10_ratio=1.3, log10_m1=12.7,
           alpha=0.86, epsilon=0.0, sigma_lambda=0.18)

df = pd.read_csv(CHAIN, sep=r'\s+', comment='#', header=None,
                 names=COLS, skiprows=3)
df['log10_m1'] = df['log10_mmin'] + df['log10_ratio']
w = df['weight'].values
ess = w.sum()**2 / (w**2).sum()
print(f"Buzzard chain: {len(df)} weighted samples, ESS ~ {ess:.0f}")


def corner(cols, labels, fname):
    c = ChainConsumer()
    c.add_chain(df[cols].values, parameters=labels, weights=w, name='Buzzard')
    c.configure(plot_hists=True, kde=1.5, colors=['#d1495b'],
                summary=True, linewidths=1.0, shade_alpha=0.4)
    c.configure_truth(color='k', ls='--', lw=1.2, alpha=0.8)
    out = os.path.join(OUTDIR, fname)
    c.plotter.plot(figsize=1.0, truth=[FID[col] for col in cols], filename=out)
    plt.close('all')
    # weighted mean/std + pull
    for col in cols:
        x = df[col].values
        m = np.average(x, weights=w)
        s = np.sqrt(np.average((x - m)**2, weights=w))
        print(f"  {col:14s} {m:8.3f} +/- {s:6.3f}   fid {FID[col]:7.3f}   pull {(m-FID[col])/s:+5.1f}")
    print(f"  -> wrote {out}\n")


print("\n=== cosmology ===")
corner(['omega_m', 'sigma8', 'omega_b', 'n_s', 'h0'],
       [r'$\Omega_m$', r'$\sigma_8$', r'$\Omega_b$', r'$n_s$', r'$h_0$'],
       'polychord_buzzard_cosmo.png')

print("=== HOD ===")
corner(['log10_mmin', 'log10_m1', 'alpha', 'sigma_lambda', 'epsilon'],
       [r'$\log_{10}M_{\min}$', r'$\log_{10}M_1$', r'$\alpha$',
        r'$\sigma_\lambda$', r'$\epsilon$'],
       'polychord_buzzard_HOD.png')
