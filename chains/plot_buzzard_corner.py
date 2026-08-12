"""Buzzard cosmology + HOD corner plots (unfixed vs h-fixed), notebook method.

Uses the same weighted ChainConsumer handling as
Plot_Polychord_widePlanck_Chains.ipynb (chain.txt 'weight' column). Overlays the
unfixed Buzzard run and the h-unit-fixed run vs fiducial truth. Marginalized
constraints via ChainConsumer -- the .stats 'Sigma' column is NOT the posterior
width and should not be used.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

_HERE = os.path.dirname(os.path.abspath(__file__))
CH = "/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc"
COLS = ['h0', 'omega_m', 'omega_b', 'n_s', 'sigma8',
        'log10_mmin', 'log10_ratio', 'alpha', 'epsilon', 'sigma_lambda',
        'prior', 'like', 'post', 'weight']
FID = dict(h0=0.6766, omega_m=0.311049, omega_b=0.048975, n_s=0.9665,
           sigma8=0.8238, log10_mmin=11.4, log10_ratio=1.3, log10_m1=12.7,
           alpha=0.86, epsilon=0.0, sigma_lambda=0.18)


def load(p):
    df = pd.read_csv(p, sep=r'\s+', comment='#', header=None, names=COLS, skiprows=3)
    df['log10_m1'] = df['log10_mmin'] + df['log10_ratio']
    return df


chains = {
    'unfixed':  load(f"{CH}/buzzard/chain.txt"),
    'h-fixed':  load(f"{CH}/buzzard_hfix/chain.txt"),
}
colors = {'unfixed': '#c94f4f', 'h-fixed': '#3d7dc9'}

for name, df in chains.items():
    ess = df['weight'].sum()**2 / (df['weight']**2).sum()
    print(f"{name:8s}: {len(df)} weighted samples, ESS ~ {ess:.0f}")


def corner(cols, labels, truth, fname, title):
    c = ChainConsumer()
    for name, df in chains.items():
        c.add_chain(df[cols].values, parameters=labels,
                    weights=df['weight'].values, name=name)
    c.configure(plot_hists=True, kde=1.5, colors=[colors[n] for n in chains],
                summary=True, linewidths=1.0, shade_alpha=0.35)
    c.configure_truth(color='k', ls='--', lw=1.2, alpha=0.8)
    out = os.path.join(_HERE, fname)
    c.plotter.plot(figsize=1.0, truth=truth, filename=out)
    plt.close('all')
    print(f"wrote {out}")
    print(f"  {title} marginalized (h-fixed):")
    s = ChainConsumer(); s.add_chain(chains['h-fixed'][cols].values, parameters=cols,
                                     weights=chains['h-fixed']['weight'].values)
    s.configure(summary=True)
    summ = s.analysis.get_summary()
    for k in cols:
        lo, mid, hi = summ[k]
        if lo is None or hi is None:
            print(f"    {k:12s} = {mid:.4f}  (rail)   fid {FID[k]:.3f}")
        else:
            print(f"    {k:12s} = {mid:.4f} (+{hi-mid:.4f}/-{mid-lo:.4f})   fid {FID[k]:.3f}")


corner(['omega_m', 'sigma8', 'omega_b', 'n_s', 'h0'],
       [r'$\Omega_m$', r'$\sigma_8$', r'$\Omega_b$', r'$n_s$', r'$h_0$'],
       [FID[c] for c in ['omega_m', 'sigma8', 'omega_b', 'n_s', 'h0']],
       'polychord_buzzard_cosmo.png', 'cosmology')

corner(['log10_mmin', 'log10_m1', 'alpha', 'sigma_lambda', 'epsilon'],
       [r'$\log_{10}M_{\min}$', r'$\log_{10}M_1$', r'$\alpha$',
        r'$\sigma_\lambda$', r'$\epsilon$'],
       [FID[c] for c in ['log10_mmin', 'log10_m1', 'alpha', 'sigma_lambda', 'epsilon']],
       'polychord_buzzard_HOD.png', 'HOD')
