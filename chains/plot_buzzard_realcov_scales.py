"""Full vs small vs large Buzzard cosmology recovery under the REALISTIC cov.

Overlays the three converged realcov runs (full 120 shear, small [0.2,1.2],
large [1.2,5.0]) vs fiducial truth, weighted (ChainConsumer). The scale
diagnostic: do small and large agree now that errors are realistic, and does
the full run recover fiducial? (Supersedes the tight-cov split, which railed.)
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


runs = {
    'full [0.2-5]':  load(f"{CH}/buzzard_realcov/chain.txt"),
    'small [0.2-1.2]': load(f"{CH}/buzzard_realcov_small/chain.txt"),
    'large [1.2-5]': load(f"{CH}/buzzard_realcov_large/chain.txt"),
}
colors = ['#2a2a2a', '#d1495b', '#3d7dc9']

for name, df in runs.items():
    ess = df['weight'].sum()**2 / (df['weight']**2).sum()
    print(f"{name:16s}: {len(df)} samples, ESS ~ {ess:.0f}")


def summ(df, cols):
    c = ChainConsumer(); c.add_chain(df[cols].values, parameters=cols, weights=df['weight'].values)
    c.configure(summary=True)
    return c.analysis.get_summary()


def corner(cols, labels, fname, title):
    c = ChainConsumer()
    for (name, df), col in zip(runs.items(), colors):
        c.add_chain(df[cols].values, parameters=labels, weights=df['weight'].values, name=name)
    c.configure(plot_hists=True, kde=1.5, colors=colors, shade_alpha=0.3,
                linewidths=1.1, summary=False, legend_kwargs={"fontsize": 11})
    c.configure_truth(color='green', ls='--', lw=1.4)
    out = os.path.join(_HERE, fname)
    c.plotter.plot(figsize=1.0, truth=[FID[x] for x in cols], filename=out)
    plt.close('all'); print(f"wrote {out}")
    # summary table
    print(f"\n{title}: median (+/-68%) vs fiducial")
    print(f"{'param':12s}{'fiducial':>10s}" + "".join(f"{n:>22s}" for n in runs))
    for k in cols:
        row = f"{k:12s}{FID[k]:10.3f}"
        for name, df in runs.items():
            s = summ(df, cols)[k]
            if s[0] is None or s[2] is None:
                row += f"{('%.3f(rail)' % s[1]):>22s}"
            else:
                row += f"{('%.3f +%.3f/-%.3f' % (s[1], s[2]-s[1], s[1]-s[0])):>22s}"
        print(row)


corner(['omega_m', 'sigma8', 'omega_b', 'n_s', 'h0'],
       [r'$\Omega_m$', r'$\sigma_8$', r'$\Omega_b$', r'$n_s$', r'$h_0$'],
       'buzzard_realcov_scales_cosmo.png', 'COSMOLOGY')
corner(['log10_mmin', 'log10_m1', 'alpha', 'sigma_lambda', 'epsilon'],
       [r'$\log_{10}M_{\min}$', r'$\log_{10}M_1$', r'$\alpha$',
        r'$\sigma_\lambda$', r'$\epsilon$'],
       'buzzard_realcov_scales_HOD.png', 'HOD')
