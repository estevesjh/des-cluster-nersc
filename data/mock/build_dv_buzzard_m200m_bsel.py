#!/usr/bin/env python
"""Buzzard M200m mock data vector WITH optical selection bias, for likelihood_cp (max-model mode).

Same as y3_cluster_cpp/data/mock/build_dv_buzzard_m200m.py (NC + Y1 covariance, shear covariance
diag((f_Y1 x data)^2) + C_JK), except that the stacked DeltaSigma is multiplied by the selection
bias the mock injects (xtang126/mock_cluster_buzzard MockDataVector.ipynb cell 37, Costanzi+2026
Eq. C1 with Matteo's DeltaSigma constants):

    B_C1(R) = 1 + A x^alpha (1 + x^gamma)^((beta - alpha)/gamma),  x = R_cMpc_h / R0,
    R0 = (lambda_rep/100)^0.2 (1 + z_rep),  (A, alpha, beta, gamma) = (0.12, 4.11, -0.18, 1.82),

with lambda_rep / z_rep the mean lambda_ob / z_ob of each (lambda_ob, z_ob) bin (replayed with the
notebook's seed; bin counts and z_rep match the notebook exactly).  B is deterministic, so the
jackknife covariance becomes B_i B_j C_ij.  No miscentering or boost: the only difference from
dv_buzzard_m200m_pipeline.npz is B_sel.

    python build_dv_buzzard_m200m_bsel.py <MockDataVector_pipeline.npz> <MockDataVector.npz> <out.npz>
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.environ.get("Y3_CLUSTER_CPP_DIR",
                                               "/pscratch/sd/j/jesteves/y3_cluster_cpp"), "data", "mock"))
import build_dv_buzzard_m200m as base  # noqa: E402

A, ALPHA, BETA, GAMMA = 0.12, 4.11, -0.18, 1.82
# mean lambda_ob per (lambda_ob, z_ob) bin, rows = lambda bins, cols = z bins (C19 sampler, seed 42)
LAM_REP = np.array([[23.926, 23.868, 23.831],
                    [35.837, 35.420, 35.596],
                    [51.214, 50.973, 50.733],
                    [82.388, 78.935, 75.719]])


def bsel_c1(r_com, lam, z):
    x = r_com / ((lam / 100.0) ** 0.2 * (1.0 + z))
    return 1.0 + A * x ** ALPHA * (1.0 + x ** GAMMA) ** ((BETA - ALPHA) / GAMMA)


def main(pipe_path, mock_path, out):
    p = dict(np.load(pipe_path))
    z_rep = np.load(mock_path)["z_rep_C19"]
    r = p["r_perp_cMpc_h"]
    b = np.concatenate([bsel_c1(r, LAM_REP[il, iz], z_rep[il, iz])
                        for iz in range(3) for il in range(4)])   # z-major, R fastest
    p["data_Shear"] = p["data_Shear"] * b
    p["cov_Shear"] = p["cov_Shear"] * np.outer(b, b)
    p["invcov_Shear"] = np.linalg.inv(p["cov_Shear"])
    with tempfile.TemporaryDirectory() as tmp:
        biased = os.path.join(tmp, "pipeline_bsel.npz")
        np.savez(biased, **p)
        base.main(biased, mock_path, out)
    d = dict(np.load(out))
    d.update(B_sel_C1=b, lambda_rep=LAM_REP, z_rep=z_rep,
             source=np.array(str(d["source"]) + "; data_Shear x B_sel C1 (A, alpha, beta, gamma) = "
                             "(0.12, 4.11, -0.18, 1.82), mock cell 37"))
    np.savez(out, **d)
    print(f"  B_sel range on the grid: {b.min():.4f}-{b.max():.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
