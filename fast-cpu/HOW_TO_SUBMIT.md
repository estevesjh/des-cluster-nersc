# How to submit the mock-MCMC jobs (NERSC Perlmutter)

This is the **active Costanzi-2026 mock-MCMC pipeline** (`fast-cpu/`), driven by
`cosmosis-models/mock_mcmc_cp_camb.ini`. CPU-only (`--constraint=cpu`).

## 0. Prerequisites (read once)

This repo is **only the orchestration layer** — the physics lives in a sibling
C++/CUDA repo that must be built separately:

- `Y3_CLUSTER_CPP_DIR = /pscratch/sd/j/jesteves/y3_cluster_cpp` — must contain a
  `release-build/` with the compiled `.so` modules (NumCountsSel, Shear1hMisSel,
  b_sel_marg, shear_prj, …). Build it with its own `BUILDING.md`.
- The CosmoSIS env (`y3cl_je` conda env) and the CosmoPower emulator under
  `camb-emulator/camb-for-cp` must be present (paths are in the `.ini`).

All paths are hard-coded in `fast-cpu/setup_env.sh`. If you run on a different
account / scratch, edit that file's exports first (`Y3_CLUSTER_CPP_DIR`,
`DES_CLUSTER_NERSC_DIR`, `TOP_DIR`, and `--account=des` in the `.sh` headers).

## 1. Environment

Every `sbatch` script **sources `setup_env.sh` itself** (`.bashrc` is not read in
batch jobs), so you do **not** need to activate anything before submitting.

To run interactively (e.g. the `test` sampler), source it first — never execute:

```bash
cd /pscratch/sd/j/jesteves/github/des-cluster-nersc/fast-cpu
source ./setup_env.sh
```

## 2. The scripts (submit in this order)

All scripts `cd` to absolute paths, so they can be `sbatch`'d from anywhere.

| Script | QOS | Wall | What it does |
|---|---|---|---|
| `generate_mock_dv.sh` | debug | ~5 min | Build the fiducial mock data vector → `data/mock/mock_dv_widePlanck_jkcov.npz`. **Run this first** if the `.npz` is missing. |
| `mock_test.sh` | debug | ≤30 min | Smoke / prior-coverage check: 128 prior draws, no MCMC. Confirms the pipeline + likelihood evaluate cleanly. |
| `mock_polychord_debug.sh` | debug | 30 min | Tiny polychord run (`live_points=100`) to confirm MPI + sampling end-to-end before a full run. **Always run before production.** |
| `mock_polychord.sh` | shared | 9 h | **Production** polychord nested-sampling chain (`live_points=500`). |
| `mock_polychord_resume.sh` | shared | 3 h | Continue a production run that timed out (see §4). |
| `mock_emcee.sh` | shared | 9 h | Production emcee chain (alternative to polychord). |

### Typical flow

```bash
cd /pscratch/sd/j/jesteves/github/des-cluster-nersc

# 1. (only if the mock DV does not yet exist)
sbatch fast-cpu/generate_mock_dv.sh

# 2. shake-down on debug QOS (REQUIRED before any full run)
sbatch fast-cpu/mock_polychord_debug.sh

# 3. production (after the debug job COMPLETED with real output)
sbatch fast-cpu/mock_polychord.sh
```

## 3. What the production polychord job runs

`mock_polychord.sh` essentially does:

```bash
source ./setup_env.sh
cd ${DES_CLUSTER_NERSC_DIR}
srun -n 64 cosmosis --mpi cosmosis-models/mock_mcmc_cp_camb.ini \
     -p runtime.sampler=polychord polychord.resume=F
```

- **64 MPI ranks** on 1 shared-QOS node (64 physical Milan cores; `-c 2`).
  polychord uses real MPI — this needs `srun`, not `--smp`.
- Override **any** `.ini` value on the command line with `-p section.key=value`.
- Outputs (chain + evidence) go to
  `/pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/` (`chain.txt`,
  `polychord/`), **not** into the repo. Job logs (`*.log`/`*.error`) land in
  `fast-cpu/` and are git-ignored.

### Likelihood space (log vs linear)

The likelihood defaults to **linear** (`log_space = F` in the ini). An A/B test
(jobs 55014977 vs 55040404) showed log-space gives statistically identical
posteriors and the same convergence cost, so linear is the default. To reproduce
the log-space run, add `-p likelihoods.log_space=T` and point it at a **separate**
output dir so it doesn't clobber the linear chain (see
`mock_polychord_logspace.sh` for the exact overrides).

## 4. Resuming a timed-out polychord run

polychord checkpoints continuously to `<base_dir>/<root>.resume`. If the 9 h
production job hits the wall before converging, continue it (do **not** restart):

```bash
# only AFTER the parent job has ended (two jobs writing the same base_dir
# corrupts the checkpoint). Use afterany, not afterok: a wall-timeout is a
# non-zero exit and would strand an afterok child forever.
sbatch --dependency=afterany:<PARENT_JOBID> fast-cpu/mock_polychord_resume.sh
```

`mock_polychord_resume.sh` sets both `runtime.resume=T` (cosmosis appends to
`chain.txt`) and `polychord.resume=T` (polychord reads its `.resume`), and
`likelihoods.log_space=F` to match the linear parent. **If your parent run used
log-space, edit the resume script to `log_space=T`** — the checkpoint encodes
likelihood values, so the space must match or the continuation is meaningless.

## 5. Monitoring & sanity checks

```bash
squeue -u $USER                                  # queue state
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode   # final disposition
# polychord live progress (evidence + dead points):
cat /pscratch/sd/j/jesteves/cluster_lib/chains/mock_mcmc/polychord/*.stats
```

- A run that **COMPLETED** with `nlive: 0` in the `.stats` converged on its own
  (hit `tolerance=0.1`). A **TIMEOUT** means it hit the wall — resume it (§4).
- `Pipeline failed on these parameters` lines in the `.error` log are normal
  theory-corner rejections (emulator / GSL box edges), not failures.
- Analyze the converged chain with
  `chains/Plot_Polychord_widePlanck_Chains.ipynb` (closure table + corner plots).
