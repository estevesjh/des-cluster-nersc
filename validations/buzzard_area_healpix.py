#!/usr/bin/env python
"""NSIDE-robust Buzzard footprint area from halo RA/DEC (issue #3, item 2).

Method (per owner spec):
  1. healpix grid over ALL halo positions. Coordinates: healpy expects
     (lon, lat) in DEGREES with ``lonlat=True`` -- RA maps to lon
     directly, DEC maps to lat directly (no colatitude conversion, no
     galactic rotation; area is rotation-invariant, so the equatorial
     frame is fine). Without ``lonlat=True`` healpy wants radians and
     colatitude theta = pi/2 - dec -- the classic mistake; we assert
     the input ranges before converting.
  2. Occupancy-corrected area per NSIDE: raw occupied-pixel counting is
     biased at BOTH ends (border pixels inflate it at low NSIDE, empty
     sampling holes deflate it at high NSIDE -- Tan Xing's scan:
     5700 -> 5350 -> 5197 -> 5025.6 -> 3742 deg^2 for NSIDE 32..512).
     For Poisson sampling with true mean n-bar halos per in-footprint
     pixel, the observed mean over OCCUPIED pixels is
         m = n-bar / (1 - exp(-n-bar)),
     so invert m -> n-bar and correct
         A = N_occ * A_pix / (1 - exp(-n-bar)).
     The corrected estimate should plateau across NSIDE; the plateau is
     the answer.
  3. Rough selection with "max" first (owner spec): a pixel counts as
     footprint if AT LEAST ONE cluster of the full sample lands in it --
     that union mask at the reference NSIDE is the footprint area.
     Then, per ~120 Mpc comoving slice, report the slice's occupied
     fraction of that union mask (occupancy-corrected per slice). This
     separates real z-dependence of the footprint from sampling noise,
     and gives the volume-weighted effective Omega_buzz for the
     per-halo NC weighting (w = Omega_Y1(z_i) / Omega_buzz).

Needs the halo catalog (NERSC): runs on a login node in seconds.

    python validations/buzzard_area_healpix.py \\
        --catalog /path/to/halo_run.fits [--ra-col RA --dec-col DEC \\
        --z-col redshift] [--nside-ref 128] [--slice-mpc 120]

Writes validations/cache/buzzard_area_healpix.csv and prints the table.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

H0, OM0 = 70.0, 0.286          # Buzzard (DeRose+2019); matches the notebook


def occupancy_corrected_area(pix, nside):
    """Return (raw_area_deg2, corrected_area_deg2, nbar, hole_frac)."""
    import healpy as hp
    counts = np.bincount(pix)
    counts = counts[counts > 0]
    n_occ = counts.size
    pix_area = hp.nside2pixarea(nside, degrees=True)
    m = counts.mean()                     # mean over OCCUPIED pixels
    # invert m = nbar / (1 - exp(-nbar)) by fixed-point iteration
    nbar = m
    for _ in range(200):
        nbar_new = m * (1.0 - np.exp(-nbar))
        if abs(nbar_new - nbar) < 1e-12:
            break
        nbar = nbar_new
    hole = float(np.exp(-nbar))
    raw = n_occ * pix_area
    return raw, raw / (1.0 - hole), float(nbar), hole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True,
                    help="halo catalog FITS with RA/DEC (and z for slices)")
    ap.add_argument("--ra-col", default="RA")
    ap.add_argument("--dec-col", default="DEC")
    ap.add_argument("--z-col", default="redshift")
    ap.add_argument("--nside-ref", type=int, default=64,
                    help="reference NSIDE for the union mask / slices "
                         "(owner spec: keep pixels coarser than ~30 arcmin, "
                         "i.e. NSIDE <= 64-128)")
    ap.add_argument("--slice-mpc", type=float, default=120.0,
                    help="comoving slice thickness [Mpc]")
    args = ap.parse_args()

    import fitsio
    import healpy as hp
    from astropy.cosmology import FlatLambdaCDM

    data = fitsio.read(args.catalog,
                       columns=[args.ra_col, args.dec_col, args.z_col])
    ra = np.asarray(data[args.ra_col], dtype=float)
    dec = np.asarray(data[args.dec_col], dtype=float)
    z = np.asarray(data[args.z_col], dtype=float)

    # --- coordinate sanity BEFORE touching healpy --------------------
    ra = np.mod(ra, 360.0)                       # wrap to [0, 360)
    assert np.all((dec >= -90.0) & (dec <= 90.0)), \
        "dec outside [-90, 90] -- wrong column or units?"
    good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
    ra, dec, z = ra[good], dec[good], z[good]
    print(f"halos: {ra.size}  RA [{ra.min():.2f}, {ra.max():.2f}]  "
          f"DEC [{dec.min():.2f}, {dec.max():.2f}]  "
          f"z [{z.min():.3f}, {z.max():.3f}]")

    # --- 1+2: occupancy-corrected area vs NSIDE ----------------------
    # Owner spec: read the plateau from pixels COARSER than ~30 arcmin
    # (NSIDE <= 64; NSIDE=128 at ~27.5' is printed for reference only) --
    # finer pixels out-resolve the halo sampling and the raw count
    # collapses into holes.
    rows = []
    print(f"\n{'NSIDE':>6} {'resol':>8} {'raw deg2':>10} {'corrected':>10} "
          f"{'nbar/pix':>9} {'hole%':>7}")
    for nside in (16, 32, 64, 128):
        # lonlat=True: (lon=RA, lat=DEC) in degrees. Equivalent to
        # ang2pix(nside, np.radians(90-dec), np.radians(ra)).
        pix = hp.ang2pix(nside, ra, dec, lonlat=True)
        raw, corr, nbar, hole = occupancy_corrected_area(pix, nside)
        resol = np.degrees(hp.nside2resol(nside)) * 60.0
        flag = "" if resol >= 30.0 else "  (< 30', reference only)"
        rows.append(dict(kind="full", nside=nside, raw_deg2=raw,
                         corrected_deg2=corr, nbar=nbar, hole_frac=hole))
        print(f"{nside:>6} {resol:>7.1f}' {raw:>10.1f} {corr:>10.1f} "
              f"{nbar:>9.2f} {100*hole:>6.2f}%{flag}")
    print("-> adopt the corrected plateau over the >=30' rows; very coarse "
          "rows carry border bias the correction does not remove.")

    # --- 3: fractional coverage in comoving slices -------------------
    cosmo = FlatLambdaCDM(H0=H0, Om0=OM0)
    chi = cosmo.comoving_distance(z).value            # Mpc
    edges = np.arange(chi.min(), chi.max() + args.slice_mpc,
                      args.slice_mpc)
    nside = args.nside_ref
    pix_all = hp.ang2pix(nside, ra, dec, lonlat=True)
    union = np.unique(pix_all)
    pix_area = hp.nside2pixarea(nside, degrees=True)
    union_area = union.size * pix_area
    print(f"\nunion mask at NSIDE={nside}: {union.size} px = "
          f"{union_area:.1f} deg2 (raw)")
    print(f"{'chi_lo':>8} {'chi_hi':>8} {'z_mid':>6} {'N':>8} "
          f"{'frac_raw':>9} {'frac_corr':>9} {'area_corr':>10}")
    zgrid = np.linspace(0.0, max(1.0, z.max() + 0.1), 2048)
    chigrid = cosmo.comoving_distance(zgrid).value
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (chi >= lo) & (chi < hi)
        if m.sum() < 100:
            continue
        raw, corr, nbar, hole = occupancy_corrected_area(pix_all[m], nside)
        z_mid = float(np.interp(0.5 * (lo + hi), chigrid, zgrid))
        rows.append(dict(kind="slice", nside=nside, chi_lo=lo, chi_hi=hi,
                         z_mid=z_mid, n_halos=int(m.sum()), raw_deg2=raw,
                         corrected_deg2=corr, nbar=nbar, hole_frac=hole))
        print(f"{lo:>8.0f} {hi:>8.0f} {z_mid:>6.3f} {m.sum():>8d} "
              f"{raw/union_area:>9.3f} {corr/union_area:>9.3f} "
              f"{corr:>10.1f}")
    print("-> flat frac_corr with z = footprint is z-independent and the "
          "full-sample corrected area is THE Omega_buzz; a trend means "
          "Omega_buzz(z) and the per-halo NC weight needs it.")

    out = os.path.join(os.path.dirname(__file__), "cache",
                       "buzzard_area_healpix.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
