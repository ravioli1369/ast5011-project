import os
import sys
import threading
import time

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

from streams import StreamConfig, Streams

MAX_RETRIES = 1  # retries per strip before skipping
RETRY_DELAY_SEC = 10  # wait between retries
STRIP_DIR = "strips"
N_STRIPS = 10


def spinner(msg, stop_event):
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    t0 = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        sys.stdout.write(f"\r  {chars[i % len(chars)]}  {msg} [{mins:02d}:{secs:02d}]")
        sys.stdout.flush()
        stop_event.wait(0.1)
        i += 1
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    sys.stdout.write(f"\r  ✓  {msg} — done [{mins:02d}:{secs:02d}]\n")
    sys.stdout.flush()


def query_strip(
    edge_low: float,
    edge_high: float,
    stream_cfg: StreamConfig,
    strip_num: int,
    total_strips: int,
    edge_type: str,
):
    """
    Query a single RA or DEC strip. If a cached CSV exists, load that
    instead. On failure, retry up to MAX_RETRIES times.
    """
    if edge_type == "RA":
        ra_lo, ra_hi = edge_low, edge_high
        dec_lo, dec_hi = stream_cfg.dec_range
    else:
        ra_lo, ra_hi = stream_cfg.ra_range
        dec_lo, dec_hi = edge_low, edge_high

    cache = os.path.join(STRIP_DIR, f"{stream_cfg.name}_strip_{strip_num:02d}.csv")

    # ── Check cache ──
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        print(
            f"Strip {strip_num}/{total_strips}: "
            f"{edge_type} {edge_low:.1f}°–{edge_high:.1f}° "
            f"— loaded from cache ({len(df):,} rows)"
        )
        return df

    # ── Query Gaia ──
    query = f"""
    SELECT
        source_id, ra, dec,
        parallax, parallax_error,
        pmra, pmra_error, pmdec, pmdec_error,
        phot_g_mean_mag   AS g,
        phot_bp_mean_mag  AS bp,
        phot_rp_mean_mag  AS rp,
        bp_rp,
        radial_velocity, radial_velocity_error,
        ruwe
    FROM gaiadr3.gaia_source
    WHERE
        ra  BETWEEN {ra_lo} AND {ra_hi}
        AND dec BETWEEN {dec_lo} AND {dec_hi}
        AND phot_g_mean_mag BETWEEN {stream_cfg.g_mag_range[0]} AND {stream_cfg.g_mag_range[1]}
        AND ruwe < {stream_cfg.ruwe_max}
        AND parallax < {stream_cfg.parallax_max}
        AND phot_bp_mean_mag IS NOT NULL
        AND phot_rp_mean_mag IS NOT NULL
    """

    label = f"Strip {strip_num}/{total_strips}: {edge_type} {edge_low:.1f}°–{edge_high:.1f}°"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stop = threading.Event()
            spin = threading.Thread(
                target=spinner,
                args=(f"{label} (attempt {attempt}/{MAX_RETRIES})", stop),
                daemon=True,
            )
            spin.start()

            job = Gaia.launch_job_async(query)
            table = job.get_results()

            stop.set()
            spin.join()

            df = table.to_pandas()
            print(f"     ↳ {len(df):,} sources")

            # ── Save to cache ──
            df.to_csv(cache, index=False)
            size_mb = os.path.getsize(cache) / 1e6
            print(f"     ↳ Cached → {cache} ({size_mb:.1f} MB)")

            return df

        except Exception as e:
            stop.set()
            spin.join()
            print(f"\n  ⚠  {label} failed: {e}")
            if attempt < MAX_RETRIES:
                print(
                    f"     Retrying in {RETRY_DELAY_SEC}s ({attempt}/{MAX_RETRIES})..."
                )
                time.sleep(RETRY_DELAY_SEC)
            else:
                print(f"     X Giving up on this strip after {MAX_RETRIES} attempts.")
                return None


# ── QUERY ALL STRIPS ────────────────────────────────────────
def query(stream_cfg: StreamConfig):
    """
    Query the full region in RA or DEC strips with per-strip caching.
    Returns a combined DataFrame of all retrieved sources.
    """
    OUTPUT_PATH = f"{stream_cfg.name}.csv"

    os.makedirs(STRIP_DIR, exist_ok=True)

    if stream_cfg.ra_range[1] - stream_cfg.ra_range[0] > 2 * (
        stream_cfg.dec_range[1] - stream_cfg.dec_range[0]
    ):
        # RA strips
        edges = np.linspace(
            stream_cfg.ra_range[0], stream_cfg.ra_range[1], N_STRIPS + 1
        )
        edge_type = "RA"
    else:
        # DEC strips
        edges = np.linspace(
            stream_cfg.dec_range[0], stream_cfg.dec_range[1], N_STRIPS + 1
        )
        edge_type = "DEC"
    strip_width = edges[1] - edges[0]

    # Check which strips are already cached
    cached = sum(
        1
        for i in range(1, N_STRIPS + 1)
        if os.path.exists(
            os.path.join(STRIP_DIR, f"{stream_cfg.name}_strip_{i:02d}.csv")
        )
    )

    print(
        f"\n📡 Querying Gaia DR3 ({stream_cfg.name} region) in {N_STRIPS} {edge_type} strips of width {strip_width:.1f}°s..."
    )
    print(f"   RA  {stream_cfg.ra_range[0]}°-{stream_cfg.ra_range[1]}°")
    print(f"   Dec {stream_cfg.dec_range[0]}°-{stream_cfg.dec_range[1]}°")
    print(f"   G {stream_cfg.g_mag_range[0]}-{stream_cfg.g_mag_range[1]}")
    if cached > 0:
        print(f"   {cached}/{N_STRIPS} strips already cached - will skip those")
    print()

    frames = []
    running_total = 0
    skipped = 0

    for i in range(N_STRIPS):
        df_chunk = query_strip(
            edges[i], edges[i + 1], stream_cfg, i + 1, N_STRIPS, edge_type
        )

        if df_chunk is not None:
            frames.append(df_chunk)
            running_total += len(df_chunk)
            print(f"     Running total: {running_total:,}\n")
        else:
            skipped += 1
            print()

    if skipped > 0:
        print(f"\n  ⚠  {skipped} strip(s) failed. Re-run to retry them.\n")

    # Concatenate everything we have
    if not frames:
        print("  ✗ No data retrieved.")
        return pd.DataFrame()

    print("  Concatenating strips...", end="", flush=True)
    df = pd.concat(frames, ignore_index=True)
    print(f" {len(df):,} total sources")
    print("  Transforming coordinates...", end="", flush=True)
    df = transform(df, stream_cfg)
    # Save combined file
    print(f"  Saving to {OUTPUT_PATH}...", end="", flush=True)
    df.to_csv(OUTPUT_PATH, index=False)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f" done ({size_mb:.1f} MB)")


def transform(df: pd.DataFrame, stream_cfg: StreamConfig) -> pd.DataFrame:
    """
    Add columns for phi1, phi2, pm_phi1, pm_phi2 by transforming from RA/Dec to
    the stream-aligned coordinate system.
    """
    coords = SkyCoord(
        ra=df["ra"].values * u.deg,
        dec=df["dec"].values * u.deg,
        pm_ra_cosdec=df["pmra"].values * u.mas / u.yr,
        pm_dec=df["pmdec"].values * u.mas / u.yr,
    )

    ra_rad = np.deg2rad(coords.ra.deg)
    dec_rad = np.deg2rad(coords.dec.deg)

    # Convert to Cartesian on unit sphere
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    xyz = np.vstack([x, y, z])

    # Rotate
    xyz_rot = stream_cfg.R @ xyz

    # Back to spherical
    phi1 = np.rad2deg(np.arctan2(xyz_rot[1], xyz_rot[0]))
    phi2 = np.rad2deg(np.arcsin(xyz_rot[2]))

    # Transform proper motions (rotation of the tangent plane)
    # This requires computing the Jacobian of the transformation
    cos_phi2 = np.cos(np.deg2rad(phi2))
    cos_dec = np.cos(dec_rad)

    # Jacobian elements for PM transformation
    # d(phi1)/d(ra), d(phi1)/d(dec), d(phi2)/d(ra), d(phi2)/d(dec)
    # Computed from the chain rule through the rotation
    pmra = coords.pm_ra_cosdec.value
    pmdec = coords.pm_dec.value

    # Numerical Jacobian
    delta = 1e-6  # rad
    pm_phi1 = np.zeros(len(coords))
    pm_phi2 = np.zeros(len(coords))

    for i in range(len(coords)):
        # Perturb RA
        ra_p = ra_rad[i] + delta
        x_p = np.cos(dec_rad[i]) * np.cos(ra_p)
        y_p = np.cos(dec_rad[i]) * np.sin(ra_p)
        z_p = np.sin(dec_rad[i])
        rot_p = stream_cfg.R @ np.array([x_p, y_p, z_p])
        dphi1_dra = (np.arctan2(rot_p[1], rot_p[0]) - np.deg2rad(phi1[i])) / delta
        dphi2_dra = (np.arcsin(rot_p[2]) - np.deg2rad(phi2[i])) / delta

        # Perturb Dec
        dec_p = dec_rad[i] + delta
        x_p = np.cos(dec_p) * np.cos(ra_rad[i])
        y_p = np.cos(dec_p) * np.sin(ra_rad[i])
        z_p = np.sin(dec_p)
        rot_p = stream_cfg.R @ np.array([x_p, y_p, z_p])
        dphi1_ddec = (np.arctan2(rot_p[1], rot_p[0]) - np.deg2rad(phi1[i])) / delta
        dphi2_ddec = (np.arcsin(rot_p[2]) - np.deg2rad(phi2[i])) / delta

        # Transform: pm_phi1*cos(phi2) and pm_phi2
        pm_phi1[i] = (
            dphi1_dra * pmra[i] / cos_dec[i] + dphi1_ddec * pmdec[i]
        ) * cos_phi2[i]
        pm_phi2[i] = dphi2_dra * pmra[i] / cos_dec[i] + dphi2_ddec * pmdec[i]
        if i % 1_000_000 == 0:
            print(f"Processed {i} / {len(coords)} stars")

    df["phi1"] = phi1
    df["phi2"] = phi2
    df["pm_phi1"] = pm_phi1
    df["pm_phi2"] = pm_phi2
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query Gaia for a stellar stream.")
    parser.add_argument(
        "stream",
        type=str.upper,
        choices=[k for k, v in vars(Streams).items() if isinstance(v, StreamConfig)],
        help="Which stream to query",
    )
    args = parser.parse_args()
    stream_cfg = getattr(Streams, args.stream)
    query(stream_cfg)
