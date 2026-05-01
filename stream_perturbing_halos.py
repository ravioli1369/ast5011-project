"""
Stream-perturbing halo analysis for IllustrisTNG.

Pipeline
--------
1. find_mw_halos()         -> list of MW-mass FoF group indices (TNG50-1, z=0)
2. fetch_subhalos(halo)    -> arrays of (id, mass, position, galactocentric R)
                              for all subhalos of that host in the perturber
                              mass band
3. perturbations_per_degree(subhalos, D_helio, ...)
                           -> predicted N_perturbations / degree along a stream

All API responses are cached to disk under illustris_data/api_cache/, and
per-halo subhalo catalogs are saved as illustris_data/halos/halo_<idx>.npz.
First run is slow (network-bound); subsequent runs are instant.

Stream-perturbation framework (cf. Erkal & Belokurov 2015, Bonaca+ 2019):
    dN/dL = (1/V_shell) * sum_i 2 * b_max(M_i) * v_rel * T_stream
    dN/d(deg) = (dN/dL) * D_helio * pi/180
where b_max(M) = alpha * r_s(M), r_s ~ 1.05 * (M / 1e8 Msun)^0.5 kpc.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

API_KEY = "c2cbf44234668a2497d0d7c259b13084"
SIM = "TNG50-1"
SNAP = 99  # z = 0
BASE_URL = f"http://www.tng-project.org/api/{SIM}/snapshots/{SNAP}/"

# TNG / Planck15 cosmology
H_HUBBLE = 0.6774
A_SCALE = 1.0  # snapshot 99 is z = 0

# Default selection cuts
# MW-analog hosts: select on the central subhalo's total bound mass, which
# corresponds to roughly the same number for the FoF group's DM mass. Range
# straddles canonical MW estimates (1.0-1.5e12) and slightly heavier analogs.
MW_HALO_MASS_MSUN = (1.0e12, 2.0e12)
PERTURBER_MASS_MSUN = (1.0e6, 1.0e9)
PERTURBER_R_RANGE_KPC = (10.0, 100.0)

# Stream-perturbation defaults
DEFAULT_STREAM_AGE_GYR = 3
DEFAULT_V_REL_KMS = 200.0
DEFAULT_BMAX_OVER_RS = 5.0
KPC_PER_KMS_GYR = 1.02271  # 1 km/s * 1 Gyr in kpc

# On-disk layout
DATA_DIR = Path(__file__).parent / "illustris_data"
API_CACHE_DIR = DATA_DIR / "api_cache"
HALO_CACHE_DIR = DATA_DIR / "halos"
DOWNLOADS_DIR = DATA_DIR / "downloads"
MW_LIST_FILE = DATA_DIR / "mw_halos.json"


# -----------------------------------------------------------------------------
# Disk-cached API client
# -----------------------------------------------------------------------------


def _cache_key(url: str, params: dict | None) -> str:
    blob = url
    if params:
        blob += "?" + json.dumps(params, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


def _read_json_cache(url: str, params: dict | None):
    f = API_CACHE_DIR / f"{_cache_key(url, params)}.json"
    if f.exists():
        with open(f) as fp:
            return json.load(fp)
    return None


def _write_json_cache(url: str, params: dict | None, data) -> None:
    API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(API_CACHE_DIR / f"{_cache_key(url, params)}.json", "w") as fp:
        json.dump(data, fp)


def _http_get(url: str, params: dict | None = None, retries: int = 8):
    headers = {"api-key": API_KEY}
    for n in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            return r
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
            if n == retries - 1:
                raise
            time.sleep(min(2**n, 30))


def get(url: str, params: dict | None = None):
    """
    GET an Illustris API URL with disk caching.

    JSON responses are cached forever (keyed by URL+params hash). Binary
    downloads are saved to illustris_data/downloads/ and the path is returned.
    """
    cached = _read_json_cache(url, params)
    if cached is not None:
        return cached

    r = _http_get(url, params)
    ctype = r.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        data = r.json()
        _write_json_cache(url, params, data)
        return data

    if "content-disposition" in r.headers:
        fname = r.headers["content-disposition"].split("filename=")[1].strip('"; ')
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        out = DOWNLOADS_DIR / fname
        with open(out, "wb") as fp:
            fp.write(r.content)
        return str(out)

    return r.content


# -----------------------------------------------------------------------------
# Unit conversions (TNG code units -> physical)
# -----------------------------------------------------------------------------
# TNG masses are in 10^10 Msun/h; lengths are comoving kpc/h.


def code_mass_to_msun(m_code):
    return np.asarray(m_code) * 1e10 / H_HUBBLE


def msun_to_code_mass(m_msun):
    return np.asarray(m_msun) * H_HUBBLE / 1e10


def code_length_to_kpc(x_code, a: float = A_SCALE):
    return np.asarray(x_code) * a / H_HUBBLE


# -----------------------------------------------------------------------------
# MW-like host halo selection
# -----------------------------------------------------------------------------


def find_mw_halos(
    mass_range_msun=MW_HALO_MASS_MSUN,
    force_refresh: bool = False,
    n_workers: int = 8,
    verbose: bool = True,
) -> list[int]:
    """
    Find MW-analog FoF groups by selecting central (primary) subhalos whose
    total bound mass lies in mass_range_msun, then reading their FoF group
    index (grnr).

    This uses the /subhalos/?primary_flag=1 list endpoint with a server-side
    mass filter, which is far faster than scanning halo indices one-by-one.

    Returns a sorted list of FoF group indices, cached to mw_halos.json.
    """
    if MW_LIST_FILE.exists() and not force_refresh:
        with open(MW_LIST_FILE) as f:
            return json.load(f)

    m_lo_code = float(msun_to_code_mass(mass_range_msun[0]))
    m_hi_code = float(msun_to_code_mass(mass_range_msun[1]))
    page = get(
        f"{BASE_URL}subhalos/",
        {
            "primary_flag": 1,
            "mass__gt": m_lo_code,
            "mass__lt": m_hi_code,
            "limit": 5000,
        },
    )
    centrals = page["results"]
    if verbose:
        print(
            f"found {len(centrals)} primary subhalos with M in "
            f"[{mass_range_msun[0]:.1e}, {mass_range_msun[1]:.1e}] Msun"
        )

    # Each central's detail gives us the parent FoF group index (grnr).
    urls = [c["url"] for c in centrals]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        details = list(ex.map(get, urls))

    halo_indices = sorted({d["grnr"] for d in details})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MW_LIST_FILE, "w") as f:
        json.dump(halo_indices, f)
    if verbose:
        print(f"wrote {len(halo_indices)} MW-analog halo indices to {MW_LIST_FILE}")
    return halo_indices


# -----------------------------------------------------------------------------
# Subhalo fetching for a given host
# -----------------------------------------------------------------------------


def _list_subhalos_in_group(halo_idx: int, mass_range_msun=PERTURBER_MASS_MSUN):
    """
    Return a list of {id, url} dicts for all Subfind subhalos in FoF group
    `halo_idx` whose mass is within mass_range_msun.

    Uses the /subhalos/ list endpoint with grnr + mass filters, paginating until
    all results are retrieved. This endpoint returns only id+url per entry, so
    full position/mass requires a follow-up per-subhalo fetch.
    """
    mass_lo_code = msun_to_code_mass(mass_range_msun[0])
    mass_hi_code = msun_to_code_mass(mass_range_msun[1])
    url = f"{BASE_URL}subhalos/"
    out = []
    offset = 0
    page_size = 5000
    while True:
        params = {
            "grnr": halo_idx,
            "mass__gt": float(mass_lo_code),
            "mass__lt": float(mass_hi_code),
            "limit": page_size,
            "offset": offset,
        }
        page = get(url, params)
        results = page.get("results", [])
        out.extend(results)
        if len(results) < page_size:
            break
        offset += page_size
    return out


def fetch_subhalos(
    halo_idx: int,
    mass_range_msun=PERTURBER_MASS_MSUN,
    n_workers: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Fetch all subhalos in mass_range_msun for FoF group halo_idx. Caches the
    result to illustris_data/halos/halo_<idx>.npz so subsequent calls are
    instant.

    Returns a dict with arrays:
        id          : (N,) int64
        mass_msun   : (N,) float
        pos_kpc     : (N, 3) float, simulation-box coordinates [physical kpc]
        R_kpc       : (N,) float, distance to host center [physical kpc]
        halo_pos_kpc: (3,) float, host center position
    """
    cache = HALO_CACHE_DIR / f"halo_{halo_idx}.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    halo = get(f"{BASE_URL}halos/{halo_idx}/")
    info = get(halo["meta"]["info"])
    halo_pos = code_length_to_kpc(np.array(info["GroupPos"], dtype=float))

    listing = _list_subhalos_in_group(halo_idx, mass_range_msun)
    urls = [s["url"] for s in listing]

    if verbose:
        print(f"halo {halo_idx}: fetching {len(urls)} subhalos...")

    # Concurrent per-subhalo fetches. `get` is cached, so re-fetches are free.
    results = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_to_i = {ex.submit(get, u): i for i, u in enumerate(urls)}
        done = 0
        for fut in as_completed(future_to_i):
            i = future_to_i[fut]
            results[i] = fut.result()
            done += 1
            if verbose and done % 250 == 0:
                print(f"  halo {halo_idx}: {done}/{len(urls)} fetched")

    ids = np.array([d["id"] for d in results], dtype=np.int64)
    mass_code = np.array([d["mass"] for d in results], dtype=float)
    pos_code = np.array(
        [[d["pos_x"], d["pos_y"], d["pos_z"]] for d in results], dtype=float
    )
    mass_msun = code_mass_to_msun(mass_code)
    pos_kpc = code_length_to_kpc(pos_code)
    R_kpc = np.linalg.norm(pos_kpc - halo_pos, axis=1)

    HALO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        id=ids,
        mass_msun=mass_msun,
        pos_kpc=pos_kpc,
        R_kpc=R_kpc,
        halo_pos_kpc=halo_pos,
    )
    return {
        "id": ids,
        "mass_msun": mass_msun,
        "pos_kpc": pos_kpc,
        "R_kpc": R_kpc,
        "halo_pos_kpc": halo_pos,
    }


# -----------------------------------------------------------------------------
# Stream-perturbation rate
# -----------------------------------------------------------------------------


def nfw_scale_radius_kpc(mass_msun):
    """
    Approximate NFW scale radius for a low-mass subhalo (Erkal+ 2016 eq. 15,
    assuming concentration c ~ 15). Used to set the encounter cross-section
    b_max = alpha * r_s.
    """
    return 1.05 * (np.asarray(mass_msun) / 1e8) ** 0.5  # kpc


def perturbations_per_kpc(
    subhalos: dict,
    stream_age_gyr: float = DEFAULT_STREAM_AGE_GYR,
    v_rel_kms: float = DEFAULT_V_REL_KMS,
    bmax_over_rs: float = DEFAULT_BMAX_OVER_RS,
    r_range_kpc: tuple = PERTURBER_R_RANGE_KPC,
    mass_range_msun: tuple | None = None,
) -> dict:
    """
    Predicted number of stream-perturbing impacts per degree of stream on sky.

    Uses the encounter-rate framework (Erkal & Belokurov 2015):
        dN/dL = (1/V_shell) * sum_i 2 b_max(M_i) v_rel    [per kpc per Gyr]
    integrated over the stream age, then projected onto the sky at the
    stream's heliocentric distance.

    Parameters
    ----------
    subhalos : output of fetch_subhalos()
    stream_distance_kpc : heliocentric distance to the stream
    stream_age_gyr : disrupting time over which encounters accumulate
    v_rel_kms : mean relative speed between subhalo and stream
    bmax_over_rs : alpha factor (b_max = alpha * r_s)
    r_range_kpc : galactocentric (lo, hi) shell where we count perturbers
    mass_range_msun : optional further mass cut; if None, all subhalos kept

    Returns
    -------
    dict with:
        per_kpc        : N_perturbations / kpc of stream
        per_deg        : N_perturbations / degree of stream
        n_perturbers   : count of subhalos passing all cuts
        V_shell_kpc3   : volume of the shell used for the density estimate
        rate_gyr       : impact rate per kpc per Gyr (before * T_stream)
    """
    R = subhalos["R_kpc"]
    M = subhalos["mass_msun"]
    sel = (R >= r_range_kpc[0]) & (R <= r_range_kpc[1])
    if mass_range_msun is not None:
        sel &= (M >= mass_range_msun[0]) & (M <= mass_range_msun[1])

    M_sel = M[sel]
    V_shell = (4.0 / 3.0) * np.pi * (r_range_kpc[1] ** 3 - r_range_kpc[0] ** 3)

    if M_sel.size == 0:
        return {
            "per_kpc": 0.0,
            "per_deg": 0.0,
            "n_perturbers": 0,
            "V_shell_kpc3": V_shell,
            "rate_gyr": 0.0,
        }

    r_s = nfw_scale_radius_kpc(M_sel)
    b_max = bmax_over_rs * r_s

    # Encounter rate per kpc of stream per Gyr (sum over perturbers / V_shell).
    # 2*b_max in kpc; v_rel*KPC_PER_KMS_GYR converts km/s to kpc/Gyr.
    rate_gyr = (2.0 * b_max.sum() * v_rel_kms * KPC_PER_KMS_GYR) / V_shell

    n_per_kpc = rate_gyr * stream_age_gyr

    return {
        "per_kpc": float(n_per_kpc),
        "n_perturbers": int(M_sel.size),
        "V_shell_kpc3": float(V_shell),
        "rate_gyr": float(rate_gyr),
    }


# -----------------------------------------------------------------------------
# Convenience: run the full pipeline over all MW hosts
# -----------------------------------------------------------------------------


def population_perturbation_rates(
    halo_indices: list[int] | None = None,
    **kwargs,
) -> dict:
    """
    Compute perturbations/degree for every MW-like host. Returns parallel
    arrays useful for histograms / population statistics.

    kwargs are forwarded to perturbations_per_degree().
    """
    if halo_indices is None:
        halo_indices = find_mw_halos()

    per_kpc = []
    n_pert = []
    for idx in halo_indices:
        sh = fetch_subhalos(idx)
        out = perturbations_per_kpc(sh, **kwargs)
        per_kpc.append(out["per_kpc"])
        n_pert.append(out["n_perturbers"])

    return {
        "halo_indices": np.array(halo_indices),
        "per_kpc": np.array(per_kpc),
        "n_perturbers": np.array(n_pert),
    }
