import json
import os
from typing import List

import numpy as np
from penquins import Kowalski

from zvar_utils.spatial import great_circle_distance


def connect_to_kowalski(path_credentials: str, verbose: bool = True) -> Kowalski:
    if path_credentials is None:
        raise ValueError("Credentials file must be specified")
    if not os.path.exists(path_credentials):
        raise ValueError(f"Credentials file {path_credentials} does not exist")
    with open(path_credentials) as f:
        passwd = json.load(f)
    username = passwd["melman"]["username"]
    password = passwd["melman"]["password"]

    k = Kowalski(
        protocol="https",
        host="melman.caltech.edu",
        port=443,
        username=username,
        password=password,
        verbose=verbose,
        timeout=600,
    )
    return k


def validate_gaia_source(source):
    if not isinstance(source, dict):
        return False, "Source must be a dictionary"
    # parallax, parallax_error = source.get("parallax"), source.get("parallax_error")
    # if not all(isinstance(val, (int, float)) for val in [parallax, parallax_error]):
    #     return False, "Parallax and parallax error must be numbers"
    # if parallax < 0:
    #     return False, "Parallax must be positive"
    # if parallax / parallax_error < 5:
    #     return False, "Parallax SNR must be greater than 5"
    for band in ["g", "bp", "rp"]:
        mag = source.get(f"phot_{band}_mean_mag")
        flux = source.get(f"phot_{band}_mean_flux")
        flux_error = source.get(f"phot_{band}_mean_flux_error")
        if not all(isinstance(val, (int, float)) for val in [mag, flux, flux_error]):
            return False, f"{band} band magnitude, flux, and flux error must be numbers"
        if mag < 0:
            return False, f"{band} band magnitude must be positive"
        if np.abs(flux) / flux_error < 5:
            return False, f"{band} band SNR must be greater than 5"

    # chi_al = source.get("astrometric_chi2_al")
    # n_good_obs_al = source.get("astrometric_n_good_obs_al")

    # if not all(isinstance(val, (int, float)) for val in [chi_al, n_good_obs_al]):
    #     return False, "Astrometric chi2 and n_good_obs must be numbers"
    # if not (chi_al / (n_good_obs_al - 5) < 1.44 and n_good_obs_al > 10):
    #     return False, "Astrometric chi2 over n_good_obs must be less than 1.44 and n_good_obs must be greater than 10"

    # ruwe = source.get("ruwe")
    # if not isinstance(ruwe, (int, float)):
    #     return False, "RUWE must be a number"
    # if ruwe > 1.40:
    #     return False, "RUWE must be less than 1.4"

    # phot_bp_rp_excess_factor = source.get("phot_bp_rp_excess_factor")
    # if not isinstance(phot_bp_rp_excess_factor, (int, float)):
    #     return False, "BP-RP excess factor must be a number"

    # if not(phot_bp_rp_excess_factor >= 1.0 and phot_bp_rp_excess_factor < 1.30):
    #     return False, "BP-RP excess factor must be between 1.0 and 1.3"

    ra, dec = source.get("ra"), source.get("dec")
    if not all(isinstance(val, (int, float)) for val in [ra, dec]):
        return False, "RA and Dec must be numbers"

    return True, None


def query_gaia(
    k: Kowalski, ids: List[int], ras: List[int], decs: List[int], radius: float
) -> dict:
    if not isinstance(k, Kowalski):
        raise ValueError("Kowalski must be an instance of the Kowalski class")
    if not isinstance(ids, list):
        raise ValueError("IDs must be provided as a list")
    if not isinstance(ras, list):
        raise ValueError("RAs must be provided as a list")
    if not isinstance(decs, list):
        raise ValueError("Decs must be provided as a list")
    if not isinstance(radius, (int, float)):
        raise ValueError("Radius must be a number")

    # Create a list of tuples of (psid, ra, dec) for each source
    inputs = []
    for i in range(len(ids)):
        inputs.append((ids[i], ras[i], decs[i]))

    # Create batches of 1000 sources
    batch_size = 1000
    batches = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]

    queries = []
    for batch in batches:
        # Create a list of dictionaries for each source
        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "cone_search_radius": radius,
                    "cone_search_unit": "arcsec",
                    "radec": {
                        str(id): [float(ra), float(dec)] for id, ra, dec in batch
                    },
                },
                "catalogs": {
                    "Gaia_EDR3": {
                        "filter": {},
                        "projection": {
                            "_id": 1,
                            "pmra": 1,
                            "pmra_error": 1,
                            "pmdec": 1,
                            "pmdec_error": 1,
                            "parallax": 1,
                            "parallax_error": 1,
                            "phot_g_mean_mag": 1,
                            "phot_bp_mean_mag": 1,
                            "phot_rp_mean_mag": 1,
                            "phot_g_mean_flux": 1,
                            "phot_bp_mean_flux": 1,
                            "phot_rp_mean_flux": 1,
                            "phot_g_mean_flux_error": 1,
                            "phot_bp_mean_flux_error": 1,
                            "phot_rp_mean_flux_error": 1,
                            "astrometric_chi2_al": 1,
                            "astrometric_n_good_obs_al": 1,
                            "phot_bp_rp_excess_factor": 1,
                            "ruwe": 1,
                            "ra": 1,
                            "dec": 1,
                        },
                    }
                },
            },
            "kwargs": {"limit": 1},
        }
        queries.append(query)

    responses = k.query(queries=queries, use_batch_query=True, max_n_threads=10)
    responses = responses.get("default")

    results = {}
    for response in responses:
        if response.get("status") != "success":
            print(f'Query failed with error: {response.get("message")}')
            continue
        for id, result in response.get("data").get("Gaia_EDR3").items():
            # Filter out sources that don't pass the quality cuts
            result_filtered = []
            for source in result:
                valid, reason = validate_gaia_source(source)
                if valid:
                    result_filtered.append(source)
                else:
                    # print(f"Source {id} failed quality cuts: {reason}")
                    pass
            result = result_filtered

            if len(result) == 0:
                results[int(id)] = None
                continue
            if len(result) == 1:
                results[int(id)] = result[0]
                continue

            idx = ids.index(int(id))
            ra, dec = ras[idx], decs[idx]
            # keep the closest source
            min_dist = np.inf
            for source in result:
                dist = great_circle_distance(ra, dec, source["ra"], source["dec"])
                if dist < min_dist:
                    min_dist = dist
                    results[int(id)] = source

    return results
