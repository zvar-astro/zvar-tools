import json
import os
from typing import List

from penquins import Kowalski


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
            "query_type": "near",
            "query": {
                "max_distance": radius,
                "distance_units": "arcsec",
                "radec": {str(id): [float(ra), float(dec)] for id, ra, dec in batch},
                "catalogs": {
                    "Gaia_EDR3": {
                        "filter": {},
                        "projection": {
                            "pmra": 1,
                            "pmra_error": 1,
                            "pmdec": 1,
                            "pmdec_error": 1,
                            "parallax": 1,
                            "parallax_error": 1,
                            "phot_g_mean_mag": 1,
                            "phot_bp_mean_mag": 1,
                            "phot_rp_mean_mag": 1,
                        },
                    }
                },
            },
            "kwargs": {"limit": 1},
        }
        queries.append(query)

    responses = k.query(queries=queries, use_batch_query=True, max_n_threads=30)
    responses = responses.get("default")

    results = {}
    for response in responses:
        if response.get("status") != "success":
            print(f'Query failed with error: {response.get("message")}')
            continue
        for id, result in response.get("data").get("Gaia_EDR3").items():
            results[id] = result

    return results
