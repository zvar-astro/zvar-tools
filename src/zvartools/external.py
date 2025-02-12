import json
import os
from typing import List
from typing import Tuple

import numpy as np
from penquins import Kowalski

from zvartools.spatial import great_circle_distance


def connect_to_kowalski(path_credentials: str, verbose: bool = True) -> Kowalski:
    """
    Connect to Kowalski using the credentials file

    Parameters
    ----------
    path_credentials : str
        Path to the JSON file with the Kowalski credentials
    verbose : bool
        Verbosity level

    Returns
    -------
    Kowalski
        Kowalski instance
    """
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


def validate_gaia_source(source: dict) -> Tuple[bool, str]:
    """
    Validate a Gaia source

    Parameters
    ----------
    source : dict

    Returns
    -------
    Tuple[bool, str]
        Validity of the source and the reason for invalidity
    """
    if not isinstance(source, dict):
        return False, "Source must be a dictionary"
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

    ra, dec = source.get("ra"), source.get("dec")
    if not all(isinstance(val, (int, float)) for val in [ra, dec]):
        return False, "RA and Dec must be numbers"

    return True, None


def validate_positional_source(source: dict) -> Tuple[bool, str]:
    """
    Validate a positional source

    Parameters
    ----------
    source : dict

    Returns
    -------
    Tuple[bool, str]
        Validity of the source and the reason for invalidity
    """
    if not isinstance(source, dict):
        return False, "Source must be a dictionary"
    ra, dec = source.get("ra"), source.get("dec")
    if not all(isinstance(val, (int, float)) for val in [ra, dec]):
        return False, "RA and Dec must be numbers"
    return True, None


def query_cone_search(
    k: Kowalski,
    ids: List[int],
    ras: List[int],
    decs: List[int],
    radius: float,
    id_field: str = "_id",
    batch_size: int = 1000,
    catalog: str = None,
    projection: dict = None,
    validate: callable = None,
) -> dict:
    """
    Query Kowalski with a cone search

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    ids : List[int]
        List of IDs
    ras : List[int]
        List of RAs
    decs : List[int]
        List of Decs
    radius : float
        Radius in arcseconds
    id_field : str
        ID field
    batch_size : int
        Batch size
    catalog : str
        Catalog name
    projection : dict
        Projection
    validate : callable
        Validation function

    Returns
    -------
    dict
        Query results
    """
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
    if not isinstance(batch_size, int):
        raise ValueError("Batch size must be an integer")
    if not isinstance(catalog, str):
        raise ValueError("Catalog must be a string")
    if not isinstance(projection, dict):
        raise ValueError("Projection must be a dictionary")
    if not callable(validate) and validate is not None:
        raise ValueError("Validate must be a callable, or None")

    # we want a mapper from id to (ra, dec)
    # so that when we loop over the xmatch results, we can easily find the original source
    # position and compute angular separation to each xmatch result
    id_to_position = {id: (ra, dec) for id, ra, dec in zip(ids, ras, decs)}

    # Create a list of tuples of (psid, ra, dec) for each source
    inputs = []
    for i in range(len(ids)):
        inputs.append((ids[i], ras[i], decs[i]))

    # Create batches of sources
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
                    catalog: {
                        "filter": {},
                        "projection": projection,
                    }
                },
            },
        }
        queries.append(query)

    responses = k.query(queries=queries, use_batch_query=True, max_n_threads=10)
    responses = responses.get("default")

    results = {}
    for response in responses:
        if response.get("status") != "success":
            print(f"Query failed with error: {response.get('message')}")
            continue
        for id, result in response.get("data").get(catalog).items():
            # Filter out sources that don't pass the quality cuts
            result_filtered = []
            if validate is None:
                result_filtered = result
            else:
                for source in result:
                    valid, _ = validate(source)
                    if valid:
                        result_filtered.append(source)
                    else:
                        # print(f"Source {id} failed quality cuts: {reason}")
                        pass
            result = result_filtered

            for source in result:
                try:
                    source["id"] = int(source[id_field])
                except ValueError:
                    source["id"] = str(source[id_field]).strip()
                source.pop(id_field)

            if len(result) == 0:
                results[int(id)] = None
                continue
            if len(result) == 1:
                results[int(id)] = result[0]
                continue

            # keep the closest source
            ra, dec = id_to_position[int(id)]
            min_dist = np.inf
            for source in result:
                dist = great_circle_distance(ra, dec, source["ra"], source["dec"])
                if dist < min_dist:
                    min_dist = dist
                    results[int(id)] = source

    return results


def query_by_id(
    k: Kowalski,
    ids: List[int],
    id_field: str = "_id",
    batch_size: int = 1000,
    catalog: str = None,
    projection: dict = None,
):
    """
    Query Kowalski by ID

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    ids : List[int]
        List of IDs
    id_field : str
        ID field
    batch_size : int
        Batch size
    catalog : str
        Catalog name
    projection : dict
        Projection

    Returns
    -------
    dict
        Query results
    """
    if not isinstance(k, Kowalski):
        raise ValueError("Kowalski must be an instance of the Kowalski class")
    if not isinstance(ids, list):
        raise ValueError("IDs must be provided as a list")
    if not isinstance(id_field, str):
        raise ValueError("ID field must be a string")
    if not isinstance(batch_size, int):
        raise ValueError("Batch size must be an integer")
    if not isinstance(catalog, str):
        raise ValueError("Catalog must be a string")
    if not isinstance(projection, dict):
        raise ValueError("Projection must be a dictionary")

    # Create batches of sources
    batches = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

    # if the type of the id is uint64, convert it to int
    for i, batch in enumerate(batches):
        batches[i] = [int(id) if isinstance(id, np.uint64) else id for id in batch]

    queries = []
    for batch in batches:
        # Create a list of dictionaries for each source
        query = {
            "query_type": "find",
            "query": {
                "catalog": catalog,
                "filter": {"_id": {"$in": batch}},
                "projection": projection,
            },
        }
        queries.append(query)

    responses = k.query(queries=queries, use_batch_query=True, max_n_threads=10)
    responses = responses.get("default")

    results = {}
    for response in responses:
        if response.get("status") != "success":
            print(f"Query failed with error: {response.get('message')}")
            continue
        for result in response.get("data"):
            id = result[id_field]
            result["id"] = result.pop(id_field)
            results[id] = result

    # for those that didn't return a result, add None
    for id in ids:
        if id not in results:
            results[id] = None

    return results


def query_ps1(
    k: Kowalski,
    ids: List[int],
    batch_size: int = 1000,
):
    """
    Query PanSTARRS1

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    ids : List[int]
        List of IDs
    batch_size : int
        Batch size

    Returns
    -------
    dict
        Query results
    """
    return query_by_id(
        k=k,
        ids=ids,
        id_field="_id",
        batch_size=batch_size,
        catalog="PS1_DR1",
        projection={
            "_id": 1,
            "ra": 1,
            "dec": 1,
            "gMeanPSFMag": 1,
            "gMeanPSFMagErr": 1,
            "rMeanPSFMag": 1,
            "rMeanPSFMagErr": 1,
            "iMeanPSFMag": 1,
            "iMeanPSFMagErr": 1,
            "zMeanPSFMag": 1,
            "zMeanPSFMagErr": 1,
            "yMeanPSFMag": 1,
            "yMeanPSFMagErr": 1,
        },
    )


def query_gaia(
    k: Kowalski,
    ids: List[int],
    ras: List[int],
    decs: List[int],
    radius: float,
    batch_size: int = 1000,
):
    """
    Query Gaia

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    ids : List[int]
        List of IDs
    ras : List[int]
        List of RAs
    decs : List[int]
        List of Decs
    radius : float
        Radius in arcseconds
    batch_size : int
        Batch size

    Returns
    -------
    dict
        Query results
    """
    return query_cone_search(
        k=k,
        ids=ids,
        ras=ras,
        decs=decs,
        radius=radius,
        batch_size=batch_size,
        catalog="Gaia_EDR3",
        projection={
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
        validate=validate_gaia_source,
    )


def query_2mass(
    k: Kowalski,
    ids: List[int],
    ras: List[int],
    decs: List[int],
    radius: float,
    batch_size: int = 1000,
):
    """
    Query 2MASS

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    ids : List[int]
        List of IDs
    ras : List[int]
        List of RAs
    decs : List[int]
        List of Decs
    radius : float
        Radius in arcseconds
    batch_size : int
        Batch size

    Returns
    -------
    dict
        Query results
    """
    return query_cone_search(
        k=k,
        ids=ids,
        ras=ras,
        decs=decs,
        radius=radius,
        batch_size=batch_size,
        catalog="2MASS_PSC",
        projection={
            "_id": 1,
            "ra": 1,
            "dec": "$decl",
            "j_m": 1,
            "j_cmsig": 1,
            "h_m": 1,
            "h_cmsig": 1,
            "k_m": 1,
            "k_cmsig": 1,
        },
        validate=validate_positional_source,
    )


def query_allwise(
    k: Kowalski,
    ids: List[int],
    ras: List[int],
    decs: List[int],
    radius: float,
    batch_size: int = 1000,
):
    """
    Query AllWISE

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    ids : List[int]
        List of IDs
    ras : List[int]
        List of RAs
    decs : List[int]
        List of Decs
    radius : float
        Radius in arcseconds
    batch_size : int
        Batch size

    Returns
    -------
    dict
        Query results
    """
    return query_cone_search(
        k=k,
        ids=ids,
        ras=ras,
        decs=decs,
        id_field="designation",
        radius=radius,
        batch_size=batch_size,
        catalog="AllWISE",
        projection={
            "designation": 1,
            "ra": 1,
            "dec": 1,
            "w1mpro": 1,
            "w1sigmpro": 1,
            "w2mpro": 1,
            "w2sigmpro": 1,
            "w3mpro": 1,
            "w3sigmpro": 1,
            "w4mpro": 1,
            "w4sigmpro": 1,
        },
        validate=validate_positional_source,
    )


def query_galex(
    k: Kowalski,
    ids: List[int],
    ras: List[int],
    decs: List[int],
    radius: float,
    batch_size: int = 1000,
):
    """
    Query GALEX

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    ids : List[int]
        List of IDs
    ras : List[int]
        List of RAs
    decs : List[int]
        List of Decs
    radius : float
        Radius in arcseconds
    batch_size : int
        Batch size

    Returns
    -------
    dict
        Query results
    """
    return query_cone_search(
        k=k,
        ids=ids,
        ras=ras,
        decs=decs,
        id_field="name",
        radius=radius,
        batch_size=batch_size,
        catalog="GALEX",
        projection={
            "ra": 1,
            "dec": 1,
            "name": 1,
            "b": 1,
            "NUVmag": 1,
            "e_NUVmag": 1,
        },
        validate=validate_positional_source,
    )
