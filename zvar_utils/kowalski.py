from typing import List
import os
import json

import numpy as np
from penquins import Kowalski

from zvar_utils.photometry import process_curve, remove_deep_drilling

def connect_to_kowalski(path_credentials: str, verbose: bool = True) -> Kowalski:
    if path_credentials is None:
        raise ValueError('Credentials file must be specified')
    if not os.path.exists(path_credentials):
        raise ValueError(f'Credentials file {path_credentials} does not exist')
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

def query_gaia(k: Kowalski, ids: List[int], ras: List[int], decs: List[int], radius: float) -> dict:
    if not isinstance(k, Kowalski):
        raise ValueError('Kowalski must be an instance of the Kowalski class')
    if not isinstance(ids, list):
        raise ValueError('IDs must be provided as a list')
    if not isinstance(ras, list):
        raise ValueError('RAs must be provided as a list')
    if not isinstance(decs, list):
        raise ValueError('Decs must be provided as a list')
    if not isinstance(radius, (int, float)):
        raise ValueError('Radius must be a number')

    #Create a list of tuples of (psid, ra, dec) for each source
    inputs = []
    for i in range(len(ids)):
        inputs.append((ids[i], ras[i], decs[i]))

    #Create batches of 1000 sources
    batch_size = 1000
    batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]

    queries = []
    for batch in batches:
        #Create a list of dictionaries for each source
        query = {
            "query_type": "near",
            "query": {
                "max_distance": radius,
                "distance_units": "arcsec",
                "radec": {
                    str(id): [float(ra), float(dec)] for id, ra, dec in batch
                },
                "catalogs": {
                    "Gaia_EDR3": {
                        "filter": {},
                        "projection": {
                            'pmra': 1,
                            'pmra_error': 1,
                            'pmdec': 1,
                            'pmdec_error': 1,
                            'parallax': 1,
                            'parallax_error': 1,
                            'phot_g_mean_mag': 1,
                            'phot_bp_mean_mag': 1,
                            'phot_rp_mean_mag': 1
                        }
                    }
                }
            },
            "kwargs": {
                "limit": 1
            }
        }
        queries.append(query)

    responses = k.query(queries = queries, use_batch_query=True, max_n_threads=30)
    responses = responses.get("default")

    results = {}
    for response in responses:
        if response.get("status") != "success":
            print(f'Query failed with error: {response.get("message")}')
            continue
        for id, result in response.get("data").get("Gaia_EDR3").items():
            results[id] = result

    return results
    
def get_ipac_dr_lightcurves(k: Kowalski, ra: float, dec: float, min_epochs: int = 50):
    if not isinstance(k, Kowalski):
        raise ValueError('Kowalski must be an instance of the Kowalski class')
    if not isinstance(ra, (int, float)):
        raise ValueError('RA must be a number')
    if not isinstance(dec, (int, float)):
        raise ValueError('Dec must be a number')
    if ra < 0 or ra > 360:
        raise ValueError('RA must be between 0 and 360')
    if dec < -90 or dec > 90:
        raise ValueError('Dec must be between -90 and 90')
    if not isinstance(min_epochs, int):
        raise ValueError('Min epochs must be an integer')
    if min_epochs < 1:
        raise ValueError('Min epochs must be greater than 0')
    
    query = {
        "query_type": "cone_search",
        "query": {
            "object_coordinates": {
                "cone_search_radius": 5,
                "cone_search_unit": "arcsec",
                "radec": {
                    'object': [
                        ra,
                        dec
                    ]
                }
            },
            "catalogs": {
                "ZTF_sources_20240117": {
                    "filter": {},
                    "projection": {
                        'data.hjd': 1,
                        'data.mag': 1,
                        'data.magerr': 1,
                        'data.ra': 1,
                        'data.dec': 1,
                        'data.fid': 1,
                        'data.programid': 1,
                        'data.catflags': 1,
                        'filter':1
                    }
                }
            }
        },
        "kwargs": {
            "filter_first": False
        }
    }

    response = k.query(query=query).get("default")
    if response.get("status") != "success":
        print(f'Query failed with error: {response.get("message")}')
        return None
    
    data = response.get("data")

    key = list(data.keys())[0]
    data = data[key]
    key = list(data.keys())[0]
    data = data[key]
    lightcurves = {}
    num_epochs = 0
    for datlist in data:
        objid = str(datlist["_id"])
        # print(datlist)
        ztf_filter = datlist["filter"]
        dat = datlist["data"]
        hjd, mag, magerr, coords_out, ztf_id, ztf_filt = [], [], [], [], [], []
        for dic in dat:
            if dic["catflags"]==0:
                hjd.append(dic["hjd"])
                mag.append(dic["mag"])
                magerr.append(dic["magerr"])
            coords_out.append(dic["ra"])
            coords_out.append(dic["dec"])
            ztf_id.append(objid)
            ztf_filt.append(ztf_filter)
        if len(hjd) < min_epochs: continue
        if len(hjd) > num_epochs:
            num_epochs = len(hjd)
            lightcurves["hjd"] = np.array(hjd)
            lightcurves["mag"] = np.array(mag)
            lightcurves["magerr"] = np.array(magerr)
            lightcurves["coords"] = np.array(coords_out)
            lightcurves['objid'] = objid
            lightcurves['filter'] = ztf_filter

        barycorr_times, flux, ferrs = process_curve(ra, dec, lightcurves['hjd'], lightcurves['mag'], lightcurves['magerr'])
        barycorr_times, flux, ferrs = remove_deep_drilling(barycorr_times, flux, ferrs)

        return barycorr_times, flux, ferrs
