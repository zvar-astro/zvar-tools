import numpy as np
import h5py as h5
import glob as glob
import pandas as pd
import os
import argparse
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from astroquery.gaia import Gaia
import json
from penquins import Kowalski


class VariabilityCandidate:
    def __init__(self, psid, ra, dec, valid, freq, fap, best_M):
        self.id = psid
        self.ra = ra
        self.dec = dec
        self.valid = valid
        self.freq = freq
        self.fap = fap
        self.best_M = best_M
        self.gaia_G = None
        self.gaia_BP = None
        self.gaia_RP = None
        self.gaia_parallax = None
        self.gaia_parallax_error = None
        self.gaia_pmra = None
        self.gaia_pmra_error = None
        self.gaia_pmdec = None
        self.gaia_pmdec_error = None
        self.gaia_MG = None
        self.gaia_BP_RP = None

    def set_gaia(self, G, BP, RP, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error):
        self.gaia_G = G
        self.gaia_BP = BP
        self.gaia_RP = RP
        self.gaia_parallax = parallax
        self.gaia_parallax_error = parallax_error
        self.gaia_pmra = pmra
        self.gaia_pmra_error = pmra_error
        self.gaia_pmdec = pmdec
        self.gaia_pmdec_error = pmdec_error

    def find_gaia_MG(self):
        if self.gaia_G is not None and self.gaia_parallax is not None:
            self.gaia_MG = self.gaia_G + 5 * np.log10(self.gaia_parallax / 1000) + 5
            # self.gaia_MG = self.gaia_G - 5 * (1 - self.gaia_parallax / 1000)
    def find_gaia_BP_RP(self):
        if self.gaia_BP is not None and self.gaia_RP is not None:
            self.gaia_BP_RP = self.gaia_BP - self.gaia_RP


def load_field_data(field, band):
    field = str(field).zfill(4)
    files = []
    path = f'/data/zvar/zvar_results/{field}/*_z{band}.h5'
    files = glob.glob(path)
    
    psids = np.array([], dtype=np.uint64)
    ratio_valid = np.array([])
    best_freqs = np.array([])
    significances = np.array([])
    ra = np.array([])
    dec = np.array([])
    for file in files:
        with h5.File(file, 'r') as dataset:
            psids = np.append(psids, np.array(dataset['psids']))
            ratio_valid = np.append(ratio_valid, np.array(dataset['valid']))
            best_freqs = np.append(best_freqs, np.array(dataset['bestFreqs']))
            significances = np.append(significances, np.array(dataset['significance']))
            ra = np.append(ra, np.array(dataset['ra']))
            dec = np.append(dec, np.array(dataset['dec']))
            print(f'Loaded {file}')
    freqs = best_freqs.reshape((len(psids), 3, 50))
    sigs = significances.reshape((len(psids), 3, 50))
    sigs_clean = np.nan_to_num(sigs, nan=0, posinf=0, neginf=0)

    return psids, ra, dec, ratio_valid, freqs, sigs_clean


def calculate_cdf(data):
    # Filter data to include only values greater than 0
    filtered_data = data[data > 0]

    # Create a distribution of the filtered data
    hist, bin_edges = np.histogram(filtered_data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the CDF of the distribution
    cdf = np.cumsum(hist) * np.diff(bin_edges)
    return bin_centers, cdf


def find_extrema(data, cdf_value=0.999):
    bin_centers, cdf = calculate_cdf(data)

    threshold_index = np.searchsorted(cdf, cdf_value)
    threshold = bin_centers[threshold_index]

    candidate_indices = data > threshold
    return np.array(candidate_indices)


def parse_candidates(psids, ra, dec, ratio_valid, freqs, sigs):
    # Precompute the CDFs for each bin
    cdfs = [calculate_cdf(sigs[:, j, 0]) for j in range(3)]
    
    extrema_20 = find_extrema(sigs[:, 0, 0], 0.999)
    print(f'Found {np.sum(extrema_20)} candidates in the 20-day bin')
    extrema_10 = find_extrema(sigs[:, 1, 0], 0.999)
    print(f'Found {np.sum(extrema_10)} candidates in the 10-day bin')
    extrema_5 = find_extrema(sigs[:, 2, 0], 0.999)
    print(f'Found {np.sum(extrema_5)} candidates in the 5-day bin')
    combined_candidates = np.logical_or.reduce([extrema_20, extrema_10, extrema_5])
    print(f'Found {np.sum(combined_candidates)} combined candidates')
    
    candidate_list = []
    bins = [20, 10, 5]
    for i in range(len(psids)):
        if combined_candidates[i]:
            # Find the probability of the 0th value in each bin
            probabilities = []
            for j in range(3):
                bin_centers, cdf = cdfs[j]
                value = sigs[i, j, 0]
                if value > 0:
                    prob_index = np.searchsorted(bin_centers, value)
                    prob = cdf[prob_index] if prob_index < len(cdf) else 1.0
                else:
                    prob = 1.0  # Assign a high probability for non-positive values
                probabilities.append(1 - prob)
            
            # Determine which bin value is the least likely to have arisen from random chance
            min_prob_index = np.argmin(probabilities)
            min_prob_value = probabilities[min_prob_index]
            # Determine the best bin
            best_M = bins[min_prob_index]
            
            # Append the candidate with probabilities instead of significance values
            candidate_list.append(VariabilityCandidate(psids[i], ra[i], dec[i], ratio_valid[i], freqs[i, :, 0], probabilities, best_M))
    
    return candidate_list

def query_gaia(k, candidate_list, radius):

    #Extract the psids, ras, and decs from the candidate list
    psids = [candidate.id for candidate in candidate_list]
    ras = [candidate.ra for candidate in candidate_list]
    decs = [candidate.dec for candidate in candidate_list]

    #Create a list of tuples of (psid, ra, dec) for each source
    inputs = []
    for i in range(len(psids)):
        inputs.append((psids[i], ras[i], decs[i]))

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
                    str(psid): [float(ra), float(dec)] for psid, ra, dec in batch
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
        for psid, result in response.get("data").get("Gaia_EDR3").items():
            results[psid] = result
    #Fill in the Gaia data for each candidate
    for candidate in candidate_list:
        result = results[str(candidate.id)]
        if result: #If Gaia found anything
            result = result[0] #It comes in a list of one element
            pmra = result['pmra'] if 'pmra' in result else None
            pmra_error = result['pmra_error'] if 'pmra_error' in result else None
            pmdec = result['pmdec'] if 'pmdec' in result else None
            pmdec_error = result['pmdec_error'] if 'pmdec_error' in result else None
            parallax = result['parallax'] if 'parallax' in result else None
            parallax_error = result['parallax_error'] if 'parallax_error' in result else None
            G = result['phot_g_mean_mag'] if 'phot_g_mean_mag' in result else None
            BP = result['phot_bp_mean_mag'] if 'phot_bp_mean_mag' in result else None
            RP = result['phot_rp_mean_mag'] if 'phot_rp_mean_mag' in result else None
            candidate.set_gaia(G, BP, RP, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error)
        candidate.find_gaia_MG()
        candidate.find_gaia_BP_RP()
    


def write_csv(candidate_list, field, band):
    data = []
    for candidate in candidate_list:
        psid = candidate.id
        ra = candidate.ra
        dec = candidate.dec
        valid = candidate.valid
        frequency = candidate.freq
        fap = candidate.fap
        best_M = candidate.best_M
        gaia_G = candidate.gaia_G
        gaia_BP = candidate.gaia_BP
        gaia_RP = candidate.gaia_RP
        gaia_parallax = candidate.gaia_parallax
        gaia_parallax_error = candidate.gaia_parallax_error
        gaia_pmra = candidate.gaia_pmra
        gaia_pmra_error = candidate.gaia_pmra_error
        gaia_pmdec = candidate.gaia_pmdec
        gaia_pmdec_error = candidate.gaia_pmdec_error
        gaia_MG = candidate.gaia_MG
        gaia_BP_RP = candidate.gaia_BP_RP
        
        data.append({
            'psid': psid,
            'ra': ra,
            'dec': dec,
            'valid': valid,
            'best_M': best_M,
            'frequency_20': frequency[0],
            'frequency_10': frequency[1],
            'frequency_5': frequency[2],
            'FAP_20': fap[0],
            'FAP_10': fap[1],
            'FAP_5': fap[2],
            'G': gaia_G,
            'BP': gaia_BP,
            'RP': gaia_RP,
            'parallax': gaia_parallax,
            'parallax_error': gaia_parallax_error,
            'pmra': gaia_pmra,
            'pmra_error': gaia_pmra_error,
            'pmdec': gaia_pmdec,
            'pmdec_error': gaia_pmdec_error,
            'MG': gaia_MG,
            'BP_RP': gaia_BP_RP
        })
    
    # Construct the candidate path
    candidate_path = f'/data/zvar/variability_candidates/{str(field).zfill(4)}/fpw_{str(field).zfill(4)}_z{band}.csv'
    
    # Extract the directory path
    directory = os.path.dirname(candidate_path)
    
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create the DataFrame and write to CSV
    df = pd.DataFrame(data)
    df.to_csv(candidate_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run candidate detection on ZVAR results')
    parser.add_argument('field1', type=int, help='Field number on the low end of the range (inclusive)')
    parser.add_argument('field2', type=int, help='Field number on the high end of the range (inclusive)')

    args = parser.parse_args()

    #Connect to Kowalski
    credentials = "/data/swhitebook/passwd.json"
    with open(credentials) as f:
        passwd = json.load(f)
    username = passwd["melman"]["username"]
    password = passwd["melman"]["password"]

    k = Kowalski(
    protocol="https",
    host="melman.caltech.edu",
    port=443,
    username=username,
    password=password,
    verbose=True,
    timeout=600,
    )

    bands = ['g', 'r']
    fields = np.arange(args.field1, args.field2 + 1)
    #convert every folder to a string and zfill it to 4 digits
    fields = [str(field).zfill(4) for field in fields]

    for field in fields:
        for band in bands:
            psids, ra, dec, ratio_valid, freqs, sigs_clean = load_field_data(field, band) #Load the data
            candidate_list = parse_candidates(psids, ra, dec, ratio_valid, freqs, sigs_clean) #Find the candidates
            query_gaia(k, candidate_list, 5) #Fill in the Gaia data
            write_csv(candidate_list, field, band) #Write the candidates to a CSV file