import os
from typing import List

import numpy as np
import pandas as pd

from zvar_utils.kowalski import Kowalski, query_gaia
from zvar_utils.stats import calculate_cdf, find_extrema


class VariabilityCandidate:
    def __init__(
        self,
        psid,
        ra,
        dec,
        valid,
        freq,
        fap,
        best_M,
        gaia_id=None,
        gaia_G=None,
        gaia_BP=None,
        gaia_RP=None,
        gaia_parallax=None,
        gaia_parallax_error=None,
        gaia_pmra=None,
        gaia_pmra_error=None,
        gaia_pmdec=None,
        gaia_pmdec_error=None,
        gaia_MG=None,
        gaia_BP_RP=None,
    ):
        self.id = psid
        self.ra = ra
        self.dec = dec
        self.valid = valid
        self.freq = freq
        self.fap = fap
        self.best_M = best_M
        self.gaia_id = gaia_id
        self.gaia_G = gaia_G
        self.gaia_BP = gaia_BP
        self.gaia_RP = gaia_RP
        self.gaia_parallax = gaia_parallax
        self.gaia_parallax_error = gaia_parallax_error
        self.gaia_pmra = gaia_pmra
        self.gaia_pmra_error = gaia_pmra_error
        self.gaia_pmdec = gaia_pmdec
        self.gaia_pmdec_error = gaia_pmdec_error
        self.gaia_MG = gaia_MG
        self.gaia_BP_RP = gaia_BP_RP

    def set_gaia(
        self,
        id,
        G,
        BP,
        RP,
        parallax,
        parallax_error,
        pmra,
        pmra_error,
        pmdec,
        pmdec_error,
    ):
        self.gaia_id = id
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
            if (
                self.gaia_parallax / 1000 <= 0
                or np.isnan(self.gaia_parallax)
                or self.gaia_parallax is None
            ):
                self.gaia_MG = None
            else:
                self.gaia_MG = self.gaia_G + 5 * np.log10(self.gaia_parallax / 1000) + 5

    def find_gaia_BP_RP(self):
        if self.gaia_BP is not None and self.gaia_RP is not None:
            self.gaia_BP_RP = self.gaia_BP - self.gaia_RP


def get_candidates(
    psids: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    ratio_valid: np.ndarray,
    freqs: np.ndarray,
    sigs: np.ndarray,
) -> List[VariabilityCandidate]:
    # Precompute the CDFs for each bin
    cdfs = [calculate_cdf(sigs[:, j, 0]) for j in range(3)]

    extrema_20 = find_extrema(sigs[:, 0, 0], 0.999)
    print(f"Found {np.sum(extrema_20)} candidates in the 20-day bin")
    extrema_10 = find_extrema(sigs[:, 1, 0], 0.999)
    print(f"Found {np.sum(extrema_10)} candidates in the 10-day bin")
    extrema_5 = find_extrema(sigs[:, 2, 0], 0.999)
    print(f"Found {np.sum(extrema_5)} candidates in the 5-day bin")
    combined_candidates = np.logical_or.reduce([extrema_20, extrema_10, extrema_5])
    print(f"Found {np.sum(combined_candidates)} combined candidates")

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
            # min_prob_value = probabilities[min_prob_index]
            # Determine the best bin
            best_M = bins[min_prob_index]

            # Append the candidate with probabilities instead of significance values
            candidate_list.append(
                VariabilityCandidate(
                    psids[i],
                    ra[i],
                    dec[i],
                    ratio_valid[i],
                    freqs[i, :, 0],
                    probabilities,
                    best_M,
                )
            )

    return candidate_list


def add_gaia_xmatch_to_candidates(
    k: Kowalski, candidate_list: List[VariabilityCandidate], radius: float
) -> List[VariabilityCandidate]:
    if not isinstance(candidate_list, list):
        raise ValueError("Candidates must be provided as a list")
    if not all(
        isinstance(candidate, VariabilityCandidate) for candidate in candidate_list
    ):
        raise ValueError("Candidates must be of type Candidate")

    # Extract the psids, ras, and decs from the candidate list
    psids = [candidate.id for candidate in candidate_list]
    ras = [candidate.ra for candidate in candidate_list]
    decs = [candidate.dec for candidate in candidate_list]

    results = query_gaia(k, psids, ras, decs, radius)
    # Fill in the Gaia data for each candidate
    for candidate in candidate_list:
        result = results[str(candidate.id)]
        if result:  # If Gaia found anything
            result = result[0]  # It comes in a list of one element
            gaia_id = result["_id"] if "_id" in result else None
            pmra = result["pmra"] if "pmra" in result else None
            pmra_error = result["pmra_error"] if "pmra_error" in result else None
            pmdec = result["pmdec"] if "pmdec" in result else None
            pmdec_error = result["pmdec_error"] if "pmdec_error" in result else None
            parallax = result["parallax"] if "parallax" in result else None
            parallax_error = (
                result["parallax_error"] if "parallax_error" in result else None
            )
            G = result["phot_g_mean_mag"] if "phot_g_mean_mag" in result else None
            BP = result["phot_bp_mean_mag"] if "phot_bp_mean_mag" in result else None
            RP = result["phot_rp_mean_mag"] if "phot_rp_mean_mag" in result else None
            candidate.set_gaia(
                gaia_id,
                G,
                BP,
                RP,
                parallax,
                parallax_error,
                pmra,
                pmra_error,
                pmdec,
                pmdec_error,
            )
        candidate.find_gaia_MG()
        candidate.find_gaia_BP_RP()

    return candidate_list


def save_candidates_to_csv(
    candidate_list: List[VariabilityCandidate], field: int, band: int, path: str
):
    if not isinstance(candidate_list, list):
        raise ValueError("Candidates must be provided as a list")
    if not all(
        isinstance(candidate, VariabilityCandidate) for candidate in candidate_list
    ):
        raise ValueError("Candidates must be of type Candidate")
    if not isinstance(field, (int, np.integer, str)):
        raise ValueError("Field must be an integer or string")
    if not isinstance(band, (int, np.integer, str)):
        raise ValueError("Band must be an integer or string")
    if path is None:
        raise ValueError("Output directory must be specified")
    data = []
    for candidate in candidate_list:
        data.append(
            {
                "psid": candidate.id,
                "ra": candidate.ra,
                "dec": candidate.dec,
                "valid": candidate.valid,
                "best_M": candidate.best_M,
                "frequency_20": candidate.freq[0],
                "frequency_10": candidate.freq[1],
                "frequency_5": candidate.freq[2],
                "FAP_20": candidate.fap[0],
                "FAP_10": candidate.fap[1],
                "FAP_5": candidate.fap[2],
                "gaia_id": candidate.gaia_id,
                "G": candidate.gaia_G,
                "BP": candidate.gaia_BP,
                "RP": candidate.gaia_RP,
                "parallax": candidate.gaia_parallax,
                "parallax_error": candidate.gaia_parallax_error,
                "pmra": candidate.gaia_pmra,
                "pmra_error": candidate.gaia_pmra_error,
                "pmdec": candidate.gaia_pmdec,
                "pmdec_error": candidate.gaia_pmdec_error,
                "MG": candidate.gaia_MG,
                "BP_RP": candidate.gaia_BP_RP,
            }
        )

    # Construct the candidate path
    candidate_path = os.path.join(
        os.path.abspath(path),
        f"{str(field).zfill(4)}",
        f"fpw_{str(field).zfill(4)}_z{band}.csv",
    )

    # Check if the directory exists, and create it if it doesn't
    os.makedirs(os.path.dirname(candidate_path), exist_ok=True)

    # Create the DataFrame and write to CSV
    pd.DataFrame(data).to_csv(candidate_path, index=False)
