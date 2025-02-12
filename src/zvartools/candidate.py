import os
from typing import List, Union

import numpy as np
import pandas as pd

from zvartools.external import (
    Kowalski,
    query_gaia,
    query_ps1,
    query_2mass,
    query_allwise,
)
from zvartools.stats import get_cdf, get_extrema
from zvartools.enums import BIN_IDX_TO_FREQ_COL, BIN_IDX_TO_FAP_COL


class PS1Match:
    def __init__(self, id, g, g_err, r, r_err, i, i_err, z, z_err, y, y_err):
        self.id = id
        self.g = g
        self.g_err = g_err
        self.r = r
        self.r_err = r_err
        self.i = i
        self.i_err = i_err
        self.z = z
        self.z_err = z_err
        self.y = y
        self.y_err = y_err

    def __repr__(self):
        return f"G: {self.g:.2f}±{self.g_err:.2f}, R: {self.r:.2f}±{self.r_err:.2f}, I: {self.i:.2f}±{self.i_err:.2f}, Z: {self.z:.2f}±{self.z_err:.2f}, Y: {self.y:.2f}±{self.y_err:.2f}"

    def __str__(self):
        return self.__repr__()


class GaiaMatch:
    def __init__(
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
        BP_RP=None,
        MG=None,
    ):
        self.id = id
        self.G = G
        self.BP = BP
        self.RP = RP
        self.parallax = parallax
        self.parallax_error = parallax_error
        self.pmra = pmra
        self.pmra_error = pmra_error
        self.pmdec = pmdec
        self.pmdec_error = pmdec_error

        self.add_bp_rp()
        self.add_MG()

    def __repr__(self):
        return f"G: {self.G:.2f}, BP: {self.BP:.2f}, RP: {self.RP:.2f}, Parallax: {self.parallax:.2f}±{self.parallax_error:.2f}, PMRA: {self.pmra:.2f}±{self.pmra_error:.2f}, PMDEC: {self.pmdec:.2f}±{self.pmdec_error:.2f}"

    def __str__(self):
        return self.__repr__()

    def add_bp_rp(self):
        if self.BP is not None and self.RP is not None:
            self.BP_RP = self.BP - self.RP

    def add_MG(self):
        if self.G is not None and self.parallax is not None:
            if (
                self.parallax / 1000 <= 0
                or np.isnan(self.parallax)
                or self.parallax is None
            ):
                self.MG = None
            else:
                self.MG = self.G + 5 * np.log10(self.parallax / 1000) + 5


class TwoMASSMatch:
    def __init__(self, id, j, j_err, h, h_err, k, k_err):
        self.id = id
        self.j = j
        self.j_err = j_err
        self.h = h
        self.h_err = h_err
        self.k = k
        self.k_err = k_err

    def __repr__(self):
        return f"J: {self.j:.2f}±{self.j_err:.2f}, H: {self.h:.2f}±{self.h_err:.2f}, K: {self.k:.2f}±{self.k_err:.2f}"

    def __str__(self):
        return self.__repr__()


class AllWISEMatch:
    def __init__(self, id, w1, w1_err, w2, w2_err, w3, w3_err, w4, w4_err):
        self.id = id
        self.w1 = w1
        self.w1_err = w1_err
        self.w2 = w2
        self.w2_err = w2_err
        self.w3 = w3
        self.w3_err = w3_err
        self.w4 = w4
        self.w4_err = w4_err

    def __repr__(self):
        return f"W1: {self.w1:.2f}±{self.w1_err:.2f}, W2: {self.w2:.2f}±{self.w2_err:.2f}, W3: {self.w3:.2f}±{self.w3_err:.2f}, W4: {self.w4:.2f}±{self.w4_err:.2f}"

    def __str__(self):
        return self.__repr__()


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
        field: int = None,
        ccd: int = None,
        quad: int = None,
        band: str = None,
        ps1: Union[PS1Match, dict] = None,
        gaia: Union[GaiaMatch, dict] = None,
        twomass: Union[TwoMASSMatch, dict] = None,
        allwise: Union[AllWISEMatch, dict] = None,
    ):
        self.psid = psid
        self.ra = ra
        self.dec = dec
        self.valid = valid
        self.freq = freq
        self.fap = fap
        self.best_M = best_M
        self.field = field
        self.ccd = ccd
        self.quad = quad
        self.band = band
        self.set_ps1(ps1)
        self.set_gaia(gaia)
        self.set_2mass(twomass)
        self.set_allwise(allwise)

    @property
    def period(self):
        """
        Calculate the period of the candidate, in hours
        """
        if self.freq is None:
            return None
        return 1 / self.freq * 24

    def set_ps1(
        self,
        ps1: Union[PS1Match, dict, None],
    ):
        """
        Set the PS1 data for the candidate

        Parameters
        ----------
        ps1 : Union[PS1Match, dict, None]
            PS1 data

        Raises
        ------
        ValueError
            If the PS1 data is not of the correct type
        """
        if isinstance(ps1, PS1Match):
            self.ps1 = ps1
        elif isinstance(ps1, dict):
            self.ps1 = PS1Match(**ps1)
        elif ps1 is None:
            self.ps1 = None
        else:
            raise ValueError("PS1 must be a PS1Match, a dictionary, or None")

    def set_gaia(
        self,
        gaia: Union[GaiaMatch, dict, None],
    ):
        """
        Set the Gaia data for the candidate

        Parameters
        ----------
        gaia : Union[GaiaMatch, dict, None]
            Gaia data

        Raises
        ------
        ValueError
            If the Gaia data is not of the correct type
        """
        if isinstance(gaia, GaiaMatch):
            self.gaia = gaia
        elif isinstance(gaia, dict):
            self.gaia = GaiaMatch(**gaia)
        elif gaia is None:
            self.gaia = None
        else:
            raise ValueError("Gaia must be a GaiaMatch, a dictionary, or None")

    def set_2mass(
        self,
        twomass: Union[TwoMASSMatch, dict, None],
    ):
        """
        Set the 2MASS data for the candidate

        Parameters
        ----------
        twomass : Union[TwoMASSMatch, dict, None]
            2MASS data

        Raises
        ------
        ValueError
            If the 2MASS data is not of the correct type
        """
        if isinstance(twomass, TwoMASSMatch):
            self.twomass = twomass
        elif isinstance(twomass, dict):
            self.twomass = TwoMASSMatch(**twomass)
        elif twomass is None:
            self.twomass = None
        else:
            raise ValueError("2MASS must be a TwoMASSMatch, a dictionary, or None")

    def set_allwise(
        self,
        allwise: Union[AllWISEMatch, dict, None],
    ):
        """
        Set the AllWISE data for the candidate

        Parameters
        ----------
        allwise : Union[AllWISEMatch, dict, None]
            AllWISE data

        Raises
        ------
        ValueError
            If the AllWISE data is not of the correct type
        """
        if isinstance(allwise, AllWISEMatch):
            self.allwise = allwise
        elif isinstance(allwise, dict):
            self.allwise = AllWISEMatch(**allwise)
        elif allwise is None:
            self.allwise = None
        else:
            raise ValueError("AllWISE must be a AllWISEMatch, a dictionary, or None")

    def __repr__(self):
        return (
            f"ID: {self.psid}, RA: {self.ra}, Dec: {self.dec}, Valid: {self.valid}, Best M: {self.best_M}, freq: {self.freq}, FAP: {self.fap}"
            + (
                f" (field: {self.field}, ccd: {self.ccd}, quad: {self.quad})"
                if self.field and self.ccd and self.quad
                else ""
            )
            + (f"\n    PS1: {self.ps1}" if self.ps1 else "")
            + (f"\n    Gaia: {self.gaia}" if self.gaia else "")
            + (f"\n    2MASS: {self.twomass}" if self.twomass else "")
            + (f"\n    AllWISE: {self.allwise}" if self.allwise else "")
        )


def get_candidates(
    psids: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    ratio_valid: np.ndarray,
    freqs: np.ndarray,
    sigs: np.ndarray,
    field: int,
    ccds: np.ndarray,
    quads: np.ndarray,
) -> List[VariabilityCandidate]:
    """
    Get the variability candidates from the data

    Parameters
    ----------
    psids : np.ndarray
        PS1 IDs
    ra : np.ndarray
        Right ascensions
    dec : np.ndarray
        Declinations
    ratio_valid : np.ndarray
        Ratio of valid data points
    freqs : np.ndarray
        Frequencies
    sigs : np.ndarray
        Significances
    field : int
        Field number
    ccds : np.ndarray
        CCD numbers
    quads : np.ndarray
        Quadrant numbers

    Returns
    -------
    List[VariabilityCandidate]
        List of variability candidates
    """
    # Precompute the CDFs for each bin
    cdfs = [get_cdf(sigs[:, j, 0]) for j in range(3)]

    extrema_20 = get_extrema(sigs[:, 0, 0], 0.999)
    print(f"Found {np.sum(extrema_20)} candidates in the 20-day bin")
    extrema_10 = get_extrema(sigs[:, 1, 0], 0.999)
    print(f"Found {np.sum(extrema_10)} candidates in the 10-day bin")
    extrema_5 = get_extrema(sigs[:, 2, 0], 0.999)
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
                    field,
                    ccds[i],
                    quads[i],
                )
            )

    return candidate_list


def add_ps1_xmatch_to_candidates(
    k: Kowalski, candidate_list: List[VariabilityCandidate]
) -> List[VariabilityCandidate]:
    """
    Add PS1 crossmatches to the candidates

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    candidate_list : List[VariabilityCandidate]
        List of variability candidates

    Returns
    -------
    List[VariabilityCandidate]
        List of variability candidates with PS1 crossmatches
    """
    # query_ps1 only needs the psids
    psids = [candidate.psid for candidate in candidate_list]
    xmatches = query_ps1(k, psids)
    # print how many sources have ps1 matches
    print(
        f"Found {len([x for x in xmatches.values() if x is not None])} PS1 matches, out of {len(candidate_list)} sources"
    )
    # Fill in the PS1 data for each candidate
    for candidate in candidate_list:
        result = xmatches.get(candidate.psid)
        if result:
            candidate.set_ps1(
                PS1Match(
                    candidate.psid,
                    result.get("gMeanPSFMag"),
                    result.get("gMeanPSFMagErr"),
                    result.get("rMeanPSFMag"),
                    result.get("rMeanPSFMagErr"),
                    result.get("iMeanPSFMag"),
                    result.get("iMeanPSFMagErr"),
                    result.get("zMeanPSFMag"),
                    result.get("zMeanPSFMagErr"),
                    result.get("yMeanPSFMag"),
                    result.get("yMeanPSFMagErr"),
                )
            )

    return candidate_list


def add_gaia_xmatch_to_candidates(
    k: Kowalski, candidate_list: List[VariabilityCandidate], radius: float = 2.0
) -> List[VariabilityCandidate]:
    """
    Add Gaia crossmatches to the candidates

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    candidate_list : List[VariabilityCandidate]
        List of variability candidates
    radius : float
        Radius in arcseconds to search for xmatches in external catalogs

    Returns
    -------
    List[VariabilityCandidate]
        List of variability candidates with Gaia crossmatches
    """
    if not isinstance(candidate_list, list):
        raise ValueError("Candidates must be provided as a list")
    if not all(
        isinstance(candidate, VariabilityCandidate) for candidate in candidate_list
    ):
        raise ValueError("Candidates must be of type Candidate")

    # Extract the psids, ras, and decs from the candidate list
    psids = [candidate.psid for candidate in candidate_list]
    ras = [candidate.ra for candidate in candidate_list]
    decs = [candidate.dec for candidate in candidate_list]

    xmatches = query_gaia(k, psids, ras, decs, radius)
    # print how many sources have gaia matches
    print(
        f"Found {len([x for x in xmatches.values() if x is not None])} Gaia matches, out of {len(candidate_list)} sources"
    )
    # Fill in the Gaia data for each candidate
    for candidate in candidate_list:
        result = xmatches.get(candidate.psid)
        if result and result.get("id"):  # If Gaia found anything
            candidate.set_gaia(
                GaiaMatch(
                    result.get("id"),
                    result.get("phot_g_mean_mag"),
                    result.get("phot_bp_mean_mag"),
                    result.get("phot_rp_mean_mag"),
                    result.get("parallax"),
                    result.get("parallax_error"),
                    result.get("pmra"),
                    result.get("pmra_error"),
                    result.get("pmdec"),
                    result.get("pmdec_error"),
                )
            )

    return candidate_list


def add_2mass_xmatch_to_candidates(
    k: Kowalski, candidate_list: List[VariabilityCandidate], radius: float = 2.0
) -> List[VariabilityCandidate]:
    """
    Add 2MASS crossmatches to the candidates

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    candidate_list : List[VariabilityCandidate]
        List of variability candidates
    radius : float
        Radius in arcseconds to search for xmatches in external catalogs

    Returns
    -------
    List[VariabilityCandidate]
        List of variability candidates with 2MASS crossmatches
    """
    if not isinstance(candidate_list, list):
        raise ValueError("Candidates must be provided as a list")
    if not all(
        isinstance(candidate, VariabilityCandidate) for candidate in candidate_list
    ):
        raise ValueError("Candidates must be of type Candidate")

    # Extract the psids, ras, and decs from the candidate list
    psids = [candidate.psid for candidate in candidate_list]
    ras = [candidate.ra for candidate in candidate_list]
    decs = [candidate.dec for candidate in candidate_list]

    xmatches = query_2mass(k, psids, ras, decs, radius)
    # print how many sources have 2mass matches
    print(
        f"Found {len([x for x in xmatches.values() if x is not None])} 2MASS matches, out of {len(candidate_list)} sources"
    )
    # Fill in the 2MASS data for each candidate
    for candidate in candidate_list:
        result = xmatches.get(candidate.psid)
        if result and result.get("id"):  # If 2MASS found anything
            candidate.set_2mass(
                TwoMASSMatch(
                    result.get("id"),
                    result.get("j_m"),
                    result.get("j_msigcom"),
                    result.get("h_m"),
                    result.get("h_msigcom"),
                    result.get("k_m"),
                    result.get("k_msigcom"),
                )
            )

    return candidate_list


def add_allwise_xmatch_to_candidates(
    k: Kowalski, candidate_list: List[VariabilityCandidate], radius: float = 2.0
) -> List[VariabilityCandidate]:
    """
    Add AllWISE crossmatches to the candidates

    Parameters
    ----------
    k : Kowalski
        Kowalski instance
    candidate_list : List[VariabilityCandidate]
        List of variability candidates
    radius : float
        Radius in arcseconds to search for xmatches in external catalogs

    Returns
    -------
    List[VariabilityCandidate]
        List of variability candidates with AllWISE crossmatches
    """
    if not isinstance(candidate_list, list):
        raise ValueError("Candidates must be provided as a list")
    if not all(
        isinstance(candidate, VariabilityCandidate) for candidate in candidate_list
    ):
        raise ValueError("Candidates must be of type Candidate")

    # Extract the psids, ras, and decs from the candidate list
    psids = [candidate.psid for candidate in candidate_list]
    ras = [candidate.ra for candidate in candidate_list]
    decs = [candidate.dec for candidate in candidate_list]

    xmatches = query_allwise(k, psids, ras, decs, radius)
    # print how many sources have allwise matches
    print(
        f"Found {len([x for x in xmatches.values() if x is not None])} AllWISE matches, out of {len(candidate_list)} sources"
    )
    # Fill in the AllWISE data for each candidate
    for candidate in candidate_list:
        result = xmatches.get(candidate.psid)
        if result and result.get("id"):  # If AllWISE found anything
            candidate.set_allwise(
                AllWISEMatch(
                    result.get("id"),
                    result.get("w1mpro"),
                    result.get("w1sigmpro"),
                    result.get("w2mpro"),
                    result.get("w2sigmpro"),
                    result.get("w3mpro"),
                    result.get("w3sigmpro"),
                    result.get("w4mpro"),
                    result.get("w4sigmpro"),
                )
            )

    return candidate_list


def export_to_parquet(
    candidate_list: List[VariabilityCandidate], field: int, band: int, path: str
):
    """
    Export the candidates to a parquet file

    Parameters
    ----------
    candidate_list : List[VariabilityCandidate]
        List of variability candidates
    field : int
        Field number
    band : int
        Band number
    path : str
        Output directory

    Raises
    ------
    ValueError
        If any of the arguments is invalid
    """
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

    # before we export the data, we want to sort the candidates by their id (psid)
    # to make sure we get consistent results when rerunning the code
    candidate_list = sorted(candidate_list, key=lambda x: x.psid)

    data = []
    for candidate in candidate_list:
        candidate_data = {
            "psid": candidate.psid,
            "ra": candidate.ra,
            "dec": candidate.dec,
            "valid": candidate.valid,
            "best_M": candidate.best_M,
            "field": candidate.field,
            "ccd": candidate.ccd,
            "quad": candidate.quad,
            "frequency_20": candidate.freq[0],
            "frequency_10": candidate.freq[1],
            "frequency_5": candidate.freq[2],
            "FAP_20": candidate.fap[0],
            "FAP_10": candidate.fap[1],
            "FAP_5": candidate.fap[2],
        }
        if candidate.ps1:
            candidate_data.update(
                {
                    f"ps_{key}": getattr(candidate.ps1, key)
                    for key in vars(candidate.ps1)
                    if key != "id"
                }
            )
        if candidate.gaia:
            candidate_data.update(
                {
                    f"gaia_{key}": getattr(candidate.gaia, key)
                    for key in vars(candidate.gaia)
                }
            )
        if candidate.twomass:
            candidate_data.update(
                {
                    f"twomass_{key}": getattr(candidate.twomass, key)
                    for key in vars(candidate.twomass)
                }
            )
        if candidate.allwise:
            candidate_data.update(
                {
                    f"allwise_{key}": getattr(candidate.allwise, key)
                    for key in vars(candidate.allwise)
                }
            )
        data.append(candidate_data)
    # Construct the candidate path
    candidate_path = os.path.join(
        os.path.abspath(path),
        f"{str(field).zfill(4)}",
        f"fpw_{str(field).zfill(4)}_z{band}.parquet",
    )

    # Check if the directory exists, and create it if it doesn't
    os.makedirs(os.path.dirname(candidate_path), exist_ok=True)

    # Create the DataFrame and write to parquet
    pd.DataFrame(data).to_parquet(candidate_path, index=False)


def import_from_parquet(path: str, best_m_only=True) -> List[VariabilityCandidate]:
    """
    Import the candidates from a parquet file

    Parameters
    ----------
    path : str
        Path to the parquet file
    best_m_only : bool
        Whether to only use the best M value

    Returns
    -------
    List[VariabilityCandidate]
        List of variability candidates
    """
    if not os.path.exists(path):
        raise ValueError(f"File {path} does not exist")
    df = pd.read_parquet(path)
    candidate_list = []

    # get the filter from the path name
    try:
        band = str(path.split("_z")[-1].split(".")[0])
    except ValueError:
        band = None

    for _, row in df.iterrows():
        ps1_data = (
            {
                key.replace("ps_", ""): row[key]
                for key in row.keys()
                if key.startswith("ps_")
            }
            if any(key.startswith("ps_") for key in row.keys())
            else None
        )
        ps1_data.update({"id": row["psid"]}) if ps1_data else None

        gaia_data = (
            {
                key.replace("gaia_", ""): row[key]
                for key in row.keys()
                if key.startswith("gaia_")
            }
            if row.get("gaia_id")
            else None
        )

        twomass_data = (
            {
                key.replace("twomass_", ""): row[key]
                for key in row.keys()
                if key.startswith("twomass_")
            }
            if row.get("twomass_id")
            else None
        )

        allwise_data = (
            {
                key.replace("allwise_", ""): row[key]
                for key in row.keys()
                if key.startswith("allwise_")
            }
            if row.get("allwise_id")
            else None
        )

        if not best_m_only:
            candidate = VariabilityCandidate(
                psid=row["psid"],
                ra=row["ra"],
                dec=row["dec"],
                valid=row["valid"],
                freq=np.array(
                    [row["frequency_20"], row["frequency_10"], row["frequency_5"]]
                ),
                fap=np.array([row["FAP_20"], row["FAP_10"], row["FAP_5"]]),
                best_M=row["best_M"],
                field=row.get("field"),
                ccd=row.get("ccd"),
                quad=row.get("quad"),
                band=band,
                ps1=ps1_data,
                gaia=gaia_data,
                twomass=twomass_data,
                allwise=allwise_data,
            )
        else:
            bin_idx = 0
            if row["best_M"] == 10:
                bin_idx = 1
            elif row["best_M"] == 5:
                bin_idx = 2

            candidate = VariabilityCandidate(
                psid=row["psid"],
                ra=row["ra"],
                dec=row["dec"],
                valid=row["valid"],
                freq=row[BIN_IDX_TO_FREQ_COL[bin_idx]],
                fap=row[BIN_IDX_TO_FAP_COL[bin_idx]],
                best_M=row["best_M"],
                field=row.get("field"),
                ccd=row.get("ccd"),
                quad=row.get("quad"),
                band=band,
                ps1=ps1_data,
                gaia=gaia_data,
                twomass=twomass_data,
                allwise=allwise_data,
            )
        candidate_list.append(candidate)

    return candidate_list
