from typing import List, Tuple

import numpy as np


def remove_nondetections(
    time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove data points with flux = NaN

    Parameters
    ----------
    time : np.ndarray
        Time of observation
    flux : np.ndarray
        Flux of the object
    flux_err : np.ndarray
        Error in the flux
    filter : np.ndarray
        Filter used for the observation

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Time, flux, flux_err, filter with NaN values removed
    """
    mask = np.isnan(flux)
    return (
        time[~mask],
        flux[~mask],
        flux_err[~mask],
        filter[~mask],
    )


def get_cdf(data: np.ndarray) -> np.ndarray:
    """
    Get the Cumulative Distribution Function (CDF) of the data

    Parameters
    ----------
    data : np.ndarray
        Data to calculate the CDF for

    Returns
    -------
    np.ndarray
        Bin centers and CDF values
    """
    # Filter data to include only values greater than 0
    filtered_data = data[data > 0]

    # Create a distribution of the filtered data
    hist, bin_edges = np.histogram(filtered_data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the CDF of the distribution
    cdf = np.cumsum(hist) * np.diff(bin_edges)
    return bin_centers, cdf


def get_extrema(data: np.ndarray, cdf_value: float = 0.999) -> np.ndarray:
    """
    Get the extrema of the data

    Parameters
    ----------
    data : np.ndarray
        Data to calculate the extrema for (e.g. flux)
    cdf_value : float, optional
        CDF value to use as the threshold, by default 0.999

    Returns
    -------
    np.ndarray
        Indices of the extrema
    """
    bin_centers, cdf = get_cdf(data)

    threshold_index = np.searchsorted(cdf, cdf_value)
    threshold = bin_centers[threshold_index]

    candidate_indices = data > threshold
    return np.array(candidate_indices)


def get_string_length(
    candidate: object,
    photometry: List[np.ndarray],
    period: float = None,
) -> float:
    """
    Calculate the string length of the candidate

    Parameters
    ----------
    candidate : VariableCandidate
        Variable candidate
    photometry : List[np.ndarray]
        Photometry data
    period : float, optional
        Period of the candidate, by default None

    Returns
    -------
    float
        String length of the candidate
    """
    if period is None:
        period = 1 / candidate.freq
    # remove data points with flux = NaN
    time, flux, _, _ = remove_nondetections(
        photometry[0], photometry[1], photometry[2], photometry[3]
    )

    phase = (time / (period * 86400)) % 2  # period is converted from days to seconds

    sorted_indices = np.argsort(phase)
    phase = phase[sorted_indices]
    flux = flux[sorted_indices]

    # Calculate string length as sum of distances between consecutive points
    string_length = 0.0
    for i in range(len(phase) - 1):
        delta_phase = phase[i + 1] - phase[i]
        delta_flux = flux[i + 1] - flux[i]
        string_length += np.sqrt(delta_phase**2 + delta_flux**2)

    # Optionally: Close the loop by connecting the last point back to the first
    delta_phase = 1.0 - phase[-1] + phase[0]  # wrap-around in phase
    delta_flux = flux[-1] - flux[0]
    string_length += np.sqrt(delta_phase**2 + delta_flux**2)

    string_length = string_length / len(phase)

    return string_length


def get_entropy(
    candidate: object,
    photometry: List[np.ndarray],
    period: float = None,
    num_bins: int = 10,
):
    """
    Calculate the entropy of the candidate

    Parameters
    ----------
    candidate : VariableCandidate
        Variable candidate
    photometry : List[np.ndarray]
        Photometry data
    period : float, optional
        Period of the candidate, by default None
    num_bins : int, optional
        Number of bins to use for the entropy calculation, by default 10

    Returns
    -------
    Tuple[float, float, float, float]
        Entropy, minimum entropy, negative entropy expectation, sigma expectation
    """
    if period is None:
        period = 1 / candidate.freq
    # remove data points with flux = NaN
    time, flux, _, _ = remove_nondetections(
        photometry[0], photometry[1], photometry[2], photometry[3]
    )

    phase = (time / (period * 86400)) % 2  # period is converted from days to seconds

    # fold the period (flux) to have the points in the same order as the phase
    sorted_indices = np.argsort(phase)
    phase = phase[sorted_indices]
    flux = flux[sorted_indices]

    num_in_bin = np.histogram(flux, num_bins)[0]
    ntot = np.sum(num_in_bin)
    # print("Num in bin,ntot: ",num_in_bin,ntot)
    frac_bin = num_in_bin / ntot + 10 ** (
        -10
    )  # Note: 10**(-10) takes care of zero bin values

    entropy = np.sum((frac_bin) * np.log2(frac_bin))

    min_entropy = np.log2(1 / num_bins)

    beta = 1.0  # dummy value
    binsx = num_bins
    sigma_expect = (
        beta * (1 / ntot) * np.sqrt((binsx - 1) / 2 + (binsx**2 - 1) / (6 * ntot))
    )  # from Hutcheson
    entropy_expect = (
        np.log2(binsx)
        - (binsx - 1) / (2 * ntot)
        - (binsx - 1) * (binsx + 1) / (12 * ntot**2)
    )  # new version, uses Hutcheson thesis

    return entropy, min_entropy, -entropy_expect, sigma_expect  # Note minus sign
