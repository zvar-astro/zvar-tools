import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.utils.data import download_file
from matplotlib.colors import LogNorm

from zvartools.candidate import VariabilityCandidate
from zvartools.enums import ALLOWED_BANDS
from zvartools.lightcurves import freq_grid, remove_nondetections

BAND_TO_COLOR = {1: "green", 2: "red", 3: "orange"}
BAND_IDX_TO_NAME = {1: "g", 2: "r", 3: "i"}
BAND_NAME_TO_IDX = {"g": 1, "r": 2, "i": 3}
MARKER_STYLES = {5: ("s", 20), 10: ("o", 20), 20: ("*", 35)}


def get_gaia_data():
    url = "https://github.com/zvar-astro/zvar-tools/raw/refs/heads/main/data/hrd_query_edr3_200pc.fits"
    filename = download_file(url, cache=True)
    return Table.read(filename).to_pandas()


def plot_gaia_cmd(
    candidate_list: List[VariabilityCandidate],
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = True,
    ax=None,
    title_size: int = 16,
):
    """
    Plot a Gaia HR diagram with variability candidates

    Parameters
    ----------
    candidate_list : List[VariabilityCandidate]
        List of variability candidates
    figsize : tuple, optional
        Figure size, by default (9, 8)
    output_path : str, optional
        Path to save the plot, by default None
    show_plot : bool, optional
        Whether to show the plot, by default True
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    title_size : int, optional
        Font size for the title, by default 16

    Raises
    ------
    ValueError
        If candidates are not provided as a list
        If candidates are not of type Candidate
        If output path is not a string or PathLike object
        If show plot is not a boolean
        If no candidates found with valid BP-RP and MG values
    """
    if not isinstance(candidate_list, list):
        raise ValueError("Candidates must be provided as a list")
    if not all(
        isinstance(candidate, VariabilityCandidate) for candidate in candidate_list
    ):
        raise ValueError("Candidates must be of type Candidate")
    if output_path is not None and not isinstance(output_path, (str, os.PathLike)):
        raise ValueError("Output path must be a string or a PathLike object")
    if not isinstance(show_plot, bool):
        raise ValueError("Show plot must be a boolean")

    valid_candidates_idx = [
        i
        for i, candidate in enumerate(candidate_list)
        if (
            candidate.gaia.BP_RP is not None
            and candidate.gaia.MG is not None
            and not np.isnan(candidate.gaia.BP_RP)
            and not np.isnan(candidate.gaia.MG)
        )
    ]
    if len(valid_candidates_idx) == 0:
        raise ValueError("No candidates found with valid BP-RP and MG values")

    periods = [1 / candidate_list[i].freq for i in valid_candidates_idx]
    best_M = [candidate_list[i].best_M for i in valid_candidates_idx]
    bp_rp = [candidate_list[i].gaia.BP_RP for i in valid_candidates_idx]
    mg = [candidate_list[i].gaia.MG for i in valid_candidates_idx]

    gaia_sample = get_gaia_data()
    gaia_bprp = (
        gaia_sample.phot_bp_mean_mag.values - gaia_sample.phot_rp_mean_mag.values
    )
    gaia_gmag = gaia_sample.phot_g_mean_mag.values + 5.0 * np.log10(
        gaia_sample.parallax.values / 100
    )

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.hist2d(
        gaia_bprp,
        gaia_gmag,
        cmap="gray",
        cmin=3,
        cmax=30,
        bins=(700, 380),
        alpha=0.8,
    )
    ax.hist2d(
        gaia_bprp,
        gaia_gmag,
        cmap="gray_r",
        cmin=30,
        cmax=1000,
        bins=(700, 380),
        alpha=0.8,
    )

    # Scatter plot for candidates colored by period with logarithmic scale
    for m_value in MARKER_STYLES:
        indices = [i for i, m in enumerate(best_M) if m == m_value]
        if not indices:
            continue
        marker, size = MARKER_STYLES[m_value]
        sc = ax.scatter(
            [bp_rp[i] for i in indices],
            [mg[i] for i in indices],
            c=[periods[i] for i in indices],
            cmap="plasma",
            s=size,
            edgecolor="none",
            norm=LogNorm(),
            marker=marker,
            label=f"best_M = {m_value}",
        )

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Period (log days)", fontsize=title_size)

    ax.set_title("Gaia HR Diagram with Variability Candidates", fontsize=title_size)
    ax.set_ylabel("$M_G$ (mag)", fontsize=title_size)
    ax.set_xlabel("$BP-RP$ (mag)", fontsize=title_size)
    ax.set_ylim(17.0, -2.0)
    ax.set_yticks([0, 5, 10, 15])
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=14)
    ax.set_axisbelow(True)
    ax.grid(c="silver", ls=":", lw=1)
    ax.legend()
    plt.tight_layout()

    # Save the plot
    if output_path:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()


def plot_2mass_cmd(
    candidate_list: List[VariabilityCandidate],
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = True,
    ax=None,
    title_size: int = 16,
):
    if not isinstance(candidate_list, list):
        raise ValueError("Candidates must be provided as a list")
    if not all(
        isinstance(candidate, VariabilityCandidate) for candidate in candidate_list
    ):
        raise ValueError("Candidates must be of type Candidate")
    if output_path is not None and not isinstance(output_path, (str, os.PathLike)):
        raise ValueError("Output path must be a string or a PathLike object")
    if not isinstance(show_plot, bool):
        raise ValueError("Show plot must be a boolean")

    valid_candidates_idx = [
        i
        for i, candidate in enumerate(candidate_list)
        if (
            candidate.twomass is not None
            and candidate.twomass.j is not None
            and candidate.twomass.k is not None
            and candidate.gaia is not None
            and candidate.gaia.parallax is not None
            and not np.isnan(candidate.twomass.j)
            and not np.isnan(candidate.twomass.k)
            and not np.isnan(candidate.gaia.parallax)
        )
    ]

    if len(valid_candidates_idx) == 0:
        raise ValueError("No candidates found with valid 2MASS and Gaia values")

    periods = [1 / candidate_list[i].freq for i in valid_candidates_idx]
    best_M = [candidate_list[i].best_M for i in valid_candidates_idx]
    j_k = [
        candidate_list[i].twomass.j - candidate_list[i].twomass.k
        for i in valid_candidates_idx
    ]
    j_abs = [
        # Use gaia parallax to calculate the absolute magnitude in the J band
        candidate_list[i].twomass.j
        - 5 * np.log10(1e3 / candidate_list[i].gaia.parallax)
        + 5
        for i in valid_candidates_idx
    ]

    if len(periods) == 0:
        raise ValueError("No candidates found with valid J-K values")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    for m_value in MARKER_STYLES:
        indices = [i for i, m in enumerate(best_M) if m == m_value]
        if not indices:
            continue
        marker, size = MARKER_STYLES[m_value]
        sc = ax.scatter(
            [j_k[i] for i in indices],
            [j_abs[i] for i in indices],
            c=[periods[i] for i in indices],
            cmap="plasma",
            s=size,
            edgecolor="none",
            norm=LogNorm(),
            marker=marker,
            label=f"best_M = {m_value}",
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Period (log days)", fontsize=title_size)

    ax.set_title("2MASS Color-Color Diagram", fontsize=title_size)
    ax.set_ylabel("$M_J$ (mag)", fontsize=title_size)
    ax.set_xlabel("$J-K$ (mag)", fontsize=title_size)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=14)
    ax.set_axisbelow(True)
    ax.grid(c="silver", ls=":", lw=1)
    ax.legend()
    plt.tight_layout()

    # Save the plot
    if output_path:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()


def plot_2mass_color(
    candidate_list: List[VariabilityCandidate],
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = True,
    ax=None,
    title_size: int = 16,
):
    if not isinstance(candidate_list, list):
        raise ValueError("Candidates must be provided as a list")
    if not all(
        isinstance(candidate, VariabilityCandidate) for candidate in candidate_list
    ):
        raise ValueError("Candidates must be of type Candidate")
    if output_path is not None and not isinstance(output_path, (str, os.PathLike)):
        raise ValueError("Output path must be a string or a PathLike object")
    if not isinstance(show_plot, bool):
        raise ValueError("Show plot must be a boolean")

    valid_candidates_idx = [
        i
        for i, candidate in enumerate(candidate_list)
        if (
            candidate.twomass is not None
            and candidate.twomass.j is not None
            and candidate.twomass.k is not None
            and not np.isnan(candidate.twomass.j)
            and not np.isnan(candidate.twomass.k)
        )
    ]

    if len(valid_candidates_idx) == 0:
        raise ValueError("No candidates found with valid J-K values")

    periods = [1 / candidate_list[i].freq for i in valid_candidates_idx]
    best_M = [candidate_list[i].best_M for i in valid_candidates_idx]
    j_k = [
        candidate_list[i].twomass.j - candidate_list[i].twomass.k
        for i in valid_candidates_idx
    ]
    j = [candidate_list[i].twomass.j for i in valid_candidates_idx]

    if len(periods) == 0:
        raise ValueError("No candidates found with valid J-K values")

    # Plot candidates
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # Scatter plot for candidates colored by period with logarithmic scale
    for m_value in MARKER_STYLES:
        indices = [i for i, m in enumerate(best_M) if m == m_value]
        if not indices:
            continue
        marker, size = MARKER_STYLES[m_value]
        sc = ax.scatter(
            [j_k[i] for i in indices],
            [j[i] for i in indices],
            c=[periods[i] for i in indices],
            cmap="plasma",
            s=size,
            edgecolor="none",
            norm=LogNorm(),
            marker=marker,
            label=f"best_M = {m_value}",
        )

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Period (log days)", fontsize=title_size)

    # Set plot appearances
    ax.set_title("2MASS Color-Color Diagram", fontsize=title_size)
    ax.set_ylabel("$J$ (mag)", fontsize=title_size)
    ax.set_xlabel("$J-K$ (mag)", fontsize=title_size)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=14)
    ax.set_axisbelow(True)
    ax.grid(c="silver", ls=":", lw=1)
    ax.legend()
    plt.tight_layout()

    # Save the plot
    if output_path:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()


def plot_folded_lightcurve(
    candidate: VariabilityCandidate,
    photometry: List[np.ndarray],
    bands: List[str] = ["g", "r"],
    period: float = None,
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = True,
    ax=None,
    marker_size: int = 4,
    title_size: int = 16,
    bins: Union[int, None] = None,
    bin_method: str = "mean",
):
    """
    Plot the folded lightcurve of a candidate

    Parameters
    ----------
    candidate : VariabilityCandidate
        Variability candidate
    photometry : List[np.ndarray]
        Photometry data
    bands : List[str], optional
        Bands to plot, by default ["g", "r"]
    period : float, optional
        Period of the candidate, by default None
    figsize : tuple, optional
        Figure size, by default (9, 8)
    output_path : str, optional
        Path to save the plot, by default None
    show_plot : bool, optional
        Whether to show the plot, by default True
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    marker_size : int, optional
        Marker size for the plot, by default 4
    title_size : int, optional
        Font size for the title, by default 16
    bins : Union[int, None], optional
        Number of bins to use for binning the data, by default None
    bin_method : str, optional
        Method to use for binning the data, by default "mean"

    Raises
    ------
    ValueError
        If an invalid band is specified
        If no data points found in the specified bands
        If no valid data points found to plot_folded_lightcurve
    """
    if period is None:
        period = 1 / candidate.freq
    bands = list({band.lower() for band in bands})
    if not all(band in ALLOWED_BANDS for band in bands):
        raise ValueError(f"Invalid band specified. Allowed bands are {ALLOWED_BANDS}")
    bands = [BAND_NAME_TO_IDX[band] for band in bands]

    # remove data points with flux = NaN
    time, flux, fluxerr, filters = remove_nondetections(
        photometry[0], photometry[1], photometry[2], photometry[3]
    )

    # only keep data points in the band of interest, so where filters is in the list of bands
    mask = np.array([f in bands for f in filters])

    try:
        time = time[mask]
        flux = flux[mask]
        fluxerr = fluxerr[mask]
        filters = filters[mask]
    except IndexError:
        raise ValueError(
            f"No data points found in the specified bands for {int(candidate.psid)}"
        )

    if len(time) == 0:
        raise ValueError(
            f"No valid data points found to plot_folded_lightcurve for {int(candidate.psid)}"
        )
        # from bands, remove bands for which we have no data points
    bands = [band for band in bands if np.any(filters == band)]

    phase = (time / (period * 86400)) % 2  # period is converted from days to seconds

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if bins is not None:
        # Create bins and calculate which bin each phase belongs to
        bin_indices = np.digitize(phase, np.linspace(0, 2, bins + 1)) - 1

        # we bin the points per band
        binned_flux = np.zeros((len(bands), bins))
        binned_fluxerr = np.zeros((len(bands), bins))
        for i, band in enumerate(bands):
            for j in range(bins):
                mask = (filters == band) & (bin_indices == j)
                # if there are no points in the bin, we set the flux to -inf
                if not np.any(mask):
                    binned_flux[i, j] = -np.inf
                    binned_fluxerr[i, j] = -np.inf
                    continue
                if bin_method == "mean":
                    binned_flux[i, j] = np.mean(flux[mask])
                    binned_fluxerr[i, j] = np.sqrt(np.sum(fluxerr[mask] ** 2)) / np.sum(
                        mask
                    )
                elif bin_method == "median":
                    binned_flux[i, j] = np.median(flux[mask])
                    binned_fluxerr[i, j] = np.median(fluxerr[mask])
                else:
                    raise ValueError(
                        "Invalid bin_method. Must be either 'mean' or 'median'"
                    )

        # # remove bins with no data
        mask = np.all(binned_flux != -np.inf, axis=0)
        binned_flux = binned_flux[:, mask]
        binned_fluxerr = binned_fluxerr[:, mask]
        x = np.arange(bins)
        x = x[mask]
        x = x / bins

        for i, band in enumerate(bands):
            ax.errorbar(
                x,
                binned_flux[i],
                yerr=binned_fluxerr[i],
                fmt="o",
                label=BAND_IDX_TO_NAME[band],
                color=BAND_TO_COLOR[band],
                ms=marker_size,
            )
    else:
        for band in bands:
            mask = filters == band
            if not np.any(mask):
                continue
            ax.errorbar(
                phase[mask] / 2,
                flux[mask],
                yerr=fluxerr[mask],
                fmt="o",
                label=band,
                color=BAND_TO_COLOR[band],
                ms=marker_size,
            )

    ax.set_xlabel("Phase")
    ax.set_ylabel("Flux")
    ax.set_title(
        f"ID: {int(candidate.psid)}\nRA: {candidate.ra:.4f}, DEC: {candidate.dec:.4f}\nPeriod: {24 / candidate.freq:.4f} hours, Best M: {candidate.best_M}",
        fontsize=title_size,
    )

    if output_path:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()
        plt.close()


def plot_periodicity(
    candidate: VariabilityCandidate,
    photometry: List[np.ndarray],
    pgram: np.ndarray,
    period: float,
    period_unit: str = "hours",
    show_plot: bool = True,
    figsize: tuple = (12, 14),
    bins: Union[int, None] = None,
    bin_method: str = "mean",
):
    """
    Plot the lightcurve, periodogram, and phased lightcurve of a candidate

    Parameters
    ----------
    candidate : VariabilityCandidate
        Variability candidate
    photometry : List[np.ndarray]
        Photometry data
    pgram : np.ndarray
        Periodogram data
    period : float
        Period of the candidate, in hours
    period_unit : str, optional
        Unit of the period if provided, by default "hours"
    show_plot : bool, optional
        Whether to show the plot, by default True
    figsize : tuple, optional
        Figure size, by default (12, 14)
    bins : Union[int, None], optional
        Number of bins to use for binning the data, by default None
    bin_method : str, optional
        Method to use for binning the data, by default "mean"

    Raises
    ------
    ValueError
        If there are multiple filters
        If no valid data points found to plot_periodicity
    """
    if period is None:
        period = 1 / candidate.freq
    elif period_unit == "minutes":
        # convert from minutes to seconds
        period *= 60
    elif period_unit == "hours":
        # convert from hours to seconds
        period *= 60 * 60
    elif period_unit == "days":
        # convert from days to seconds
        period *= 86400
    # for now this method supports single band data, so to prevent any errors we throw an exception if there are multiple filters
    if len(set(photometry[3])) > 1:
        raise ValueError("This method only supports single band data (for now)")

    # remove data points with flux = NaN
    time, flux, fluxerr, _ = remove_nondetections(
        photometry[0], photometry[1], photometry[2], photometry[3]
    )
    if len(time) == 0:
        raise ValueError("No valid data points found to plot_periodicity")

    fgrid = freq_grid(time)

    phase = (time / period) % 2

    # we plot all 3 plots above in a row of 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=figsize)

    # plot the lightcurve
    axs[0].errorbar(time, flux, yerr=fluxerr, fmt="o")
    axs[0].set_title("Lightcurve")
    axs[0].set_xlabel("Time (MJD)")

    # plot the periodogram
    axs[1].plot(fgrid, pgram)
    axs[1].set_xscale("log")
    axs[1].set_title("Periodogram")
    axs[1].set_xlabel("Frequency (Hz)")

    if bins is not None:
        # Create bins and calculate which bin each phase belongs to
        bin_indices = np.digitize(phase, np.linspace(0, 2, bins + 1)) - 1

        # we bin the points
        binned_flux = np.zeros(bins)
        binned_fluxerr = np.zeros(bins)
        for j in range(bins):
            mask = bin_indices == j
            # if there are no points in the bin, we set the flux to -inf
            if not np.any(mask):
                binned_flux[j] = -np.inf
                binned_fluxerr[j] = -np.inf
                continue
            if bin_method == "mean":
                binned_flux[j] = np.mean(flux[mask])
                binned_fluxerr[j] = np.sqrt(np.sum(fluxerr[mask] ** 2)) / np.sum(mask)
            elif bin_method == "median":
                binned_flux[j] = np.median(flux[mask])
                binned_fluxerr[j] = np.median(fluxerr[mask])
            else:
                raise ValueError(
                    "Invalid bin_method. Must be either 'mean' or 'median'"
                )

        # # remove bins with no data
        mask = binned_flux != -np.inf
        flux = binned_flux[mask]
        fluxerr = binned_fluxerr[mask]
        x = np.arange(bins)
        x = x[mask]
        # go back to phase
        x = x / bins
    else:
        x = phase / 2

    # plot the phased lightcurve
    axs[2].errorbar(x, flux, yerr=fluxerr, fmt="o")
    axs[2].set_title(f"Phased Lightcurve (Period: {period / 60 / 60} hours)")
    axs[2].set_xlabel("Phase")

    # add a title to the entire figure
    fig.suptitle(
        f"ID: {int(candidate.psid)}, RA: {candidate.ra}, DEC: {candidate.dec}\nBest M: {candidate.best_M}",
        fontsize=16,
    )

    # add spacing between the subplots
    plt.tight_layout()

    if show_plot:
        plt.show()


def plot_folded_photometry_stat_per_bin(
    candidate: VariabilityCandidate,
    photometry: List[np.ndarray],
    period: float = None,
    num_bins: int = 20,
    method: str = "median",
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = True,
    ax=None,
    line_width: int = 1,
    title_size: int = 16,
):
    """
    Plot the median or mean flux per bin for the folded lightcurve

    Parameters
    ----------
    candidate : VariabilityCandidate
        Variability candidate
    photometry : List[np.ndarray]
        Photometry data
    period : float, optional
        Period of the candidate, by default None
    num_bins : int, optional
        Number of bins to use for binning the data, by default 20
    method : str, optional
        Method to use for calculating the flux per bin, by default "median"
    figsize : tuple, optional
        Figure size, by default (9, 8)
    output_path : str, optional
        Path to save the plot, by default None
    show_plot : bool, optional
        Whether to show the plot, by default True
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    line_width : int, optional
        Line width for the plot, by default 1
    title_size : int, optional
        Font size for the title, by default 16

    Raises
    ------
    ValueError
        If an invalid method is specified
    """
    method = method.lower()
    if method not in ["median", "mean"]:
        raise ValueError("Invalid method. Must be either 'median' or 'mean'")
    if period is None:
        period = 1 / candidate.freq

    # remove data points with flux = NaN
    time, flux, _, _ = remove_nondetections(
        photometry[0], photometry[1], photometry[2], photometry[3]
    )

    phase = (time / (period * 86400)) % 2  # period is converted from days to seconds

    # Create bins and calculate which bin each phase belongs to
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(phase, bin_edges) - 1  # -1 to make it zero-indexed

    method_flux_per_bin = np.zeros(num_bins)
    for i in range(num_bins):
        bin_fluxes = flux[bin_indices == i]

        if len(bin_fluxes) == 0 or np.all(np.isnan(bin_fluxes)):
            method_flux_per_bin[i] = 0
        else:
            if method == "median":
                method_flux_per_bin[i] = np.median(bin_fluxes)
            elif method == "mean":
                method_flux_per_bin[i] = np.mean(bin_fluxes)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.step(
        np.arange(2 * num_bins),
        np.concatenate((method_flux_per_bin, method_flux_per_bin)),
        where="mid",
        linewidth=line_width,
        color="black",
    )

    ax.set_xlabel("Bin Number")
    ax.set_ylabel("Median Flux")
    ax.set_title(
        f"{method.capitalize()} Flux vs Bin Number\nID: {int(candidate.psid)} Period: {24 / candidate.freq:.4f}",
        fontsize=title_size,
    )
    plt.grid(True)

    if output_path:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()
        plt.close()


# just a wrapper with method='median' for plot_photometry_stat_per_bin
def plot_folded_photometry_median_per_bin(
    candidate: VariabilityCandidate,
    photometry: List[np.ndarray],
    period: float = None,
    num_bins: int = 20,
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = True,
    ax=None,
    line_width: int = 1,
    title_size: int = 16,
):
    """
    Plot the median flux per bin for the folded lightcurve

    Parameters
    ----------
    candidate : VariabilityCandidate
        Variability candidate
    photometry : List[np.ndarray]
        Photometry data
    period : float, optional
        Period of the candidate, by default None
    num_bins : int, optional
        Number of bins to use for binning the data, by default 20
    figsize : tuple, optional
        Figure size, by default (9, 8)
    output_path : str, optional
        Path to save the plot, by default None
    show_plot : bool, optional
        Whether to show the plot, by default True
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    line_width : int, optional
        Line width for the plot, by default 1
    title_size : int, optional
        Font size for the title, by default 16

    Raises
    ------
    ValueError
        If an invalid method is specified
    """
    plot_folded_photometry_stat_per_bin(
        candidate,
        photometry,
        period,
        num_bins,
        "median",
        figsize,
        output_path,
        show_plot,
        ax,
        line_width,
        title_size,
    )


# just a wrapper with method='mean' for plot_photometry_stat_per_bin
def plot_folded_photometry_mean_per_bin(
    candidate: VariabilityCandidate,
    photometry: List[np.ndarray],
    period: float = None,
    num_bins: int = 20,
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = True,
    ax=None,
    line_width: int = 1,
    title_size: int = 16,
):
    """
    Plot the mean flux per bin for the folded lightcurve

    Parameters
    ----------
    candidate : VariabilityCandidate
        Variability candidate
    photometry : List[np.ndarray]
        Photometry data
    period : float, optional
        Period of the candidate, by default None
    num_bins : int, optional
        Number of bins to use for binning the data, by default 20
    figsize : tuple, optional
        Figure size, by default (9, 8)
    output_path : str, optional
        Path to save the plot, by default None
    show_plot : bool, optional
        Whether to show the plot, by default True
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    line_width : int, optional
        Line width for the plot, by default 1
    title_size : int, optional
        Font size for the title, by default 16

    Raises
    ------
    ValueError
        If an invalid method is specified
    """
    plot_folded_photometry_stat_per_bin(
        candidate,
        photometry,
        period,
        num_bins,
        "mean",
        figsize,
        output_path,
        show_plot,
        ax,
        line_width,
        title_size,
    )
