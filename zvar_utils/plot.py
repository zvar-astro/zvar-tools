import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from zvar_utils.candidate import VariabilityCandidate
from zvar_utils.enums import ALLOWED_BANDS
from zvar_utils.lightcurves import freq_grid

BAND_TO_COLOR = {1: "green", 2: "red", 3: "orange"}
BAND_IDX_TO_NAME = {1: "g", 2: "r", 3: "i"}
BAND_NAME_TO_IDX = {"g": 1, "r": 2, "i": 3}
MARKER_STYLES = {5: ("s", 20), 10: ("o", 20), 20: ("*", 35)}


def plot_gaia_cmd(
    candidate_list: List[VariabilityCandidate],
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = True,
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

    periods = [
        1 / candidate.freq
        for candidate in candidate_list
        if (
            candidate.gaia.BP_RP is not None
            and candidate.gaia.MG is not None
            and not np.isnan(candidate.gaia.BP_RP)
            and not np.isnan(candidate.gaia.MG)
        )
    ]
    bp_rp = [
        candidate.gaia.BP_RP
        for candidate in candidate_list
        if (
            candidate.gaia.BP_RP is not None
            and candidate.gaia.MG is not None
            and not np.isnan(candidate.gaia.BP_RP)
            and not np.isnan(candidate.gaia.MG)
        )
    ]
    mg = [
        candidate.gaia.MG
        for candidate in candidate_list
        if (
            candidate.gaia.BP_RP is not None
            and candidate.gaia.MG is not None
            and not np.isnan(candidate.gaia.BP_RP)
            and not np.isnan(candidate.gaia.MG)
        )
    ]
    best_M = [
        candidate.best_M
        for candidate in candidate_list
        if (
            candidate.gaia.BP_RP is not None
            and candidate.gaia.MG is not None
            and not np.isnan(candidate.gaia.BP_RP)
            and not np.isnan(candidate.gaia.MG)
        )
    ]

    if len(periods) == 0:
        raise ValueError("No candidates found with valid BP-RP and MG values")

    # Load a Gaia HR diagram
    sample_path = os.path.join(
        os.path.dirname(__file__), "../data/hrd_query_edr3_200pc.fits"
    )
    gaia_sample = Table.read(sample_path).to_pandas()
    gaia_bprp = (
        gaia_sample.phot_bp_mean_mag.values - gaia_sample.phot_rp_mean_mag.values
    )
    gaia_gmag = gaia_sample.phot_g_mean_mag.values + 5.0 * np.log10(
        gaia_sample.parallax.values / 100
    )

    # Overplot candidates on the Gaia HR Diagram
    figa = plt.figure("a", figsize=figsize)
    gs = GridSpec(1, 1)
    ax = figa.add_subplot(gs[0])

    # Plot 2D-Histogram of 200pc sample
    ax.hist2d(
        gaia_bprp,
        gaia_gmag,
        cmap="gray",
        cmin=3,
        cmax=30,  # vmin=1, vmax=30,
        bins=(700, 380),
    )
    ax.hist2d(
        gaia_bprp,
        gaia_gmag,
        cmap="gray_r",
        cmin=30,
        cmax=1000,  # vmin=1, vmax=500,
        bins=(700, 380),
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
    cbar.set_label("Period (log days)", fontsize=16)

    # Set plot appearances
    ax.set_title("Gaia HR Diagram with Variability Candidates", fontsize=16)
    ax.set_ylabel("$M_G$ (mag)", fontsize=16)
    ax.set_xlabel("$BP-RP$ (mag)", fontsize=16)
    # ax.set_xlim(-1.0, 4.5)
    ax.set_ylim(17.0, -2.0)
    ax.set_yticks([0, 5, 10, 15])
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=14)
    ax.set_axisbelow(True)
    ax.grid(c="silver", ls=":", lw=1)

    # Add legend
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


def mask_non_detections(time, flux, fluxerr, filters):
    mask = ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]
    fluxerr = fluxerr[mask]
    filters = filters[mask]

    return time, flux, fluxerr, filters


def plot_folded_lightcurve(
    candidate: VariabilityCandidate,
    photometry: List[np.ndarray],
    bands: List[str] = ["g", "r"],
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = True,
):
    bands = list({band.lower() for band in bands})
    if not all(band in ALLOWED_BANDS for band in bands):
        raise ValueError(f"Invalid band specified. Allowed bands are {ALLOWED_BANDS}")
    bands = [BAND_NAME_TO_IDX[band] for band in bands]

    # remove data points with flux = NaN
    time, flux, fluxerr, filters = mask_non_detections(
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
            f"No data points found in the specified bands for {int(candidate.id)}"
        )

    if len(time) == 0:
        raise ValueError(
            f"No valid data points found to plot_folded_lightcurve for {int(candidate.id)}"
        )

    times_days = time / 86400
    phase = (times_days * candidate.freq) % 2

    _, ax = plt.subplots(1, 1, figsize=figsize)

    for band in bands:
        mask = filters == band
        if not np.any(mask):
            continue
        ax.errorbar(
            phase[mask],
            flux[mask],
            yerr=fluxerr[mask],
            fmt="o",
            label=band,
            color=BAND_TO_COLOR[band],
            ms=4,
        )
    # ax.errorbar(phase, flux, yerr=fluxerr, fmt="o", color="black", ms=3)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Flux")
    ax.set_title(
        f"ID: {int(candidate.id)}, RA: {candidate.ra}, DEC: {candidate.dec}\nPeriod: {24/candidate.freq:.4f} hours, Best M: {candidate.best_M}"
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
    best_period: float,
    show_plot: bool = True,
    figsize: tuple = (12, 14),
):
    # for now this method supports single band data, so to prevent any errors we throw an exception if there are multiple filters
    if len(set(photometry[3])) > 1:
        raise ValueError("This method only supports single band data (for now)")

    # remove data points with flux = NaN
    time, flux, fluxerr, _ = mask_non_detections(
        photometry[0], photometry[1], photometry[2], photometry[3]
    )
    if len(time) == 0:
        raise ValueError("No valid data points found to plot_periodicity")

    fgrid = freq_grid(time)

    phase = (time / best_period) % 2

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

    # plot the phased lightcurve
    axs[2].errorbar(phase, flux, yerr=fluxerr, fmt="o")
    axs[2].set_title(f"Phased Lightcurve (Period: {best_period / 60 / 60} hours)")
    axs[2].set_xlabel("Phase")

    # add a title to the entire figure
    fig.suptitle(
        f"ID: {int(candidate.id)}, RA: {candidate.ra}, DEC: {candidate.dec}\nBest M: {candidate.best_M}",
        fontsize=16,
    )

    # add spacing between the subplots
    plt.tight_layout()

    if show_plot:
        plt.show()
