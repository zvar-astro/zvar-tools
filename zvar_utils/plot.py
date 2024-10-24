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

MARKER_STYLES = {5: ("s", 20), 10: ("o", 20), 20: ("*", 35)}


def plot_hr_diagram(
    candidate_list: List[VariabilityCandidate],
    band: str,
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = False,
):
    if not isinstance(candidate_list, list):
        raise ValueError("Candidates must be provided as a list")
    if not all(
        isinstance(candidate, VariabilityCandidate) for candidate in candidate_list
    ):
        raise ValueError("Candidates must be of type Candidate")
    if band not in ALLOWED_BANDS:
        raise ValueError(f"Invalid band specified. Allowed bands are {ALLOWED_BANDS}")
    if output_path is not None and not isinstance(output_path, (str, os.PathLike)):
        raise ValueError("Output path must be a string or a PathLike object")
    if not isinstance(show_plot, bool):
        raise ValueError("Show plot must be a boolean")

    periods = [
        1 / candidate.freq
        for candidate in candidate_list
        if candidate.gaia_BP_RP is not None and candidate.gaia_MG is not None
    ]
    bp_rp = [
        candidate.gaia_BP_RP
        for candidate in candidate_list
        if candidate.gaia_BP_RP is not None and candidate.gaia_MG is not None
    ]
    mg = [
        candidate.gaia_MG
        for candidate in candidate_list
        if candidate.gaia_BP_RP is not None and candidate.gaia_MG is not None
    ]
    best_M = [
        candidate.best_M
        for candidate in candidate_list
        if candidate.gaia_BP_RP is not None and candidate.gaia_MG is not None
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
    ax.set_title(
        f"Gaia HR Diagram with Variability Candidates Sloan {band} band", fontsize=16
    )
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


def plot_lightcurve(
    candidate: VariabilityCandidate,
    time,
    flux,
    fluxerr,
    figsize: tuple = (9, 8),
    output_path: str = None,
    show_plot: bool = False,
):
    times_days = time / 86400
    phase = (times_days * candidate.freq) % 2

    _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.errorbar(phase, flux, yerr=fluxerr, fmt="o", color="black", ms=3)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Flux")
    ax.set_title(
        f"ID: {candidate.id}, RA: {candidate.ra}, DEC: {candidate.dec}\nPeriod: {24/candidate.freq:.4f} hours, Best M: {candidate.best_M}"
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
    time: np.ndarray,
    flux: np.ndarray,
    fluxerr: np.ndarray,
    pgram: np.ndarray,
    best_period: float,
    show_plot: bool = True,
    figsize: tuple = (12, 14),
):
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
        f"ID: {candidate.id}, RA: {candidate.ra}, DEC: {candidate.dec}\nBest M: {candidate.best_M}",
        fontsize=16,
    )

    # add spacing between the subplots
    plt.tight_layout()

    if show_plot:
        plt.show()
