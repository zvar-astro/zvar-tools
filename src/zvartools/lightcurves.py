from typing import Tuple
import os

import h5py
import numpy as np

from zvartools.enums import FILTERS


def remove_deep_drilling(
    time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove deep drilling data from a light curve.

    Parameters
    ----------
    time : np.ndarray
        List of timestamps
    flux : np.ndarray
        List of flux values
    flux_err : np.ndarray
        List of flux errors
    filter : np.ndarray
        List of filters

    Returns
    -------
    time : np.ndarray
        Processed timestamps
    flux : np.ndarray
        Processed flux values
    flux_err : np.ndarray
        Processed flux errors
    filter : np.ndarray
        Processed filters
    """
    if len(time) == 0:
        return time, flux, flux_err, filter
    dt = 40.0  # seconds
    znew_times, znew_flux, znew_flux_err, znew_filter = [], [], [], []

    unique_days = np.unique(np.floor(time / 86400.0))
    for ud in unique_days:
        mask = (time > ud * 86400.0) & (time < (ud + 1.0) * 86400.0)
        zday_times = time[mask]
        zday_flux = flux[mask]
        zday_flux_err = flux_err[mask]
        Nz = len(zday_times)

        # If more than 10 exposures in a night, check time sampling
        if Nz > 10:
            tsec = zday_times
            zdiff = (tsec[1:] - tsec[:-1]) < dt

            # Handle some edge cases
            if zdiff[0]:
                zdiff = np.insert(zdiff, 0, True)
            else:
                zdiff = np.insert(zdiff, 0, False)
            for j in range(1, len(zdiff)):
                if zdiff[j]:
                    zdiff[j - 1] = True

            # Only keep exposures with sampling > dt
            znew_times.append(zday_times[~zdiff])
            znew_flux.append(zday_flux[~zdiff])
            znew_flux_err.append(zday_flux_err[~zdiff])
            znew_filter.append(filter[mask][~zdiff])
        else:
            znew_times.append(zday_times)
            znew_flux.append(zday_flux)
            znew_flux_err.append(zday_flux_err)
            znew_filter.append(filter[mask])

    znew_times = np.concatenate(znew_times)
    znew_flux = np.concatenate(znew_flux)
    znew_flux_err = np.concatenate(znew_flux_err)
    znew_filter = np.concatenate(znew_filter)

    return znew_times, znew_flux, znew_flux_err, znew_filter


def cleanup_lightcurve(
    time: list, flux: list, flux_err: list, filter: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a light curve by removing NaNs, outliers, and deep drilling fields.

    Parameters
    ----------
    time : list
        List of timestamps
    flux : list
        List of flux values
    flux_err : list
        List of flux errors
    filter : list
        List of filters

    Returns
    -------
    time : np.ndarray
        Processed timestamps
    flux : np.ndarray
        Processed flux values
    flux_err : np.ndarray
        Processed flux errors
    filter : np.ndarray
        Processed filters
    """
    time *= 86400  # Convert from BJD to BJS
    time += 15  # midpoint correct ZTF timestamps
    time -= np.min(time)  # Subtract off the zero point

    mad = np.nanmedian(np.abs(flux - np.nanmedian(flux)))
    valid = np.where((np.abs(flux) < 1.483 * mad) | (np.isnan(flux)))
    time = time[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]
    filter = filter[valid]

    # subtract off the mean (of non-NaN values)
    # flux -= np.mean(flux[np.isnan(flux) == False])  # noqa E712

    # Remove deep drilling fields
    time, flux, flux_err, filter = remove_deep_drilling(time, flux, flux_err, filter)

    return time, flux, flux_err, filter


def freq_grid(
    t: np.ndarray, fmin: float = None, fmax: float = None, oversample: int = 3
) -> np.ndarray:
    """
    Generate a frequency grid for a given time series.

    Parameters
    ----------
    t : np.ndarray
        List of timestamps
    fmin : float, optional
        Minimum frequency, by default None
    fmax : float, optional
        Maximum frequency, by default None
    oversample : int, optional
        Oversampling factor, by default 3

    Returns
    -------
    np.ndarray
        Frequency grid
    """
    trange = max(t) - min(t)
    texp = np.nanmin(np.diff(np.sort(t)))
    fres = 1.0 / trange / oversample
    if fmax is None:
        fmax = 0.5 / texp
    if fmin is None:
        fmin = fres
    fgrid = np.arange(fmin, fmax, fres)
    return fgrid


def remove_nondetections(
    time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove non-detections from a light curve.

    Parameters
    ----------
    time : list
        List of timestamps
    flux : list
        List of flux values
    flux_err : list
        List of flux errors
    filter : list
        List of filters

    Returns
    -------
    time : np.ndarray
        Processed timestamps
    flux : np.ndarray
        Processed flux values
    flux_err : np.ndarray
        Processed flux errors
    filter : np.ndarray
        Processed filters
    """

    mask = np.isnan(flux)
    return (
        time[~mask],
        flux[~mask],
        flux_err[~mask],
        filter[~mask],
    )


def flag_terrestrial_freq(frequencies: np.ndarray) -> np.ndarray:
    """
    Function to identify and flag terrestrial frequencies
    from an array of frequency values.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies in Hz

    Returns
    -------
    np.ndarray
        Array of flags indicating whether each frequency is terrestrial
    """
    # Define some standard frequencies in terms of Hz
    f_year = 0.002737803 / 86400  # Sidereal year (365.256363 days)
    f_lunar = 0.036600996 / 86400  # Lunar sidereal month (27.321661 days)
    f_day = 1.00273719 / 86400  # Sidereal day (23.9344696 hours)

    # List of frequency/width pairs indicating the
    # regions to be excluded from the frequencies array
    ef = np.array(
        [
            (f_year, 0.00025 / 86400),  # 1 year
            (f_year / 2, 0.000125 / 86400),  # 2 year
            (f_year / 3, 0.0000835 / 86400),  # 3 year
            (f_year / 4, 0.0000625 / 86400),  # 4 year
            (f_year / 5, 0.0000500 / 86400),  # 5 year
            (f_year * 2, 0.00025 / 86400),  # 1/2 year
            (f_year * 3, 0.00025 / 86400),  # 1/3 year
            (f_year * 4, 0.00025 / 86400),  # 1/4 year
            (f_lunar, 0.005 / 86400),  # 1 lunar month
            (f_day, 0.05 / 86400),  # 1 day
            (f_day / 2, 0.02 / 86400),  # 2 day
            (f_day / 3, 0.01 / 86400),  # 3 day
            (f_day * 2, 0.05 / 86400),  # 1/2 day
            (f_day * 3, 0.05 / 86400),  # 1/3 day
            (f_day * 3 / 2, 0.02 / 86400),  # 2/3 day
            (f_day * 4 / 3, 0.02 / 86400),  # 3/4 day
            (f_day * 5 / 2, 0.02 / 86400),  # 2/5 day
            (f_day * 7 / 2, 0.02 / 86400),  # 2/7 day
            (f_day * 4, 0.05 / 86400),  # 1/4 day
            (f_day * 5, 0.05 / 86400),  # 1/5 day
            (f_day * 6, 0.05 / 86400),  # 1/6 day
            (f_day * 7, 0.05 / 86400),  # 1/7 day
            (f_day * 8, 0.05 / 86400),  # 1/8 day
            (f_day * 9, 0.05 / 86400),  # 1/9 day
            (f_day * 10, 0.05 / 86400),  # 1/10 day
        ]
    )

    # Create array to store keep/discard indices
    keep_freq = np.ones(len(frequencies), dtype=int)

    # Vectorized operation to flag frequencies
    for center, width in ef:
        mask = (frequencies > center - width) & (frequencies < center + width)
        keep_freq[mask] = 0

    return keep_freq


def minify_lightcurve(
    field: int,
    ccd: int,
    quad: int,
    bands: list,
    psids: list,
    path_lc: str,
    out_dir_lc: str = None,
    delete_existing: bool = False,
):
    """
    Minify a lightcurve file (matchfile), only keeping the photometry and sources for a given list.

    Parameters
    ----------
    field : int
        Field number
    ccd : int
        CCD number
    quad : int
        Quadrant number
    bands : list
        List of bands
    psids : list
        List of PS1 IDs
    path_lc : str
        Path to the lightcurve file
    out_dir_lc : str, optional
        Output directory, by default None
    delete_existing : bool, optional
        Whether to delete existing files, by default False
    """
    if len(psids) == 0:
        raise ValueError("psids must be provided")
    if path_lc is None:
        raise ValueError("path_lc must be provided")
    if field is None or ccd is None or quad is None:
        raise ValueError("field, ccd, and quad must be provided")
    if out_dir_lc is None:
        raise ValueError("out_dir must be provided")

    if not os.path.isdir(path_lc):
        raise ValueError(f"path_lc {path_lc} does not exist")

    if not os.path.isdir(out_dir_lc):
        os.makedirs(out_dir_lc)

    if bands is None:
        bands = FILTERS

    psids = np.array(psids, dtype=np.uint64)
    # deduplicate if necessary
    psids = np.unique(psids)

    for band in bands:
        path = f"{path_lc}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_z{band}.h5"
        # check if the file exists
        if not os.path.isfile(path):
            print(f"File {path} does not exist")
            continue

        with h5py.File(path, "r") as f:
            data = f["data"]
            sources = data["sources"]
            sourcedata = data["sourcedata"]
            exposures = data["exposures"]

            # check for each psid one by one
            for psid in psids:
                if psid not in sources["gaia_id"]:
                    print(f"Source {psid} not found in {field}, {ccd}, {quad}, {band}")
                    continue

            idxs = np.where(np.isin(sources["gaia_id"], psids))[0]

            new_sources = sources[idxs]
            new_sourcedata = []
            for idx in idxs:
                start, end = idx * exposures.shape[0], (idx + 1) * exposures.shape[0]
                new_sourcedata.extend(sourcedata[start:end])
            new_sourcedata = np.array(new_sourcedata, order="C")

            new_exposures = exposures[
                :
            ]  # should be identical to the original exposures

            assert new_sources.shape[0] * exposures.shape[0] == new_sourcedata.shape[0]
            assert new_sources.shape[0] <= len(psids)
            assert new_exposures.shape[0] == exposures.shape[0]

            if new_sources.shape[0] == 0:
                print(f"No sources found for {field}, {ccd}, {quad}, {band}")
                continue

            if new_sources.shape[0] != len(psids):
                print(
                    f"Only {new_sources.shape[0]} sources found for {field}, {ccd}, {quad}, {band} instead of {len(psids)}"
                )

        del sources, sourcedata, exposures, data

        new_path = (
            f"{out_dir_lc}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_z{band}.h5"
        )
        if not os.path.isdir(f"{out_dir_lc}/{field:04d}"):
            os.makedirs(f"{out_dir_lc}/{field:04d}")

        if os.path.isfile(new_path) and delete_existing:
            os.remove(new_path)

        with h5py.File(new_path, "w") as f:
            data = f.create_group("data")
            data.create_dataset("sources", data=new_sources)
            data.create_dataset("sourcedata", data=new_sourcedata)
            data.create_dataset("exposures", data=new_exposures)

    return


def adu_to_jansky(flux_adu: float, flux_err_adu: float, exptime: float, mzpsci: float):
    """
    Convert ZTF flux measurements from ADU to Jansky, accounting for the
    instrument's electron/ADU gain and exposure time.

    The conversion follows the physical process of measurement:
    1. ADU -> electrons (using ZTF's gain of 6.2 e-/ADU)
    2. electrons -> photon count rate (dividing by exposure time)
    3. count rate -> flux in Jansky (using zero point calibration)

    Parameters:
    -----------
    flux_adu : float or array
        Flux in ADU units from difference imaging
    flux_err_adu : float or array
        Flux uncertainty in ADU units
    exptime : float
        Exposure time in seconds
    mzpsci : float
        Zero-point magnitude from the image header

    Returns:
    --------
    flux_jy : float or array
        Calibrated flux in Jansky
    flux_err_jy : float or array
        Flux uncertainty in Jansky
    """
    # ZTF-specific gain value (electrons per ADU)
    gain = 6.2

    # Step 1: Convert ADU to electrons
    # This recovers the actual number of electrons generated by incoming photons
    electrons = flux_adu * gain
    electrons_err = flux_err_adu * gain

    # Step 2: Convert to electron rate (electrons per second)
    # This normalizes our measurement by time, giving us a rate instead of a total
    electron_rate = electrons / exptime
    electron_rate_err = electrons_err / exptime

    # Step 3: Convert to Jansky using the zero point calibration
    # The zero point magnitude relates our instrumental measurements
    # to the standard flux unit of Jansky
    F0 = 3631.0  # Standard zero point flux in Jansky
    flux_scale = 10 ** (-mzpsci / 2.5)

    flux_jy = electron_rate * flux_scale * F0
    flux_err_jy = electron_rate_err * flux_scale * F0

    return flux_jy, flux_err_jy
