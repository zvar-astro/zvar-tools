from typing import Tuple, List
import os

import h5py
import numpy as np
import paramiko

from zvar_utils.candidate import VariabilityCandidate

# from zvar_utils.stats import median_abs_deviation
from zvar_utils.files import get_files_list, get_files
from zvar_utils.enums import FILTERS, FILTER2IDX


def process_curve(
    time: list, flux: list, flux_err: list, filter: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time *= 86400  # Convert from BJD to BJS
    time += 15  # midpoint correct ZTF timestamps
    time -= np.min(time)  # Subtract off the zero point (ephemeris may be added later)

    # Set nan values to zero
    # flux[np.isnan(flux)] = 0
    # flux_err[flux == 0] = np.inf

    # valid = np.where(np.abs(flux) < 1.483 * median_abs_deviation(flux))

    # time = time[valid]
    # flux = flux[valid]
    # flux_err = flux_err[valid]

    # subtract off the mean (of non-NaN values)

    flux -= np.mean(flux[np.isnan(flux) == False])  # noqa E712

    return time, flux, flux_err, filter


def remove_deep_drilling(
    time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def remove_nans(
    time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isnan(flux)
    return (
        time[~mask],
        flux[~mask],
        flux_err[~mask],
        filter[~mask],
    )


def freq_grid(
    t: np.ndarray, fmin: float = None, fmax: float = None, oversample: int = 3
) -> np.ndarray:
    trange = max(t) - min(t)
    texp = np.nanmin(np.diff(np.sort(t)))
    fres = 1.0 / trange / oversample
    if fmax is None:
        fmax = 0.5 / texp
    if fmin is None:
        fmin = fres
    fgrid = np.arange(fmin, fmax, fres)
    return fgrid


def flag_terrestrial_freq(frequencies: np.ndarray) -> np.ndarray:
    """
    Function to identify and flag terrestrial frequencies
    from an array of frequency values.

    frequencies = numpy array of frequencies in Hz
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


def read_lightcurves(ids_per_files, local_path):
    all_ids = set()
    all_files = set()
    for file, ids in ids_per_files.items():
        all_ids.update(ids)
        all_files.add(file)

    all_photometry = {id: [] for id in all_ids}

    for file in all_files:
        ids = ids_per_files[file]
        file_path = f"{local_path}/{file}"

        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist")
            continue
        try:
            with h5py.File(file_path, "r") as f:
                data = f["data"]
                sources = data["sources"]

                sources_data = data["sourcedata"]
                exposures = data["exposures"]

                for id in ids:
                    idx = np.where(sources["gaia_id"] == id)[0]
                    if not idx:
                        continue
                    idx = idx[0]

                    rows = exposures.shape[0]
                    start, end = idx * rows, (idx + 1) * rows

                    raw_photometry = sources_data[start:end]
                    times = exposures["bjd"]
                    # photometry is a list of tuples: flux, flux_err, flag (where flag = 1 means flux is NaN)
                    # to which we want to add a fourth element: the filter
                    # we get the filter simply by looking at the last character of the file name (before the extension)
                    filter = file.split(".")[-2][-1]
                    if filter not in FILTERS:
                        print(f"Filter {filter} not in {FILTERS}")
                        continue

                    photometry = []
                    # also just make the flux = NaN where flag = 1
                    for i in range(rows):
                        if int(raw_photometry[i][2]) == 1:
                            photometry.append(
                                [
                                    times[i],
                                    np.nan,
                                    float(raw_photometry[i][1]),
                                    FILTER2IDX[filter],
                                ]
                            )
                        else:
                            photometry.append(
                                [
                                    times[i],
                                    float(raw_photometry[i][0]),
                                    float(raw_photometry[i][1]),
                                    FILTER2IDX[filter],
                                ]
                            )

                    # process the photometry with the process_curve function
                    # and the remove_deep_drilling function
                    time, flux, flux_err = (
                        np.array([x[0] for x in photometry]),
                        np.array([x[1] for x in photometry]),
                        np.array([x[2] for x in photometry]),
                    )
                    # print(
                    #     f"NaN: {np.sum(np.isnan(flux))}, >0: {np.sum(flux > 0)}, <0: {np.sum(flux < 0)}, =0: {np.sum(flux == 0)}"
                    # )
                    # time, flux, flux_err = process_curve(time, flux, flux_err)
                    # time, flux, flux_err = remove_deep_drilling(time, flux, flux_err)

                    # put the photometry back together
                    photometry = list(
                        zip(time, flux, flux_err, [x[3] for x in photometry])
                    )

                    all_photometry[id] = all_photometry[id] + photometry
        except FileNotFoundError:
            print(f"File {file} not found")
            continue
        except OSError:
            print(f"Error reading file {file}")
            continue
        except KeyError as e:
            print(f"Key not found in file {file}: {e}")
            continue
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

    for id, photometry in all_photometry.items():
        # sort the photometry by time and reshape
        all_photometry[id] = np.array(sorted(photometry, key=lambda x: x[0])).T
        # clean the photometry
        all_photometry[id] = process_curve(*all_photometry[id])
        all_photometry[id] = remove_deep_drilling(*all_photometry[id])

        # instead of a tuple of arrays, we want an array of shape (4, n_photometry)
        # so that when accessing the photometry for a given id, we can do:
        # time, flux, flux_err, filter = all_photometry[id]
        all_photometry[id] = np.array(all_photometry[id], order="C")

    return all_photometry


def retrieve_objs_lightcurve(
    objs: List[VariabilityCandidate],
    path_lc,
    ssh_client: paramiko.SSHClient = None,
    remote_path_lc=None,
    bands=FILTERS,
    limit_fields=None,  # limit to specific fields
):
    # get files for each object
    ids_per_file = {}
    for candidate in objs:
        psid, ra, dec = candidate.id, candidate.ra, candidate.dec
        files = get_files_list(ra, dec, "data", bands, limit_fields=limit_fields)
        for file in files:
            if file not in ids_per_file:
                ids_per_file[file] = set()
            ids_per_file[file].add(psid)

    # download files, and remove files that are not available
    available_files = get_files(
        list(ids_per_file.keys()), path_lc, ssh_client, remote_path_lc
    )
    for file in list(ids_per_file.keys()):
        if file not in available_files:
            del ids_per_file[file]

    lightcurves_per_obj = read_lightcurves(ids_per_file, path_lc)

    return lightcurves_per_obj
