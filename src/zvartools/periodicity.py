import glob
import os
from typing import Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import h5py
import numpy as np


def load_file(
    file: str,
) -> Tuple[
    str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Load the periodicity (FPW) data from a file

    Parameters
    ----------
    file : str
        Path to the file

    Returns
    -------
    Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        File name and the data from the file
    """
    try:
        psids = np.array([], dtype=np.uint64)
        ratio_valid = np.array([])
        best_freqs = np.array([])
        significances = np.array([])
        ra = np.array([])
        dec = np.array([])

        with h5py.File(file, "r") as dataset:
            psids = np.array(dataset["psids"])
            ratio_valid = np.array(dataset["valid"])
            best_freqs = np.array(dataset["bestFreqs"])
            significances = np.array(dataset["significance"])
            # clean the significances
            np.nan_to_num(significances, nan=0, posinf=0, neginf=0, copy=False)
            ra = np.array(dataset["ra"])
            dec = np.array(dataset["dec"])

        # make it a tuple so we can return it
        return file, (psids, ra, dec, ratio_valid, best_freqs, significances)
    except FileNotFoundError:
        print(f"File {file} not found")
        return file, None
    except OSError:
        print(f"Error reading file {file}")
        return file, None
    except KeyError as e:
        print(f"Key not found in file {file}: {e}")
        return file, None
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        return file, None


def get_ccd_quad_from_filename(file_name: str) -> Tuple[int, int]:
    """
    Get the CCD and quadrant numbers from the file name

    Parameters
    ----------
    file_name : str
        Name of the file

    Returns
    -------
    Tuple[int, int]
        CCD and quadrant numbers
    """
    ccd = int(file_name.split("_")[-3])
    quad = int(file_name.split("_")[-2])
    return ccd, quad


def load_field_periodicity_data_parallel(
    field: int, band: int, path: str
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load the periodicity data for a given field and band, loading the data in parallel

    Parameters
    ----------
    field : int
        Field number
    band : int
        Band number
    path : str
        Path to the data files

    Returns
    -------
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
        PSIDs, RA, Dec, ratio_valid, best_freqs, significances, CCDs, quadrants
    """
    if not isinstance(field, (int, np.integer, str)):
        raise ValueError("Field must be an integer or string")
    if not isinstance(band, (int, np.integer, str)):
        raise ValueError("Band must be an integer or string")
    if path is None:
        raise ValueError("Path must be specified")

    path = os.path.join(
        os.path.abspath(path), f"{str(field).zfill(4)}", f"fpw_*_z{band}.h5"
    )

    files = list(set(glob.glob(path)))

    if not files:
        raise ValueError("No files found")

    # create an array of size len(files) to store the results
    data_per_file = np.empty(len(files), dtype=object)

    # create a pool of workers
    with Pool(min(8, cpu_count() - 2)) as pool:
        with tqdm(total=len(files), desc="Loading files") as pbar:
            for file_name, data in pool.imap_unordered(load_file, files):
                if data is None:
                    continue
                idx = files.index(file_name)
                data_per_file[idx] = data
                pbar.update(1)

    # remove the empty entries
    data_per_file = data_per_file[data_per_file != np.array(None)]

    if len(data_per_file) == 0:
        raise ValueError("No data found in files")

    # now we are left with a numpy array of arrays, i.e. a 3D array
    # reshape to a 2D array

    # create an empty array to store the results
    psids = np.empty(sum(len(data[0]) for data in data_per_file), dtype=np.uint64)
    ra = np.empty(sum(len(data[1]) for data in data_per_file))
    dec = np.empty(sum(len(data[2]) for data in data_per_file))
    ratio_valid = np.empty(sum(len(data[3]) for data in data_per_file))
    best_freqs = np.empty(sum(len(data[4]) for data in data_per_file))
    significances = np.empty(sum(len(data[5]) for data in data_per_file))

    # we also want arrays of ccd and quadrant numbers, to keep track of where the data comes from
    ccds = np.empty(sum(len(data[0]) for data in data_per_file), dtype=np.uint8)
    quads = np.empty(sum(len(data[0]) for data in data_per_file), dtype=np.uint8)

    current_idx_psids = 0
    current_idx_ra = 0
    current_idx_dec = 0
    current_idx_ratio_valid = 0
    current_idx_best_freqs = 0
    current_idx_significances = 0
    for i, data in tqdm(
        enumerate(data_per_file), desc="Processing data", total=len(data_per_file)
    ):
        # now we don't append to the arrays, but rather insert the data at the correct positions
        psids[current_idx_psids : current_idx_psids + len(data[0])] = data[0]
        ra[current_idx_ra : current_idx_ra + len(data[1])] = data[1]
        dec[current_idx_dec : current_idx_dec + len(data[2])] = data[2]
        ratio_valid[
            current_idx_ratio_valid : current_idx_ratio_valid + len(data[3])
        ] = data[3]
        best_freqs[current_idx_best_freqs : current_idx_best_freqs + len(data[4])] = (
            data[4]
        )
        significances[
            current_idx_significances : current_idx_significances + len(data[5])
        ] = data[5]

        file_name = files[i]
        ccd, quad = get_ccd_quad_from_filename(file_name)
        ccds[current_idx_psids : current_idx_psids + len(data[0])] = ccd
        quads[current_idx_psids : current_idx_psids + len(data[0])] = quad

        current_idx_psids += len(data[0])
        current_idx_ra += len(data[1])
        current_idx_dec += len(data[2])
        current_idx_ratio_valid += len(data[3])
        current_idx_best_freqs += len(data[4])
        current_idx_significances += len(data[5])

    if len(psids) == 0:
        raise ValueError("No data found in files")

    freqs = best_freqs.reshape((len(psids), 3, 50))
    sigs = significances.reshape((len(psids), 3, 50))

    return psids, ra, dec, ratio_valid, freqs, sigs, ccds, quads
