import glob
import os
from typing import Tuple

import h5py as h5
import numpy as np

def load_field_periodicity_data(field: int, band: int, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(field, (int, np.integer, str)):
        raise ValueError('Field must be an integer or string')
    if not isinstance(band, (int, np.integer, str)):
        raise ValueError('Band must be an integer or string')
    if path is None:
        raise ValueError('Path must be specified')
    
    path = os.path.join(
        os.path.abspath(path),
        f'{str(field).zfill(4)}',
        f'*_z{band}.h5'
    )

    files = glob.glob(path)

    # maybe we just want to return an empty list if no files are found,
    # rather than raising an error?
    if not files:
        raise ValueError('No files found')
    
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