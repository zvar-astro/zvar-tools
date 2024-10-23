from typing import Tuple

import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time

from zvar_utils.stats import median_abs_deviation

def process_curve(ra: float, dec: float, times: list, mags: list, magerrs: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.array(times)
    mags = np.array(mags)
    magerrs = np.array(magerrs)

    valid_indices = ~np.isnan(times)
    times = times[valid_indices]
    mags = mags[valid_indices]
    magerrs = magerrs[valid_indices]

    skycoord = coord.SkyCoord(ra, dec,
                            unit=(u.deg, u.deg), frame='icrs')
    palomar = coord.EarthLocation.of_site('Palomar')
    times = Time(np.array(times), format='mjd', scale='utc', location=palomar)
    ltt_bary = times.light_travel_time(skycoord) #Calculate light travel time
    barycorr_times = (times.tdb + ltt_bary).value #Add on the light travel time
    barycorr_times = barycorr_times - 2400000.5 #Convert from JD to BJD
    barycorr_times *= 86400 #Convert from BJD to BJS
    barycorr_times += 15 #midpoint correct ZTF timestamps
    barycorr_times -= np.min(barycorr_times) #Subtract off the zero point (ephemeris may be added later)

    #Convert mags to flux values
    mags = np.array(mags)
    magerrs = np.array(magerrs)
    med_mag = np.median(mags)

    flux = 10**(0.4*(med_mag - mags)) - 1.0
    ferrs = magerrs/1.086

    #Set nan values to zero
    flux[np.isnan(flux)] = 0
    ferrs[flux == 0] = np.inf
    
    #Basic MAD clip
    valid = np.where(np.abs(flux) < 5 * 1.483 * median_abs_deviation(flux))

    barycorr_times = barycorr_times[valid]
    flux = flux[valid]
    ferrs = ferrs[valid]

    flux -= np.mean(flux)

    return barycorr_times, flux, ferrs


def remove_deep_drilling(barycorr_times: np.ndarray, flux: np.ndarray, ferrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = 40.0  # seconds
    znew_times, znew_flux, znew_ferrs = [], [], []

    unique_days = np.unique(np.floor(barycorr_times / 86400.0))
    for ud in unique_days:
        mask = (barycorr_times > ud * 86400.0) & (barycorr_times < (ud + 1.0) * 86400.0)
        zday_times = barycorr_times[mask]
        zday_flux = flux[mask]
        zday_ferrs = ferrs[mask]
        Nz = len(zday_times)

        # If more than 10 exposures in a night, check time sampling
        if Nz > 10:
            tsec = zday_times
            zdiff = ((tsec[1:] - tsec[:-1]) < dt)

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
            znew_ferrs.append(zday_ferrs[~zdiff])
        else:
            znew_times.append(zday_times)
            znew_flux.append(zday_flux)
            znew_ferrs.append(zday_ferrs)

    znew_times = np.concatenate(znew_times)
    znew_flux = np.concatenate(znew_flux)
    znew_ferrs = np.concatenate(znew_ferrs)

    return znew_times, znew_flux, znew_ferrs

def freq_grid(t: np.ndarray, fmin: float = None, fmax: float = None, oversample: int = 3) -> np.ndarray:
    trange = max(t) - min(t)
    texp = np.nanmin(np.diff(np.sort(t)))
    fres = 1./trange/oversample
    if fmax is None:
        fmax = 0.5 / texp
    if fmin is None:
        fmin = fres
    fgrid = np.arange(fmin,fmax,fres)
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
    ef = np.array([
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
        (f_day * 10, 0.05 / 86400)  # 1/10 day
    ])

    # Create array to store keep/discard indices
    keep_freq = np.ones(len(frequencies), dtype=int)

    # Vectorized operation to flag frequencies
    for center, width in ef:
        mask = (frequencies > center - width) & (frequencies < center + width)
        keep_freq[mask] = 0

    return keep_freq