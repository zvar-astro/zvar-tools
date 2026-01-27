import numpy as np
import os
import glob
import argparse
import sys
import shutil
import multiprocessing
import signal
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import re

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(processName)s/%(threadName)s] %(message)s"
if os.environ.get("ZVAR_VERBOSE", "").lower() in ("1", "true", "yes"):
    logging.basicConfig(level=logging.DEBUG, format=_LOG_FORMAT)
else:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
LOG = logging.getLogger("merge_matchfiles")

def look_for_files(input_path):
    """
    Find zforce (*.cor.h5) and imdiff data_*_z[gri].h5 files, grouping them
    by base name with the band letter stripped.

    Returns: dict {base_name: [file_paths]}
    """
    from collections import defaultdict

    files = []
    files.extend(glob.glob(os.path.join(input_path, "zforce*_[gri].cor.h5")))
    files.extend(glob.glob(os.path.join(input_path, "data*_z[gri].h5")))

    grouped_files = defaultdict(list)
    # regex for zforce and data patterns
    pat_zforce = re.compile(r"(?P<base>.+)_[gri]\.cor\.h5$")
    pat_data = re.compile(r"(?P<base>.+)_z[gri]\.h5$")

    for file in files:
        name = os.path.basename(file)
        base = None
        m = pat_zforce.match(name)
        if m:
            base = m.group("base")
        else:
            m = pat_data.match(name)
            if m:
                base = m.group("base")
        if base:
            grouped_files[base].append(file)

    LOG.debug("look_for_files found %d groups", len(grouped_files))
    return grouped_files

def ensure_output_dir(output_path, field):
    """Create the output directory for a given field if it doesn't exist."""
    dirpath = os.path.join(output_path, f"{field:04d}")
    os.makedirs(dirpath, exist_ok=True)

def remove_polynomial_baseline_gpufit(
    data: np.ndarray,
    times: np.ndarray,
    tolerance: float = 1e-4,
    max_iterations: int = 1000,
    weights: np.ndarray = None,
    return_params: bool = False,
    verbose: bool = True
) -> np.ndarray:
    """
    Remove a 7-degree polynomial baseline from light curves using GPU-accelerated fitting.
    
    Fits a 7-degree polynomial model to N_curves of N_exp observations each with the 
    same timestamps, then subtracts the best-fit model from the original data.
    Uses gpufit for fast GPU-accelerated polynomial fitting.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (N_curves, N_exp). Light curve flux values.
    times : np.ndarray
        Shape (N_exp,). Time values for all observations (same for all curves).
    tolerance : float, optional
        Convergence tolerance for fitting (default: 1e-3).
    max_iterations : int, optional
        Maximum iterations for fitter (default: 5000).
    weights : np.ndarray, optional
        Shape (N_curves, N_exp) or (N_exp,). Inverse-variance weights.
        If None, equal weights are used. If 1D, broadcast to all curves.
    return_params : bool, optional
        If True, also return fitted parameters array (default: False).
    verbose : bool, optional
        If True, log convergence statistics (default: True).
    
    Returns
    -------
    np.ndarray
        Shape (N_curves, N_exp). Residuals (data - best_fit_model).
    np.ndarray (optional)
        If return_params=True, also returns fitted parameters array
        of shape (N_curves, 8) for 7-degree polynomial coefficients.
    
    Raises
    ------
    ImportError
        If pygpufit is not installed.
    ValueError
        If data dimensions are invalid.
    
    Examples
    --------
    >>> data = np.random.randn(100, 1000)  # 100 curves, 1000 points each
    >>> times = np.linspace(0, 100, 1000)
    >>> residuals = remove_polynomial_baseline_gpufit(data, times)
    >>> residuals.shape
    (100, 1000)
    """

    try:
        import pygpufit.gpufit as gf
    except ImportError:
        raise ImportError(
            "pygpufit is required for polynomial baseline removal. "
            "Install it with: pip install pygpufit"
        )
    
    # Validate inputs
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    
    n_curves, n_points = data.shape
    
    if times.ndim != 1 or len(times) != n_points:
        raise ValueError(
            f"times must be 1D with length {n_points}, got shape {times.shape}"
        )
    
    # Use 7-degree polynomial (8 parameters)
    model_id = gf.ModelID.POLYNOMIAL_7DEGREE_1D
    n_parameters = 8
    
    # Ensure data is float32 for gpufit
    data_fit = data.astype(np.float32)
    
    # Normalize times: subtract minimum and divide by range to get [0, 1]
    t_min = np.min(times)
    t_max = np.max(times)
    t_range = t_max - t_min
    if t_range == 0:
        raise ValueError("times cannot all be identical")
    times_normalized = ((times - t_min) / t_range).astype(np.float32)
    
    # Handle weights
    if weights is None:
        # Equal weights for all points
        weights_fit = np.ones((n_curves, n_points), dtype=np.float32)
    else:
        weights_fit = weights.astype(np.float32)
        if weights_fit.ndim == 1:
            if len(weights_fit) != n_points:
                raise ValueError(
                    f"weights must have length {n_points}, got {len(weights_fit)}"
                )
            # Broadcast 1D weights to all curves
            weights_fit = np.tile(weights_fit, (n_curves, 1))
        elif weights_fit.shape != (n_curves, n_points):
            raise ValueError(
                f"weights shape must be ({n_curves}, {n_points}), "
                f"got {weights_fit.shape}"
            )
    
    # Sanitize data: replace NaN/Inf with flux=0, zero weight
    # This allows fitting to proceed while ignoring bad points
    bad_flux_mask = ~np.isfinite(data_fit)
    bad_weight_mask = ~np.isfinite(weights_fit) | (weights_fit <= 0)
    
    n_bad_flux = np.sum(bad_flux_mask)
    n_bad_weights = np.sum(bad_weight_mask)
    
    if n_bad_flux > 0:
        data_fit[bad_flux_mask] = 0.0
        if verbose:
            n_affected_curves = np.sum(np.any(bad_flux_mask, axis=1))
            LOG.warning(f"  Found {n_bad_flux} NaN/Inf flux values in {n_affected_curves} curves - replacing with 0 for fitting")
    
    if n_bad_weights > 0:
        weights_fit[bad_weight_mask] = 0.0  # Zero weight = ignore these points
        if verbose:
            n_affected_curves = np.sum(np.any(bad_weight_mask, axis=1))
            LOG.warning(f"  Found {n_bad_weights} invalid weights in {n_affected_curves} curves - setting to 0 (ignored in fit)")
    
    # Check for curves with insufficient valid data
    n_valid_per_curve = np.sum(np.isfinite(data_fit) & (weights_fit > 0), axis=1)
    min_points_needed = 50  # Need at least 50 points for robust 8-parameter fit
    insufficient_mask = n_valid_per_curve < min_points_needed
    
    if np.any(insufficient_mask):
        # Set all weights to tiny value for these curves so fit returns zeros
        weights_fit[insufficient_mask, :] = 1e-10
        if verbose:
            n_insufficient = np.sum(insufficient_mask)
            LOG.warning(f"  Found {n_insufficient} curves with <{min_points_needed} valid points - baseline removal will be skipped for these")
    
    # Initialize parameters (zero guess works well for polynomials)
    initial_parameters = np.zeros((n_curves, n_parameters), dtype=np.float32)
    
    # Run the fit
    params, states, chi2, n_iter, t_exec = gf.fit(
        data_fit,
        weights_fit,
        model_id,
        initial_parameters,
        tolerance=tolerance,
        max_number_iterations=max_iterations,
        parameters_to_fit=None,
        estimator_id=gf.EstimatorID.LSE,
        user_info=times_normalized,
    )

    if verbose:
        n_converged = np.sum(states == 0)
        n_max_iter = np.sum(states == 1)
        convergence_rate = n_converged / n_curves * 100
        
        LOG.info(
            "Baseline removal: %d curves, %d converged (%.1f%%), %d max iterations",
            n_curves, n_converged, convergence_rate, n_max_iter
        )
        
        if n_converged > 0:
            converged_chi2 = chi2[states == 0]
            LOG.info(
                "  Chi-squared (converged): mean=%.2f, median=%.2f, std=%.2f",
                np.mean(converged_chi2), np.median(converged_chi2), np.std(converged_chi2)
            )
    
    # Generate fitted models for all curves
    fitted_models = np.zeros_like(data_fit)
    
    for curve_idx in range(n_curves):
        y_fit = np.zeros(n_points, dtype=np.float32)
        for param_idx, coeff in enumerate(params[curve_idx]):
            power = n_parameters - 1 - param_idx
            y_fit += coeff * (times_normalized ** power)
        fitted_models[curve_idx, :] = y_fit
    
    # Compute residuals
    residuals = data_fit - fitted_models
    
    if return_params:
        return residuals, params
    else:
        return residuals

def copy_h5_file(src, dst):
    """
    Copy an HDF5 file from src to dst.
    """
    # Ensure the destination directory exists to avoid FileNotFoundError
    dst_dir = os.path.dirname(dst)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    LOG.debug("copy_h5_file src=%s dst=%s", src, dst)
    shutil.copy2(src, dst)

def copy_single_file(input_path, output_path, filename, remove_baseline=False):
    """
    Copy or process a single g-band file.
    Args:
        input_path: directory containing the file
        output_path: directory for output
        filename: basename of file (e.g., "data_0450_03_1_zg.h5")
        remove_baseline: if True, apply GPU baseline removal
    """
    import h5py
    import gc
    
    input_file = os.path.join(input_path, filename)
    output_file = os.path.join(output_path, f"comb_{os.path.basename(filename)}")
    os.makedirs(output_path, exist_ok=True)
    
    if not remove_baseline:
        # Simple copy if no baseline removal needed
        copy_h5_file(input_file, output_file)
        LOG.info("copy_single_file done filename=%s", filename)
        return
    
    # Load all data for baseline removal
    LOG.info("Removing polynomial baselines for filename=%s", filename)
    
    with h5py.File(input_file, "r") as f:
        # Load exposure metadata
        exposures_data = f["/data/exposures"][:]
        bjd = f["/data/exposures"]["bjd"][:]
        
        # Load source metadata
        sources_data = f["/data/sources"][:]
        psid = f["/data/sources"]["gaia_id"][:]
        
        # Load sourcedata
        flux = f["/data/sourcedata"]["flux"][:]
        fluxerr = f["/data/sourcedata"]["flux_err"][:]
        flag = f["/data/sourcedata"]["flag"][:]
    
    # Reshape and compute weights
    n_sources = len(psid)
    n_exp = len(bjd)
    flux_2d = flux.reshape(n_sources, n_exp)
    fluxerr_2d = fluxerr.reshape(n_sources, n_exp)
    
    fluxerr_safe = np.where(fluxerr_2d > 0, fluxerr_2d, 1e10)
    weights = 1.0 / (fluxerr_safe ** 2)
    
    LOG.info("  Using flux errors as inverse-variance weights")
    
    # Remove baseline
    flux_residuals = remove_polynomial_baseline_gpufit(
        flux_2d, bjd, weights=weights, tolerance=1e-3, max_iterations=5000, verbose=True
    )
    
    # Flatten for output
    flux_flat = flux_residuals.flatten()
    
    LOG.info("Baseline removal complete")
    
    # Write output file with baseline-removed flux
    sourcedata_dtype = np.dtype([
        ('flux', flux_flat.dtype),
        ('flux_err', fluxerr.dtype),
        ('flag', flag.dtype)
    ])
    n_total = flux_flat.size
    sourcedata_table = np.zeros(n_total, dtype=sourcedata_dtype)
    sourcedata_table["flux"] = flux_flat
    sourcedata_table["flux_err"] = fluxerr
    sourcedata_table["flag"] = flag
    
    with h5py.File(output_file, "w") as f:
        data_group = f.create_group("data")
        data_group.create_dataset("sources", data=sources_data, compression="gzip", compression_opts=1)
        data_group.create_dataset("exposures", data=exposures_data, compression="gzip", compression_opts=1)
        data_group.create_dataset("sourcedata", data=sourcedata_table, compression="gzip", compression_opts=1)
    
    gc.collect()
    LOG.info("copy single_file done filename=%s", filename)

def merge_g_r(input_path, output_path, g_filename, r_filename, remove_baseline=False):
    import h5py  # lazy import
    import gc

    # g_file = os.path.join(input_path, g_filename)
    # r_file = os.path.join(input_path, r_filename)

    g_basename = os.path.basename(g_filename)
    r_basename = os.path.basename(r_filename)

    g_file = g_file = os.path.join(input_path, g_basename) if not os.path.isabs(g_filename) else g_filename
    r_file = r_file = os.path.join(input_path, r_basename) if not os.path.isabs(r_filename) else r_filename
    
    # Extract base name (strip band letter) for output
    base_g = g_basename.replace("_zg.h5", "").replace("_g.cor.h5", "")
    output_file = os.path.join(output_path, f"comb_{base_g}.h5")
    os.makedirs(output_path, exist_ok=True)
    
    LOG.info("merge_g_r start g_file=%s r_file=%s", g_filename, r_filename)
    
    if not os.path.exists(g_file):
        raise FileNotFoundError(f"g file not found: {g_file}")
    if not os.path.exists(r_file):
        raise FileNotFoundError(f"r file not found: {r_file}")

    # Load relevant data from each file
    try:
        LOG.debug("Opening g file: %s", g_file)
        with h5py.File(g_file, "r") as f:
            # g_jd = f["/data/exposures"]["jd"][:]
            g_bjd = f["/data/exposures"]["bjd"][:]
            # g_filterid = f["/data/exposures"]["filterid"][:]
            g_filterid = np.full_like(g_bjd, fill_value=1, dtype=np.uint16)  # g-band=1
            # g_exptime = f["/data/exposures"]["exptime"][:]
            # g_pid = f["/data/exposures"]["pid"][:]
            # g_field = f["/data/exposures"]["field"][:]
            # g_ccd = f["/data/exposures"]["ccd"][:]
            # g_quad = f["/data/exposures"]["quad"][:]
            # g_imstat = f["/data/exposures"]["imstat"][:]
            # g_infobits = f["/data/exposures"]["infobits"][:]
            # g_seeing = f["/data/exposures"]["seeing"][:]
            # g_mzpsci = f["/data/exposures"]["mzpsci"][:]
            # g_mzpsciunc = f["/data/exposures"]["mzpsciunc"][:]
            # g_mzpscirms = f["/data/exposures"]["mzpscirms"][:]
            # g_clrco = f["/data/exposures"]["clrco"][:]
            # g_clrcounc = f["/data/exposures"]["clrcounc"][:]
            # g_maglim = f["/data/exposures"]["maglim"][:]
            # g_airmass = f["/data/exposures"]["airmass"][:]
            # g_nps1matches = f["/data/exposures"]["nps1matches"][:]

            g_psid = f["/data/sources"]["gaia_id"][:]
            g_ra = f["/data/sources"]["ra"][:]
            g_dec = f["/data/sources"]["decl"][:]
            # g_mag_ref = f["/data/sources"]["mag_ref"][:]
            # g_mag_err_ref = f["/data/sources"]["mag_err_ref"][:]
            # g_objtype = f["/data/sources"]["objtype"][:]

            g_flux = f["/data/sourcedata"]["flux"][:]
            g_fluxerr = f["/data/sourcedata"]["flux_err"][:]
            # g_flag = f["/data/sourcedata"]["flag"][:]
        LOG.debug("Read g: %d sources, %d exposures", len(g_psid), len(g_bjd))
        gc.collect()

        LOG.debug("Opening r file: %s", r_file)
        with h5py.File(r_file, "r") as f:
            # r_jd = f["/data/exposures"]["jd"][:]
            r_bjd = f["/data/exposures"]["bjd"][:]
            # r_filterid = f["/data/exposures"]["filterid"][:]
            r_filterid = np.full_like(r_bjd, fill_value=2, dtype=np.uint16)  # r-band=2
            # r_exptime = f["/data/exposures"]["exptime"][:]
            # r_pid = f["/data/exposures"]["pid"][:]
            # r_field = f["/data/exposures"]["field"][:]
            # r_ccd = f["/data/exposures"]["ccd"][:]
            # r_quad = f["/data/exposures"]["quad"][:]
            # r_imstat = f["/data/exposures"]["imstat"][:]
            # r_infobits = f["/data/exposures"]["infobits"][:]
            # r_seeing = f["/data/exposures"]["seeing"][:]
            # r_mzpsci = f["/data/exposures"]["mzpsci"][:]
            # r_mzpsciunc = f["/data/exposures"]["mzpsciunc"][:]
            # r_mzpscirms = f["/data/exposures"]["mzpscirms"][:]
            # r_clrco = f["/data/exposures"]["clrco"][:]
            # r_clrcounc = f["/data/exposures"]["clrcounc"][:]
            # r_maglim = f["/data/exposures"]["maglim"][:]
            # r_airmass = f["/data/exposures"]["airmass"][:]
            # r_nps1matches = f["/data/exposures"]["nps1matches"][:]

            r_psid = f["/data/sources"]["gaia_id"][:]
            r_ra = f["/data/sources"]["ra"][:]
            r_dec = f["/data/sources"]["decl"][:]
            # r_mag_ref = f["/data/sources"]["mag_ref"][:]
            # r_mag_err_ref = f["/data/sources"]["mag_err_ref"][:]
            # r_objtype = f["/data/sources"]["objtype"][:]

            r_flux = f["/data/sourcedata"]["flux"][:]
            r_fluxerr = f["/data/sourcedata"]["flux_err"][:]
            # r_flag = f["/data/sourcedata"]["flag"][:]
        LOG.debug("Read r: %d sources, %d exposures", len(r_psid), len(r_bjd))
        gc.collect()

    except Exception as e:
        LOG.error("Error reading files for g_file=%s r_file=%s: %s", g_filename, r_filename, e, exc_info=True)
        raise

    finally:
        # Explicit cleanup
        gc.collect()

    if remove_baseline:
        LOG.info("Removing polynomial baselines for g_file=%s r_file=%s", g_file, r_file)
        
        # Reshape from flattened (n_sources * n_exp) to 2D (n_sources, n_exp)
        g_n_sources = len(g_psid)
        r_n_sources = len(r_psid)
        g_n_exp = len(g_bjd)
        r_n_exp = len(r_bjd)
        
        g_flux_2d = g_flux.reshape(g_n_sources, g_n_exp)
        r_flux_2d = r_flux.reshape(r_n_sources, r_n_exp)
        
        # Reshape flux errors and compute inverse-variance weights
        g_fluxerr_2d = g_fluxerr.reshape(g_n_sources, g_n_exp)
        r_fluxerr_2d = r_fluxerr.reshape(r_n_sources, r_n_exp)
        
        # Compute weights as inverse variance, handling zeros/negatives
        # Set minimum error floor to avoid division by zero
        g_fluxerr_safe = np.where(g_fluxerr_2d > 0, g_fluxerr_2d, 1e10)
        r_fluxerr_safe = np.where(r_fluxerr_2d > 0, r_fluxerr_2d, 1e10)
        
        g_weights = 1.0 / (g_fluxerr_safe ** 2)
        r_weights = 1.0 / (r_fluxerr_safe ** 2)
        
        LOG.info("  Using flux errors as inverse-variance weights")
        
        # Remove baselines with proper weighting
        g_flux_residuals = remove_polynomial_baseline_gpufit(
            g_flux_2d, g_bjd, weights=g_weights, tolerance=1e-4, max_iterations=1000, verbose=True
        )
        r_flux_residuals = remove_polynomial_baseline_gpufit(
            r_flux_2d, r_bjd, weights=r_weights, tolerance=1e-4, max_iterations=1000, verbose=True
        )
        
        # Flatten back to 1D for downstream processing
        g_flux = g_flux_residuals.flatten()
        r_flux = r_flux_residuals.flatten()
        
        LOG.info("Baseline removal complete")
        gc.collect()

    # Find unique gaia_ids (psids)
    # Build quick lookup maps from gaia_id->index to avoid np.where inside loop
    # Convert keys to python int to make lookups faster and robust
    g_map = {int(k): idx for idx, k in enumerate(g_psid)}
    r_map = {int(k): idx for idx, k in enumerate(r_psid)}
    LOG.debug("Built g_map (%d) and r_map (%d)", len(g_map), len(r_map))

    keys = sorted(g_map.keys() | r_map.keys())
    unique_ids = np.array(keys, dtype=g_psid.dtype)

    del g_psid, r_psid, keys
    gc.collect()

    LOG.info("Total unique sources to combine: %d", len(unique_ids))

    # Combine exposures metadata
    # comb_jd = np.concatenate((g_jd, r_jd))
    comb_bjd = np.concatenate((g_bjd, r_bjd))
    comb_filterid = np.concatenate((g_filterid, r_filterid))
    # comb_exptime = np.concatenate((g_exptime, r_exptime))
    # comb_pid = np.concatenate((g_pid, r_pid))
    # comb_field = np.concatenate((g_field, r_field))
    # comb_ccd = np.concatenate((g_ccd, r_ccd))
    # comb_quad = np.concatenate((g_quad, r_quad))
    # comb_imstat = np.concatenate((g_imstat, r_imstat))
    # comb_infobits = np.concatenate((g_infobits, r_infobits))
    # comb_seeing = np.concatenate((g_seeing, r_seeing))
    # comb_mzpsci = np.concatenate((g_mzpsci, r_mzpsci))
    # comb_mzpsciunc = np.concatenate((g_mzpsciunc, r_mzpsciunc))
    # comb_mzpscirms = np.concatenate((g_mzpscirms, r_mzpscirms))
    # comb_clrco = np.concatenate((g_clrco, r_clrco))
    # comb_clrcounc = np.concatenate((g_clrcounc, r_clrcounc))
    # comb_maglim = np.concatenate((g_maglim, r_maglim))
    # comb_airmass = np.concatenate((g_airmass, r_airmass))
    # comb_nps1matches = np.concatenate((g_nps1matches, r_nps1matches))

    g_nexp = len(g_bjd)
    r_nexp = len(r_bjd)

    # Prepare output arrays
    comb_ra = np.zeros(len(unique_ids), dtype=g_ra.dtype)
    comb_dec = np.zeros(len(unique_ids), dtype=g_dec.dtype)
    # comb_mag_ref = np.zeros(len(unique_ids), dtype=g_mag_ref.dtype)
    # comb_mag_err_ref = np.zeros(len(unique_ids), dtype=g_mag_err_ref.dtype)
    # comb_objtype = np.zeros(len(unique_ids), dtype=g_objtype.dtype)

    comb_flux = np.zeros((len(unique_ids), g_nexp + r_nexp), dtype=g_flux.dtype)
    comb_fluxerr = np.zeros((len(unique_ids), g_nexp + r_nexp), dtype=g_fluxerr.dtype)
    # comb_flag = np.zeros((len(unique_ids), g_nexp + r_nexp), dtype=g_flag.dtype)

    # del g_jd, r_jd, g_bjd, r_bjd, g_filterid, r_filterid
    # del g_exptime, r_exptime, g_pid, r_pid, g_field, r_field
    # del g_ccd, r_ccd, g_quad, r_quad, g_imstat, r_imstat
    # del g_infobits, r_infobits, g_seeing, r_seeing
    # del g_mzpsci, r_mzpsci, g_mzpsciunc, r_mzpsciunc
    # del g_mzpscirms, r_mzpscirms, g_clrco, r_clrco
    # del g_clrcounc, r_clrcounc, g_maglim, r_maglim
    # del g_airmass, r_airmass, g_nps1matches, r_nps1matches
    del g_bjd, r_bjd, g_filterid, r_filterid
    gc.collect()

    # Choose a progress interval to avoid logging every iteration
    total = len(unique_ids)
    progress_interval = max(1, total // 10)  # log 10 times across the run
    for uid_idx, psid in enumerate(unique_ids):
        pid = int(psid)
        # default values (will be overwritten if present)
        ra = 0
        dec = 0
        mag_ref = 0
        mag_err_ref = 0
        objtype = 0

        if pid in g_map:
            g_index = g_map[pid]
            start = g_index * g_nexp
            comb_flux[uid_idx, :g_nexp] = g_flux[start: start + g_nexp]
            comb_fluxerr[uid_idx, :g_nexp] = g_fluxerr[start: start + g_nexp]
            # comb_flag[uid_idx, :g_nexp] = g_flag[start: start + g_nexp]

            ra = g_ra[g_index]
            dec = g_dec[g_index]
            # mag_ref = g_mag_ref[g_index]
            # mag_err_ref = g_mag_err_ref[g_index]
            # objtype = g_objtype[g_index]

        if pid in r_map:
            r_index = r_map[pid]
            start = r_index * r_nexp
            comb_flux[uid_idx, g_nexp:] = r_flux[start: start + r_nexp]
            comb_fluxerr[uid_idx, g_nexp:] = r_fluxerr[start: start + r_nexp]
            # comb_flag[uid_idx, g_nexp:] = r_flag[start: start + r_nexp]

            # if not present in g, take r position (otherwise g already set)
            if pid not in g_map:
                ra = r_ra[r_index]
                dec = r_dec[r_index]
                # mag_ref = r_mag_ref[r_index]
                # mag_err_ref = r_mag_err_ref[r_index]
                # objtype = r_objtype[r_index]

        comb_ra[uid_idx] = ra
        comb_dec[uid_idx] = dec
        # comb_mag_ref[uid_idx] = mag_ref
        # comb_mag_err_ref[uid_idx] = mag_err_ref
        # comb_objtype[uid_idx] = objtype

        if uid_idx % progress_interval == 0:
            LOG.info("Processed %d / %d sources", uid_idx, total)

    # del g_flux, g_fluxerr, g_flag, r_flux, r_fluxerr, r_flag
    # del g_ra, r_ra, g_dec, r_dec
    # del g_mag_ref, r_mag_ref, g_mag_err_ref, r_mag_err_ref
    # del g_objtype, r_objtype
    # del g_map, r_map

    del g_flux, g_fluxerr, r_flux, r_fluxerr
    del g_ra, r_ra, g_dec, r_dec
    del g_map, r_map
    gc.collect()

    LOG.info("Finished combining sources; sorting exposures and building datasets")

    # Argsort everything by time
    sort_idx = np.argsort(comb_bjd)
    # comb_jd_sorted = comb_jd[sort_idx]
    comb_bjd_sorted = comb_bjd[sort_idx]
    comb_filterid_sorted = comb_filterid[sort_idx]
    # comb_exptime_sorted = comb_exptime[sort_idx]
    # comb_pid_sorted = comb_pid[sort_idx]
    # comb_field_sorted = comb_field[sort_idx]
    # comb_ccd_sorted = comb_ccd[sort_idx]
    # comb_quad_sorted = comb_quad[sort_idx]
    # comb_imstat_sorted = comb_imstat[sort_idx]
    # comb_infobits_sorted = comb_infobits[sort_idx]
    # comb_seeing_sorted = comb_seeing[sort_idx]
    # comb_mzpsci_sorted = comb_mzpsci[sort_idx]
    # comb_mzpsciunc_sorted = comb_mzpsciunc[sort_idx]
    # comb_mzpscirms_sorted = comb_mzpscirms[sort_idx]
    # comb_clrco_sorted = comb_clrco[sort_idx]
    # comb_clrcounc_sorted = comb_clrcounc[sort_idx]
    # comb_maglim_sorted = comb_maglim[sort_idx]
    # comb_airmass_sorted = comb_airmass[sort_idx]
    # comb_nps1matches_sorted = comb_nps1matches[sort_idx]

    comb_flux_sorted = comb_flux[:, sort_idx]
    comb_fluxerr_sorted = comb_fluxerr[:, sort_idx]
    # comb_flag_sorted = comb_flag[:, sort_idx]

    # del comb_jd, comb_bjd, comb_filterid, comb_exptime, comb_pid
    # del comb_field, comb_ccd, comb_quad, comb_imstat, comb_infobits
    # del comb_seeing, comb_mzpsci, comb_mzpsciunc, comb_mzpscirms
    # del comb_clrco, comb_clrcounc, comb_maglim, comb_airmass, comb_nps1matches
    # del comb_flux, comb_fluxerr, comb_flag
    # del sort_idx

    del comb_bjd, comb_filterid
    del comb_flux, comb_fluxerr
    del sort_idx
    gc.collect()

    # Write out the combined data to a new HDF5 file
    LOG.debug("Writing output file: %s", output_file)
    exposures_dtype = np.dtype([
        # ("jd", comb_jd_sorted.dtype),
        ("bjd", comb_bjd_sorted.dtype),
        ("filterid", comb_filterid_sorted.dtype)
        # ("exptime", comb_exptime_sorted.dtype),
        # ("pid", comb_pid_sorted.dtype),
        # ("field", comb_field_sorted.dtype),
        # ("ccd", comb_ccd_sorted.dtype),
        # ("quad", comb_quad_sorted.dtype),
        # ("imstat", comb_imstat_sorted.dtype),
        # ("infobits", comb_infobits_sorted.dtype),
        # ("seeing", comb_seeing_sorted.dtype),
        # ("mzpsci", comb_mzpsci_sorted.dtype),
        # ("mzpsciunc", comb_mzpsciunc_sorted.dtype),
        # ("mzpscirms", comb_mzpscirms_sorted.dtype),
        # ("clrco", comb_clrco_sorted.dtype),
        # ("clrcounc", comb_clrcounc_sorted.dtype),
        # ("maglim", comb_maglim_sorted.dtype),
        # ("airmass", comb_airmass_sorted.dtype),
        # ("nps1matches", comb_nps1matches_sorted.dtype)
    ])
    exposures_table = np.zeros(len(comb_bjd_sorted), dtype=exposures_dtype)
    # exposures_table["jd"] = comb_jd_sorted
    exposures_table["bjd"] = comb_bjd_sorted
    exposures_table["filterid"] = comb_filterid_sorted
    # exposures_table["exptime"] = comb_exptime_sorted
    # exposures_table["pid"] = comb_pid_sorted
    # exposures_table["field"] = comb_field_sorted
    # exposures_table["ccd"] = comb_ccd_sorted
    # exposures_table["quad"] = comb_quad_sorted
    # exposures_table["imstat"] = comb_imstat_sorted
    # exposures_table["infobits"] = comb_infobits_sorted
    # exposures_table["seeing"] = comb_seeing_sorted
    # exposures_table["mzpsci"] = comb_mzpsci_sorted
    # exposures_table["mzpsciunc"] = comb_mzpsciunc_sorted
    # exposures_table["mzpscirms"] = comb_mzpscirms_sorted
    # exposures_table["clrco"] = comb_clrco_sorted
    # exposures_table["clrcounc"] = comb_clrcounc_sorted
    # exposures_table["maglim"] = comb_maglim_sorted
    # exposures_table["airmass"] = comb_airmass_sorted
    # exposures_table["nps1matches"] = comb_nps1matches_sorted

    sources_dtype = np.dtype([
        ("gaia_id", unique_ids.dtype),
        ("ra", comb_ra.dtype),
        ("decl", comb_dec.dtype)
        # ("mag_ref", comb_mag_ref.dtype),
        # ("mag_err_ref", comb_mag_err_ref.dtype),
        # ("objtype", comb_objtype.dtype)
    ])
    sources_table = np.zeros(len(unique_ids), dtype=sources_dtype)
    sources_table["gaia_id"] = unique_ids
    sources_table["ra"] = comb_ra
    sources_table["decl"] = comb_dec
    # sources_table["mag_ref"] = comb_mag_ref
    # sources_table["mag_err_ref"] = comb_mag_err_ref
    # sources_table["objtype"] = comb_objtype

    n_total = comb_flux_sorted.size
    sourcedata_dtype = np.dtype([
        ('flux', comb_flux_sorted.dtype),
        ('flux_err', comb_fluxerr_sorted.dtype),
        # ('flag', comb_flag_sorted.dtype)
    ])
    sourcedata_table = np.zeros(n_total, dtype=sourcedata_dtype)
    sourcedata_table["flux"] = comb_flux_sorted.flatten()
    sourcedata_table["flux_err"] = comb_fluxerr_sorted.flatten()
    # sourcedata_table["flag"] = comb_flag_sorted.flatten()

    with h5py.File(output_file, "w") as f:
        # Create groups as in the original files
        data_group = f.create_group("data")
        data_group.create_dataset("sources", data=sources_table, compression="gzip", compression_opts=1)
        data_group.create_dataset("exposures", data=exposures_table, compression="gzip", compression_opts=1)
        data_group.create_dataset("sourcedata", data=sourcedata_table, compression="gzip", compression_opts=1)

    # del comb_jd_sorted, comb_bjd_sorted, comb_filterid_sorted
    # del comb_exptime_sorted, comb_pid_sorted, comb_field_sorted
    # del comb_ccd_sorted, comb_quad_sorted, comb_imstat_sorted
    # del comb_infobits_sorted, comb_seeing_sorted, comb_mzpsci_sorted
    # del comb_mzpsciunc_sorted, comb_mzpscirms_sorted, comb_clrco_sorted
    # del comb_clrcounc_sorted, comb_maglim_sorted, comb_airmass_sorted
    # del comb_nps1matches_sorted
    # del comb_ra, comb_dec, comb_mag_ref, comb_mag_err_ref, comb_objtype
    # del comb_flux_sorted, comb_fluxerr_sorted, comb_flag_sorted
    # del exposures_table, sources_table, sourcedata_table

    del comb_bjd_sorted, comb_filterid_sorted
    del comb_ra, comb_dec
    del comb_flux_sorted, comb_fluxerr_sorted
    del exposures_table, sources_table, sourcedata_table
    gc.collect()

def merge_g_r_i(input_path, output_path, g_filename, r_filename, i_filename, remove_baseline=False):
    import h5py  # lazy import
    import gc

    # g_file = os.path.join(input_path, g_filename)
    # r_file = os.path.join(input_path, r_filename)
    # i_file = os.path.join(input_path, i_filename)

    g_basename = os.path.basename(g_filename)
    r_basename = os.path.basename(r_filename)
    i_basename = os.path.basename(i_filename)

    g_file = g_file = os.path.join(input_path, g_basename) if not os.path.isabs(g_filename) else g_filename
    r_file = r_file = os.path.join(input_path, r_basename) if not os.path.isabs(r_filename) else r_filename
    i_file = i_file = os.path.join(input_path, i_basename) if not os.path.isabs(i_filename) else i_filename

    # Extract base name (strip band letter) for output
    base_g = g_basename.replace("_zg.h5", "").replace("_g.cor.h5", "")
    output_file = os.path.join(output_path, f"comb_{base_g}.h5")
    os.makedirs(output_path, exist_ok=True)

    LOG.info("merge_g_r_i start g_file=%s r_file=%s i_file=%s", g_file, r_file, i_file)

    if not os.path.exists(g_file):
        raise FileNotFoundError(f"g file not found: {g_file}")
    if not os.path.exists(r_file):
        raise FileNotFoundError(f"r file not found: {r_file}")
    if not os.path.exists(i_file):
        raise FileNotFoundError(f"i file not found: {i_file}")

    #Load relevant data from each file
    LOG.debug("Opening g file: %s", g_file)
    with h5py.File(g_file, "r") as f:
        g_jd = f["/data/exposures"]["jd"][:]
        g_bjd = f["/data/exposures"]["bjd"][:]
        # g_filterid = f["/data/exposures"]["filterid"][:]
        g_filterid = np.full_like(g_bjd, fill_value=1, dtype=np.uint16)  # g-band=1
        g_exptime = f["/data/exposures"]["exptime"][:]
        g_pid = f["/data/exposures"]["pid"][:]
        g_field = f["/data/exposures"]["field"][:]
        g_ccd = f["/data/exposures"]["ccd"][:]
        g_quad = f["/data/exposures"]["quad"][:]
        g_imstat = f["/data/exposures"]["imstat"][:]
        g_infobits = f["/data/exposures"]["infobits"][:]
        g_seeing = f["/data/exposures"]["seeing"][:]
        g_mzpsci = f["/data/exposures"]["mzpsci"][:]
        g_mzpsciunc = f["/data/exposures"]["mzpsciunc"][:]
        g_mzpscirms = f["/data/exposures"]["mzpscirms"][:]
        g_clrco = f["/data/exposures"]["clrco"][:]
        g_clrcounc = f["/data/exposures"]["clrcounc"][:]
        g_maglim = f["/data/exposures"]["maglim"][:]
        g_airmass = f["/data/exposures"]["airmass"][:]
        g_nps1matches = f["/data/exposures"]["nps1matches"][:]

        g_psid = f["/data/sources"]["gaia_id"][:]
        g_ra = f["/data/sources"]["ra"][:]
        g_dec = f["/data/sources"]["decl"][:]
        g_mag_ref = f["/data/sources"]["mag_ref"][:]
        g_mag_err_ref = f["/data/sources"]["mag_err_ref"][:]
        g_objtype = f["/data/sources"]["objtype"][:]

        g_flux = f["/data/sourcedata"]["flux"][:]
        g_fluxerr = f["/data/sourcedata"]["flux_err"][:]
        g_flag = f["/data/sourcedata"]["flag"][:]
    LOG.debug("Read g: %d sources, %d exposures", len(g_psid), len(g_bjd))

    LOG.debug("Opening r file: %s", r_file)
    with h5py.File(r_file, "r") as f:
        r_jd = f["/data/exposures"]["jd"][:]
        r_bjd = f["/data/exposures"]["bjd"][:]
        # r_filterid = f["/data/exposures"]["filterid"][:]
        r_filterid = np.full_like(r_bjd, fill_value=2, dtype=np.uint16)  # r-band=2
        r_exptime = f["/data/exposures"]["exptime"][:]
        r_pid = f["/data/exposures"]["pid"][:]
        r_field = f["/data/exposures"]["field"][:]
        r_ccd = f["/data/exposures"]["ccd"][:]
        r_quad = f["/data/exposures"]["quad"][:]
        r_imstat = f["/data/exposures"]["imstat"][:]
        r_infobits = f["/data/exposures"]["infobits"][:]
        r_seeing = f["/data/exposures"]["seeing"][:]
        r_mzpsci = f["/data/exposures"]["mzpsci"][:]
        r_mzpsciunc = f["/data/exposures"]["mzpsciunc"][:]
        r_mzpscirms = f["/data/exposures"]["mzpscirms"][:]
        r_clrco = f["/data/exposures"]["clrco"][:]
        r_clrcounc = f["/data/exposures"]["clrcounc"][:]
        r_maglim = f["/data/exposures"]["maglim"][:]
        r_airmass = f["/data/exposures"]["airmass"][:]
        r_nps1matches = f["/data/exposures"]["nps1matches"][:]

        r_psid = f["/data/sources"]["gaia_id"][:]
        r_ra = f["/data/sources"]["ra"][:]
        r_dec = f["/data/sources"]["decl"][:]
        r_mag_ref = f["/data/sources"]["mag_ref"][:]
        r_mag_err_ref = f["/data/sources"]["mag_err_ref"][:]
        r_objtype = f["/data/sources"]["objtype"][:]

        r_flux = f["/data/sourcedata"]["flux"][:]
        r_fluxerr = f["/data/sourcedata"]["flux_err"][:]
        r_flag = f["/data/sourcedata"]["flag"][:]
    LOG.debug("Read r: %d sources, %d exposures", len(r_psid), len(r_bjd))

    LOG.debug("Opening i file: %s", i_file)
    with h5py.File(i_file, "r") as f:
        i_jd = f["/data/exposures"]["jd"][:]
        i_bjd = f["/data/exposures"]["bjd"][:]
        # i_filterid = f["/data/exposures"]["filterid"][:]
        i_filterid = np.full_like(i_bjd, fill_value=3, dtype=np.uint16)  # i-band=3
        i_exptime = f["/data/exposures"]["exptime"][:]
        i_pid = f["/data/exposures"]["pid"][:]
        i_field = f["/data/exposures"]["field"][:]
        i_ccd = f["/data/exposures"]["ccd"][:]
        i_quad = f["/data/exposures"]["quad"][:]
        i_imstat = f["/data/exposures"]["imstat"][:]
        i_infobits = f["/data/exposures"]["infobits"][:]
        i_seeing = f["/data/exposures"]["seeing"][:]
        i_mzpsci = f["/data/exposures"]["mzpsci"][:]
        i_mzpsciunc = f["/data/exposures"]["mzpsciunc"][:]
        i_mzpscirms = f["/data/exposures"]["mzpscirms"][:]
        i_clrco = f["/data/exposures"]["clrco"][:]
        i_clrcounc = f["/data/exposures"]["clrcounc"][:]
        i_maglim = f["/data/exposures"]["maglim"][:]
        i_airmass = f["/data/exposures"]["airmass"][:]
        i_nps1matches = f["/data/exposures"]["nps1matches"][:]

        i_psid = f["/data/sources"]["gaia_id"][:]
        i_ra = f["/data/sources"]["ra"][:]
        i_dec = f["/data/sources"]["decl"][:]
        i_mag_ref = f["/data/sources"]["mag_ref"][:]
        i_mag_err_ref = f["/data/sources"]["mag_err_ref"][:]
        i_objtype = f["/data/sources"]["objtype"][:]

        i_flux = f["/data/sourcedata"]["flux"][:]
        i_fluxerr = f["/data/sourcedata"]["flux_err"][:]
        i_flag = f["/data/sourcedata"]["flag"][:]
    LOG.debug("Read i: %d sources, %d exposures", len(i_psid), len(i_bjd))

    if remove_baseline:
        LOG.info("Removing polynomial baselines for g_file=%s r_file=%s i_file=%s", g_file, r_file, i_file)
        
        # Reshape from flattened to 2D
        g_n_sources = len(g_psid)
        r_n_sources = len(r_psid)
        i_n_sources = len(i_psid)
        g_n_exp = len(g_bjd)
        r_n_exp = len(r_bjd)
        i_n_exp = len(i_bjd)
        
        g_flux_2d = g_flux.reshape(g_n_sources, g_n_exp)
        r_flux_2d = r_flux.reshape(r_n_sources, r_n_exp)
        i_flux_2d = i_flux.reshape(i_n_sources, i_n_exp)
        
        # Compute inverse-variance weights
        g_fluxerr_2d = g_fluxerr.reshape(g_n_sources, g_n_exp)
        r_fluxerr_2d = r_fluxerr.reshape(r_n_sources, r_n_exp)
        i_fluxerr_2d = i_fluxerr.reshape(i_n_sources, i_n_exp)
        
        g_fluxerr_safe = np.where(g_fluxerr_2d > 0, g_fluxerr_2d, 1e10)
        r_fluxerr_safe = np.where(r_fluxerr_2d > 0, r_fluxerr_2d, 1e10)
        i_fluxerr_safe = np.where(i_fluxerr_2d > 0, i_fluxerr_2d, 1e10)
        
        g_weights = 1.0 / (g_fluxerr_safe ** 2)
        r_weights = 1.0 / (r_fluxerr_safe ** 2)
        i_weights = 1.0 / (i_fluxerr_safe ** 2)
        
        LOG.info("Using flux errors as inverse-variance weights")
        
        # Remove baselines
        g_flux_residuals = remove_polynomial_baseline_gpufit(
            g_flux_2d, g_bjd, weights=g_weights, tolerance=1e-3, max_iterations=5000, verbose=True
        )
        r_flux_residuals = remove_polynomial_baseline_gpufit(
            r_flux_2d, r_bjd, weights=r_weights, tolerance=1e-3, max_iterations=5000, verbose=True
        )
        i_flux_residuals = remove_polynomial_baseline_gpufit(
            i_flux_2d, i_bjd, weights=i_weights, tolerance=1e-3, max_iterations=5000, verbose=True
        )
        
        # Flatten back
        g_flux = g_flux_residuals.flatten()
        r_flux = r_flux_residuals.flatten()
        i_flux = i_flux_residuals.flatten()
        
        LOG.info("Baseline removal complete")
        gc.collect()

    #Build quick lookup maps from gaia_id->index to avoid np.where inside loop
    g_map = {int(k): idx for idx, k in enumerate(g_psid)}
    r_map = {int(k): idx for idx, k in enumerate(r_psid)}
    i_map = {int(k): idx for idx, k in enumerate(i_psid)}
    LOG.debug("Built g_map (%d), r_map (%d), i_map (%d)", len(g_map), len(r_map), len(i_map))

    #Find unique gaia_ids (psids)
    keys = sorted(g_map.keys() | r_map.keys() | i_map.keys())
    unique_ids = np.array(keys, dtype=g_psid.dtype)
    LOG.info("Total unique sources to combine: %d", len(unique_ids))

    #Combine the data
    comb_jd = np.concatenate((g_jd, r_jd, i_jd))
    comb_bjd = np.concatenate((g_bjd, r_bjd, i_bjd))
    comb_filterid = np.concatenate((g_filterid, r_filterid, i_filterid))
    comb_exptime = np.concatenate((g_exptime, r_exptime, i_exptime))
    comb_pid = np.concatenate((g_pid, r_pid, i_pid))
    comb_field = np.concatenate((g_field, r_field, i_field))
    comb_ccd = np.concatenate((g_ccd, r_ccd, i_ccd))
    comb_quad = np.concatenate((g_quad, r_quad, i_quad))
    comb_imstat = np.concatenate((g_imstat, r_imstat, i_imstat))
    comb_infobits = np.concatenate((g_infobits, r_infobits, i_infobits))
    comb_seeing = np.concatenate((g_seeing, r_seeing, i_seeing))
    comb_mzpsci = np.concatenate((g_mzpsci, r_mzpsci, i_mzpsci))
    comb_mzpsciunc = np.concatenate((g_mzpsciunc, r_mzpsciunc, i_mzpsciunc))
    comb_mzpscirms = np.concatenate((g_mzpscirms, r_mzpscirms, i_mzpscirms))
    comb_clrco = np.concatenate((g_clrco, r_clrco, i_clrco))
    comb_clrcounc = np.concatenate((g_clrcounc, r_clrcounc, i_clrcounc))
    comb_maglim = np.concatenate((g_maglim, r_maglim, i_maglim))
    comb_airmass = np.concatenate((g_airmass, r_airmass, i_airmass))
    comb_nps1matches = np.concatenate((g_nps1matches, r_nps1matches, i_nps1matches))

    comb_ra = np.zeros(len(unique_ids), dtype=g_ra.dtype)
    comb_dec = np.zeros(len(unique_ids), dtype=g_dec.dtype)
    comb_mag_ref = np.zeros(len(unique_ids), dtype=g_mag_ref.dtype)
    comb_mag_err_ref = np.zeros(len(unique_ids), dtype=g_mag_err_ref.dtype)
    comb_objtype = np.zeros(len(unique_ids), dtype=g_objtype.dtype)

    g_nexp = len(g_bjd)
    r_nexp = len(r_bjd)
    i_nexp = len(i_bjd)

    comb_flux = np.zeros((len(unique_ids), g_nexp + r_nexp + i_nexp), dtype=g_flux.dtype)
    comb_fluxerr = np.zeros((len(unique_ids), g_nexp + r_nexp + i_nexp), dtype=g_fluxerr.dtype)
    comb_flag = np.zeros((len(unique_ids), g_nexp + r_nexp + i_nexp), dtype=g_flag.dtype)

    total = len(unique_ids)
    progress_interval = max(1, total // 10)  # log 10 times across the run

    for uid_idx, psid in enumerate(unique_ids):
        pid = int(psid)
        ra = 0; dec = 0; mag_ref = 0; mag_err_ref = 0; objtype = 0

        # g band
        if pid in g_map:
            g_index = g_map[pid]
            start = g_index * g_nexp
            comb_flux[uid_idx, :g_nexp] = g_flux[start: start + g_nexp]
            comb_fluxerr[uid_idx, :g_nexp] = g_fluxerr[start: start + g_nexp]
            comb_flag[uid_idx, :g_nexp] = g_flag[start: start + g_nexp]
            ra = g_ra[g_index]; dec = g_dec[g_index]
            mag_ref = g_mag_ref[g_index]; mag_err_ref = g_mag_err_ref[g_index]
            objtype = g_objtype[g_index]

        # r band
        if pid in r_map:
            r_index = r_map[pid]
            start = r_index * r_nexp
            comb_flux[uid_idx, g_nexp:g_nexp + r_nexp] = r_flux[start: start + r_nexp]
            comb_fluxerr[uid_idx, g_nexp:g_nexp + r_nexp] = r_fluxerr[start: start + r_nexp]
            comb_flag[uid_idx, g_nexp:g_nexp + r_nexp] = r_flag[start: start + r_nexp]
            if pid not in g_map:
                ra = r_ra[r_index]; dec = r_dec[r_index]
                mag_ref = r_mag_ref[r_index]; mag_err_ref = r_mag_err_ref[r_index]
                objtype = r_objtype[r_index]

        # i band
        if pid in i_map:
            i_index = i_map[pid]
            start = i_index * i_nexp
            comb_flux[uid_idx, g_nexp + r_nexp:] = i_flux[start: start + i_nexp]
            comb_fluxerr[uid_idx, g_nexp + r_nexp:] = i_fluxerr[start: start + i_nexp]
            comb_flag[uid_idx, g_nexp + r_nexp:] = i_flag[start: start + i_nexp]
            if (pid not in g_map) and (pid not in r_map):
                ra = i_ra[i_index]; dec = i_dec[i_index]
                mag_ref = i_mag_ref[i_index]; mag_err_ref = i_mag_err_ref[i_index]
                objtype = i_objtype[i_index]

        comb_ra[uid_idx] = ra
        comb_dec[uid_idx] = dec
        comb_mag_ref[uid_idx] = mag_ref
        comb_mag_err_ref[uid_idx] = mag_err_ref
        comb_objtype[uid_idx] = objtype

        if uid_idx % progress_interval == 0:
            LOG.info("Processed %d / %d sources", uid_idx, total)
    LOG.info("Finished combining sources; sorting exposures and building datasets")

    # Argsort everything by time
    sort_idx = np.argsort(comb_bjd)
    comb_jd_sorted = comb_jd[sort_idx]
    comb_bjd_sorted = comb_bjd[sort_idx]
    comb_filterid_sorted = comb_filterid[sort_idx]
    comb_exptime_sorted = comb_exptime[sort_idx]
    comb_pid_sorted = comb_pid[sort_idx]
    comb_field_sorted = comb_field[sort_idx]
    comb_ccd_sorted = comb_ccd[sort_idx]
    comb_quad_sorted = comb_quad[sort_idx]
    comb_imstat_sorted = comb_imstat[sort_idx]
    comb_infobits_sorted = comb_infobits[sort_idx]
    comb_seeing_sorted = comb_seeing[sort_idx]
    comb_mzpsci_sorted = comb_mzpsci[sort_idx]
    comb_mzpsciunc_sorted = comb_mzpsciunc[sort_idx]
    comb_mzpscirms_sorted = comb_mzpscirms[sort_idx]
    comb_clrco_sorted = comb_clrco[sort_idx]
    comb_clrcounc_sorted = comb_clrcounc[sort_idx]
    comb_maglim_sorted = comb_maglim[sort_idx]
    comb_airmass_sorted = comb_airmass[sort_idx]
    comb_nps1matches_sorted = comb_nps1matches[sort_idx]

    comb_flux_sorted = comb_flux[:, sort_idx]
    comb_fluxerr_sorted = comb_fluxerr[:, sort_idx]
    comb_flag_sorted = comb_flag[:, sort_idx]

    # Write out the combined data to a new HDF5 file
    LOG.debug("Writing output file: %s", output_file)
    exposures_dtype = np.dtype([
        ("jd", g_jd.dtype),
        ("bjd", g_bjd.dtype),
        ("filterid", g_filterid.dtype),
        ("exptime", g_exptime.dtype),
        ("pid", g_pid.dtype),
        ("field", g_field.dtype),
        ("ccd", g_ccd.dtype),
        ("quad", g_quad.dtype),
        ("imstat", g_imstat.dtype),
        ("infobits", g_infobits.dtype),
        ("seeing", g_seeing.dtype),
        ("mzpsci", g_mzpsci.dtype),
        ("mzpsciunc", g_mzpsciunc.dtype),
        ("mzpscirms", g_mzpscirms.dtype),
        ("clrco", g_clrco.dtype),
        ("clrcounc", g_clrcounc.dtype),
        ("maglim", g_maglim.dtype),
        ("airmass", g_airmass.dtype),
        ("nps1matches", g_nps1matches.dtype)
    ])
    exposures_table = np.zeros(len(comb_bjd_sorted), dtype=exposures_dtype)
    exposures_table["jd"] = comb_jd_sorted
    exposures_table["bjd"] = comb_bjd_sorted
    exposures_table["filterid"] = comb_filterid_sorted
    exposures_table["exptime"] = comb_exptime_sorted
    exposures_table["pid"] = comb_pid_sorted
    exposures_table["field"] = comb_field_sorted
    exposures_table["ccd"] = comb_ccd_sorted
    exposures_table["quad"] = comb_quad_sorted
    exposures_table["imstat"] = comb_imstat_sorted
    exposures_table["infobits"] = comb_infobits_sorted
    exposures_table["seeing"] = comb_seeing_sorted
    exposures_table["mzpsci"] = comb_mzpsci_sorted
    exposures_table["mzpsciunc"] = comb_mzpsciunc_sorted
    exposures_table["mzpscirms"] = comb_mzpscirms_sorted
    exposures_table["clrco"] = comb_clrco_sorted
    exposures_table["clrcounc"] = comb_clrcounc_sorted
    exposures_table["maglim"] = comb_maglim_sorted
    exposures_table["airmass"] = comb_airmass_sorted
    exposures_table["nps1matches"] = comb_nps1matches_sorted

    sources_dtype = np.dtype([
        ("gaia_id", g_psid.dtype),
        ("ra", g_ra.dtype),
        ("decl", g_dec.dtype),
        ("mag_ref", g_mag_ref.dtype),
        ("mag_err_ref", g_mag_err_ref.dtype),
        ("objtype", g_objtype.dtype)
    ])
    sources_table = np.zeros(len(unique_ids), dtype=sources_dtype)
    sources_table["gaia_id"] = unique_ids
    sources_table["ra"] = comb_ra
    sources_table["decl"] = comb_dec
    sources_table["mag_ref"] = comb_mag_ref
    sources_table["mag_err_ref"] = comb_mag_err_ref
    sources_table["objtype"] = comb_objtype

    n_total = comb_flux_sorted.size
    sourcedata_dtype = np.dtype([
        ('flux', comb_flux_sorted.dtype),
        ('flux_err', comb_fluxerr_sorted.dtype),
        ('flag', comb_flag_sorted.dtype)
    ])
    sourcedata_table = np.zeros(n_total, dtype=sourcedata_dtype)
    sourcedata_table["flux"] = comb_flux_sorted.flatten()
    sourcedata_table["flux_err"] = comb_fluxerr_sorted.flatten()
    sourcedata_table["flag"] = comb_flag_sorted.flatten()

    with h5py.File(output_file, "w") as f:
        # Create groups as in the original files
        data_group = f.create_group("data")
        data_group.create_dataset("sources", data=sources_table, compression="gzip", compression_opts=1)
        data_group.create_dataset("exposures", data=exposures_table, compression="gzip", compression_opts=1)
        data_group.create_dataset("sourcedata", data=sourcedata_table, compression="gzip", compression_opts=1)
    
    LOG.info("merge_g_r_i complete g_file=%s r_file=%s i_file=%s output=%s", 
             g_filename, r_filename, i_filename, output_file)

def which_function(file_list):
    """
    Determine which merge function to use based on available files in a group.
    Args:
        file_list: list of filenames (basenames) in the group
    Returns:
        tuple: (function, g_filename, r_filename, i_filename) or (None, None, None, None)
    """
    has_g = any('_zg.h5' in f or '_g.cor.h5' in f for f in file_list)
    has_r = any('_zr.h5' in f or '_r.cor.h5' in f for f in file_list)
    has_i = any('_zi.h5' in f or '_i.cor.h5' in f for f in file_list)
    
    g_file = next((f for f in file_list if '_zg.h5' in f or '_g.cor.h5' in f), None)
    r_file = next((f for f in file_list if '_zr.h5' in f or '_r.cor.h5' in f), None)
    i_file = next((f for f in file_list if '_zi.h5' in f or '_i.cor.h5' in f), None)
    
    # if has_g and has_r and has_i:
    #     return merge_g_r_i, g_file, r_file, i_file
    if has_g and has_r:
        return merge_g_r, g_file, r_file, None
    elif has_g:
        return copy_single_file, g_file, None, None
    elif has_r:
        return copy_single_file, r_file, None, None
    else:
        return None, None, None, None

def process_file_group(input_path, output_path, file_group, remove_baseline=False):
    """
    Process a group of related files (g/r/i band variants).
    Args:
        input_path: directory with files
        output_path: output directory
        file_group: list of filenames to process together
        remove_baseline: if True, apply GPU baseline removal
    """
    try:
        LOG.info("Processing file group: %s", file_group)
        func, g_file, r_file, i_file = which_function(file_group)
        
        if func is None:
            LOG.warning("No g or r files found in group %s. Skipping.", file_group)
            return True
        
        if func == copy_single_file:
            # Single band file (g or r only)
            filename = g_file if g_file else r_file
            copy_single_file(input_path, output_path, filename, remove_baseline=remove_baseline)
        elif func == merge_g_r:
            merge_g_r(input_path, output_path, g_file, r_file, remove_baseline=remove_baseline)
        elif func == merge_g_r_i:
            merge_g_r_i(input_path, output_path, g_file, r_file, i_file, remove_baseline=remove_baseline)
        
        LOG.info("Completed processing file group: %s", file_group)
        return True
    except Exception as e:
        LOG.error("Error processing file group %s: %s", file_group, str(e))
        return False

def process_field(field_dir, output_path, field_str, n_threads=4, remove_baseline=False):
    """
    Scan input_path for zforce/imdiff files, group g/r/i variants, and process.
    """

    input_path = os.path.join(field_dir, field_str)
    output_path = os.path.join(output_path, field_str)
    os.makedirs(output_path, exist_ok=True)

    groups = look_for_files(input_path)
    LOG.info("Found %d file groups in field %s", len(groups), field_str)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_threads, mp_context=ctx) as executor:
        futures = {
            executor.submit(process_file_group, input_path, output_path, file_list, remove_baseline): base
            for base, file_list in groups.items()
        }

        try:
            for future in as_completed(futures):
                base = futures[future]
                future.result()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received — cancelling remaining tasks and shutting down workers...", file=sys.stderr)
            # Try to cancel pending futures
            for f in futures:
                try:
                    f.cancel()
                except Exception:
                    pass
            executor.shutdown(wait=False)
            raise

def process_all_imdiff_fields(field_dir, output_path, n_threads=4, remove_baseline=False):
    """
    Process all fields sequentially; for each field submit CCD/quad tasks in parallel
    (up to n_threads) and wait until that field's tasks finish before moving to the
    next field. Uses a spawn context and handles KeyboardInterrupt to avoid zombies.
    """
    fields = np.arange(245, 882)
    ctx = multiprocessing.get_context("spawn")

    for field in fields:
        field_str = f"{int(field):04d}"
        LOG.info("Starting work for field %s", field_str)
        try:
            process_field(field_dir, output_path, field_str, n_threads=n_threads, remove_baseline=remove_baseline)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received — aborting all processing.", file=sys.stderr)
            raise
        LOG.info("Completed all tasks for field %s", field_str)

def process_subset_imdiff_fields(field_dir, output_path, lower_field, upper_field, n_threads=4, remove_baseline=False):
    fields = np.arange(lower_field, upper_field + 1)

    for field in fields:
        field_str = f"{int(field):04d}"
        LOG.info("Starting work for field %s", field_str)
        try:
            process_field(field_dir, output_path, field_str, n_threads=n_threads, remove_baseline=remove_baseline)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received — aborting all processing.", file=sys.stderr)
            raise
        LOG.info("Completed all tasks for field %s", field_str)

def clamp_n_threads(n):
    try:
        n = int(n)
    except Exception:
        return 1
    if n < 1:
        return 1
    # don't go crazy with threads by default; user can override
    max_allowed = max(1, min(32, (os.cpu_count() or 4)))
    return min(n, max_allowed)

if __name__ == "__main__":
    # Ensure 'spawn' start method to avoid HDF5/h5py deadlocks
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--input_path",
        type=str,
        default="/data/zvar/matchfiles/",
        help="Root directory containing field subdirectories with matchfiles inside",
    )
    parent_parser.add_argument(
        "--output_path",
        type=str,
        default="/data/zvar/comb_matchfiles/",
        help="Root directory for combined outputs (comb_*.h5 under each field subdir)",
    )
    parent_parser.add_argument(
        "--remove_baseline",
        action="store_true",
        help="Remove baseline from data (requires pygpufit)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_field = subparsers.add_parser(
        "process_field", parents=[parent_parser], help="Process a single field"
    )
    # parser_field.add_argument("field", type=int, help="Field number")
    parser_field.add_argument("field", type=str, help="Field identifier (e.g. 0450 or F0450)")
    parser_field.add_argument(
        "--n_threads", type=int, default=4, help="Number of worker processes (default: 4)"
    )

    parser_all = subparsers.add_parser(
        "process_all_fields", parents=[parent_parser], help="Process all fields"
    )
    parser_all.add_argument(
        "--n_threads", type=int, default=4, help="Number of worker processes (default: 4)"
    )

    parser_subset = subparsers.add_parser(
        "process_subset_fields",
        parents=[parent_parser],
        help="Process fields in an inclusive range",
    )
    parser_subset.add_argument("lower_field", type=int, help="Lower field number (inclusive)")
    parser_subset.add_argument("upper_field", type=int, help="Upper field number (inclusive)")
    parser_subset.add_argument(
        "--n_threads", type=int, default=4, help="Number of worker processes (default: 4)"
    )

    args = parser.parse_args()

    if args.remove_baseline:
        try:
            import pygpufit.gpufit as gf  # noqa: F401
            LOG.info("pygpufit imported successfully for baseline removal")
        except ImportError:
            LOG.error("Failed to import pygpufit. Please install it to use --remove_baseline")
            print("ERROR: pygpufit is required when using --remove_baseline flag.", file=sys.stderr)
            sys.exit(1)

    try:
        if args.command == "process_field":
            n = clamp_n_threads(args.n_threads)
            process_field(
                args.input_path, args.output_path, args.field,
                n_threads=n, remove_baseline=args.remove_baseline
            )
        elif args.command == "process_all_fields":
            n = clamp_n_threads(args.n_threads)
            process_all_imdiff_fields(
                args.input_path, args.output_path, n_threads=n, remove_baseline=args.remove_baseline
            )
        elif args.command == "process_subset_fields":
            n = clamp_n_threads(args.n_threads)
            process_subset_imdiff_fields(
                args.input_path, args.output_path,
                args.lower_field, args.upper_field,
                n_threads=n, remove_baseline=args.remove_baseline
            )
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user — exiting.", file=sys.stderr)
        try:
            os.killpg(os.getpgid(0), signal.SIGTERM)
        except Exception:
            pass
        sys.exit(1)