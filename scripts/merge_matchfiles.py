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

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(processName)s/%(threadName)s] %(message)s"
if os.environ.get("ZVAR_VERBOSE", "").lower() in ("1", "true", "yes"):
    logging.basicConfig(level=logging.DEBUG, format=_LOG_FORMAT)
else:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
LOG = logging.getLogger("merge_matchfiles")

def look_for_files(input_path, field, ccd, quad):
    pattern = f'{input_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_*.h5'
    files = glob.glob(pattern)
    LOG.debug("look_for_files pattern=%s -> %d files", pattern, len(files))
    return files

def ensure_output_dir(output_path, field):
    """Create the output directory for a given field if it doesn't exist."""
    dirpath = os.path.join(output_path, f"{field:04d}")
    os.makedirs(dirpath, exist_ok=True)

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

def copy_g(input_path, output_path, field, ccd, quad):
    import h5py  # lazy import
    LOG.info("copy_g field=%d ccd=%d quad=%d", field, ccd, quad)
    g_filename = f'{input_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zg.h5'
    output_filename = f'{output_path}/{field:04d}/comb_data_{field:04d}_{ccd:02d}_{quad:01d}.h5'
    ensure_output_dir(output_path, field)
    copy_h5_file(g_filename, output_filename)
    LOG.info("copy_g done field=%d ccd=%d quad=%d", field, ccd, quad)

def copy_r(input_path, output_path, field, ccd, quad):
    import h5py  # lazy import
    LOG.info("copy_r field=%d ccd=%d quad=%d", field, ccd, quad)
    r_filename = f'{input_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zr.h5'
    output_filename = f'{output_path}/{field:04d}/comb_data_{field:04d}_{ccd:02d}_{quad:01d}.h5'
    ensure_output_dir(output_path, field)
    copy_h5_file(r_filename, output_filename)
    LOG.info("copy_r done field=%d ccd=%d quad=%d", field, ccd, quad)

def merge_g_r(input_path, output_path, field, ccd, quad):
    import h5py  # lazy import
    LOG.info("merge_g_r start field=%d ccd=%d quad=%d", field, ccd, quad)
    g_filename = f'{input_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zg.h5'
    r_filename = f'{input_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zr.h5'
    output_filename = f'{output_path}/{field:04d}/comb_data_{field:04d}_{ccd:02d}_{quad:01d}.h5'
    ensure_output_dir(output_path, field)

    # Load relevant data from each file
    LOG.debug("Opening g file: %s", g_filename)
    with h5py.File(g_filename, "r") as f:
        g_jd = f["/data/exposures"]["jd"][:]
        g_bjd = f["/data/exposures"]["bjd"][:]
        g_filterid = f["/data/exposures"]["filterid"][:]
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

    LOG.debug("Opening r file: %s", r_filename)
    with h5py.File(r_filename, "r") as f:
        r_jd = f["/data/exposures"]["jd"][:]
        r_bjd = f["/data/exposures"]["bjd"][:]
        r_filterid = f["/data/exposures"]["filterid"][:]
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

    # Find unique gaia_ids (psids)
    # Build quick lookup maps from gaia_id->index to avoid np.where inside loop
    # Convert keys to python int to make lookups faster and robust
    g_map = {int(k): idx for idx, k in enumerate(g_psid)}
    r_map = {int(k): idx for idx, k in enumerate(r_psid)}
    LOG.debug("Built g_map (%d) and r_map (%d)", len(g_map), len(r_map))

    keys = sorted(g_map.keys() | r_map.keys())
    unique_ids = np.array(keys, dtype=g_psid.dtype)
    LOG.info("Total unique sources to combine: %d", len(unique_ids))

    # Combine exposures metadata
    comb_jd = np.concatenate((g_jd, r_jd))
    comb_bjd = np.concatenate((g_bjd, r_bjd))
    comb_filterid = np.concatenate((g_filterid, r_filterid))
    comb_exptime = np.concatenate((g_exptime, r_exptime))
    comb_pid = np.concatenate((g_pid, r_pid))
    comb_field = np.concatenate((g_field, r_field))
    comb_ccd = np.concatenate((g_ccd, r_ccd))
    comb_quad = np.concatenate((g_quad, r_quad))
    comb_imstat = np.concatenate((g_imstat, r_imstat))
    comb_infobits = np.concatenate((g_infobits, r_infobits))
    comb_seeing = np.concatenate((g_seeing, r_seeing))
    comb_mzpsci = np.concatenate((g_mzpsci, r_mzpsci))
    comb_mzpsciunc = np.concatenate((g_mzpsciunc, r_mzpsciunc))
    comb_mzpscirms = np.concatenate((g_mzpscirms, r_mzpscirms))
    comb_clrco = np.concatenate((g_clrco, r_clrco))
    comb_clrcounc = np.concatenate((g_clrcounc, r_clrcounc))
    comb_maglim = np.concatenate((g_maglim, r_maglim))
    comb_airmass = np.concatenate((g_airmass, r_airmass))
    comb_nps1matches = np.concatenate((g_nps1matches, r_nps1matches))

    # Prepare output arrays
    comb_ra = np.zeros(len(unique_ids), dtype=g_ra.dtype)
    comb_dec = np.zeros(len(unique_ids), dtype=g_dec.dtype)
    comb_mag_ref = np.zeros(len(unique_ids), dtype=g_mag_ref.dtype)
    comb_mag_err_ref = np.zeros(len(unique_ids), dtype=g_mag_err_ref.dtype)
    comb_objtype = np.zeros(len(unique_ids), dtype=g_objtype.dtype)

    g_nexp = len(g_bjd)
    r_nexp = len(r_bjd)

    comb_flux = np.zeros((len(unique_ids), g_nexp + r_nexp), dtype=g_flux.dtype)
    comb_fluxerr = np.zeros((len(unique_ids), g_nexp + r_nexp), dtype=g_fluxerr.dtype)
    comb_flag = np.zeros((len(unique_ids), g_nexp + r_nexp), dtype=g_flag.dtype)

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
            comb_flag[uid_idx, :g_nexp] = g_flag[start: start + g_nexp]

            ra = g_ra[g_index]
            dec = g_dec[g_index]
            mag_ref = g_mag_ref[g_index]
            mag_err_ref = g_mag_err_ref[g_index]
            objtype = g_objtype[g_index]

        if pid in r_map:
            r_index = r_map[pid]
            start = r_index * r_nexp
            comb_flux[uid_idx, g_nexp:] = r_flux[start: start + r_nexp]
            comb_fluxerr[uid_idx, g_nexp:] = r_fluxerr[start: start + r_nexp]
            comb_flag[uid_idx, g_nexp:] = r_flag[start: start + r_nexp]

            # if not present in g, take r position (otherwise g already set)
            if pid not in g_map:
                ra = r_ra[r_index]
                dec = r_dec[r_index]
                mag_ref = r_mag_ref[r_index]
                mag_err_ref = r_mag_err_ref[r_index]
                objtype = r_objtype[r_index]

        comb_ra[uid_idx] = ra
        comb_dec[uid_idx] = dec
        comb_mag_ref[uid_idx] = mag_ref
        comb_mag_err_ref[uid_idx] = mag_err_ref
        comb_objtype[uid_idx] = objtype

        if uid_idx % progress_interval == 0:
            LOG.info("merge_g_r progress field=%d ccd=%d quad=%d: %d/%d", field, ccd, quad, uid_idx, total)
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

    comb_ra_sorted = comb_ra[sort_idx]
    comb_dec_sorted = comb_dec[sort_idx]
    comb_mag_ref_sorted = comb_mag_ref[sort_idx]
    comb_mag_err_ref_sorted = comb_mag_err_ref[sort_idx]
    comb_objtype_sorted = comb_objtype[sort_idx]

    comb_flux_sorted = comb_flux[:, sort_idx]
    comb_fluxerr_sorted = comb_fluxerr[:, sort_idx]
    comb_flag_sorted = comb_flag[:, sort_idx]

    # Write out the combined data to a new HDF5 file
    LOG.debug("Writing output file: %s", output_filename)
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

    with h5py.File(output_filename, "w") as f:
        # Create groups as in the original files
        data_group = f.create_group("data")
        data_group.create_dataset("sources", data=sources_table, compression="gzip", compression_opts=1)
        data_group.create_dataset("exposures", data=exposures_table, compression="gzip", compression_opts=1)
        data_group.create_dataset("sourcedata", data=sourcedata_table, compression="gzip", compression_opts=1)
    LOG.info("merge_g_r complete field=%d ccd=%d quad=%d written=%s", field, ccd, quad, output_filename)

def merge_g_r_i(input_path, output_path, field, ccd, quad):
    import h5py  # lazy import
    LOG.info("merge_g_r_i start field=%d ccd=%d quad=%d", field, ccd, quad)
    g_filename = f'{input_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zg.h5'
    r_filename = f'{input_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zr.h5'
    i_filename = f'{input_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zi.h5'
    output_filename = f'{output_path}/{field:04d}/comb_data_{field:04d}_{ccd:02d}_{quad:01d}.h5'
    ensure_output_dir(output_path, field)

    #Load relevant data from each file
    LOG.debug("Opening g file: %s", g_filename)
    with h5py.File(g_filename, "r") as f:
        g_jd = f["/data/exposures"]["jd"][:]
        g_bjd = f["/data/exposures"]["bjd"][:]
        g_filterid = f["/data/exposures"]["filterid"][:]
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

    LOG.debug("Opening r file: %s", r_filename)
    with h5py.File(r_filename, "r") as f:
        r_jd = f["/data/exposures"]["jd"][:]
        r_bjd = f["/data/exposures"]["bjd"][:]
        r_filterid = f["/data/exposures"]["filterid"][:]
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

    LOG.debug("Opening i file: %s", i_filename)
    with h5py.File(i_filename, "r") as f:
        i_jd = f["/data/exposures"]["jd"][:]
        i_bjd = f["/data/exposures"]["bjd"][:]
        i_filterid = f["/data/exposures"]["filterid"][:]
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

    comb_flux = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd) + len(i_bjd)), dtype=g_flux.dtype)
    comb_fluxerr = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd) + len(i_bjd)), dtype=g_fluxerr.dtype)
    comb_flag = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd) + len(i_bjd)), dtype=g_flag.dtype)

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
            LOG.info("merge_g_r_i progress field=%d ccd=%d quad=%d: %d/%d", field, ccd, quad, uid_idx, total)
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

    comb_ra_sorted = comb_ra[sort_idx]
    comb_dec_sorted = comb_dec[sort_idx]
    comb_mag_ref_sorted = comb_mag_ref[sort_idx]
    comb_mag_err_ref_sorted = comb_mag_err_ref[sort_idx]
    comb_objtype_sorted = comb_objtype[sort_idx]

    comb_flux_sorted = comb_flux[:, sort_idx]
    comb_fluxerr_sorted = comb_fluxerr[:, sort_idx]
    comb_flag_sorted = comb_flag[:, sort_idx]

    # Write out the combined data to a new HDF5 file
    LOG.debug("Writing output file: %s", output_filename)
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


    with h5py.File(output_filename, "w") as f:
        # Create groups as in the original files
        data_group = f.create_group("data")
        data_group.create_dataset("sources", data=sources_table, compression="gzip", compression_opts=1)
        data_group.create_dataset("exposures", data=exposures_table, compression="gzip", compression_opts=1)
        data_group.create_dataset("sourcedata", data=sourcedata_table, compression="gzip", compression_opts=1)
    LOG.info("merge_g_r_i complete field=%d ccd=%d quad=%d written=%s", field, ccd, quad, output_filename)

def which_function(input_path, field, ccd, quad):
    files = look_for_files(input_path, field, ccd, quad)
    has_g = any('_zg.h5' in f for f in files)
    has_r = any('_zr.h5' in f for f in files)
    has_i = any('_zi.h5' in f for f in files)

    if has_g and has_r and has_i:
        return merge_g_r_i
    elif has_g and has_r:
        return merge_g_r
    elif has_g:
        return copy_g
    elif has_r:
        return copy_r
    else:
        return None

def process_file(input_path, output_path, field, ccd, quad):
    print(f"Processing field {field}, ccd {ccd}, quad {quad} on thread {threading.get_ident()}")
    func = which_function(input_path, field, ccd, quad)
    if func:
        try:
            func(input_path, output_path, field, ccd, quad)
            print(f"Completed processing for field {field}, ccd {ccd}, quad {quad}.")
        except Exception as e:
            # Log error and continue other tasks
            print(f"Error processing field {field}, ccd {ccd}, quad {quad}: {e}", file=sys.stderr)
    else:
        print(f"No g or r files found for field {field}, ccd {ccd}, quad {quad}. Skipping.")

def process_field(input_path, output_path, field):
    if field < 245 or field > 881:
        print("Field number must be between 245 and 881.")
        return
    ccds = range(1, 17)      # CCDs 1 to 16
    quads = range(1, 5)      # Quads 1 to 4

    for ccd in ccds:
        for quad in quads:
            process_file(input_path, output_path, field, ccd, quad)

def process_all_fields(input_path, output_path, n_threads=4):
    """
    Process all fields sequentially; for each field submit CCD/quad tasks in parallel
    (up to n_threads) and wait until that field's tasks finish before moving to the
    next field. Uses a spawn context and handles KeyboardInterrupt to avoid zombies.
    """
    fields = np.arange(245, 882)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_threads, mp_context=ctx) as executor:
        try:
            for field in fields:
                LOG.info("Starting work for field %d", int(field))
                # build per-field tasks (CCD 1..16, quad 1..4)
                futures = []
                for ccd in range(1, 17):
                    for quad in range(1, 5):
                        # submit the per-file work; process_file will skip if no file present
                        futures.append(executor.submit(process_file, input_path, output_path, int(field), ccd, quad))

                # wait for this field's tasks to finish before moving on
                try:
                    for future in as_completed(futures):
                        # re-raise exceptions from workers
                        future.result()
                except KeyboardInterrupt:
                    print("KeyboardInterrupt received — cancelling remaining tasks for current field and shutting down workers...", file=sys.stderr)
                    # try to cancel pending futures
                    for f in futures:
                        try:
                            f.cancel()
                        except Exception:
                            pass
                    # ask executor to shutdown quickly; context manager will finalize
                    executor.shutdown(wait=False)
                    raise
                LOG.info("Completed all tasks for field %d", int(field))
        except KeyboardInterrupt:
            # top-level handler: ensure children are terminated
            print("KeyboardInterrupt received — aborting all fields.", file=sys.stderr)
            raise


def process_subset_fields(input_path, output_path, lower_field, upper_field, n_threads=4):
    """
    Process fields in the inclusive range [lower_field, upper_field].
    For each field submit CCD/quad tasks in parallel (up to n_threads) and wait for
    that field's tasks to finish before moving to the next field.
    """
    fields = np.arange(lower_field, upper_field + 1)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_threads, mp_context=ctx) as executor:
        try:
            for field in fields:
                LOG.info("Starting work for field %d", int(field))
                futures = []
                for ccd in range(1, 17):
                    for quad in range(1, 5):
                        futures.append(executor.submit(process_file, input_path, output_path, int(field), ccd, quad))

                try:
                    for future in as_completed(futures):
                        future.result()
                except KeyboardInterrupt:
                    print("KeyboardInterrupt received — cancelling remaining tasks for current field and shutting down workers...", file=sys.stderr)
                    for f in futures:
                        try:
                            f.cancel()
                        except Exception:
                            pass
                    executor.shutdown(wait=False)
                    raise
                LOG.info("Completed all tasks for field %d", int(field))
        except KeyboardInterrupt:
            print("KeyboardInterrupt received — aborting subset processing.", file=sys.stderr)
            raise

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
        # start method already set by parent environment; ignore
        pass

    parser = argparse.ArgumentParser(description="Merge or copy ZTF matchfiles.")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--input_path", type=str, default="/data/zvar/matchfiles/",
                               help="Input path for matchfiles")
    parent_parser.add_argument("--output_path", type=str, default="/data/zvar/comb_matchfiles/",
                               help="Output path for combined matchfiles")

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_field = subparsers.add_parser("process_field", parents=[parent_parser],
                                         help="Process a single field")
    parser_field.add_argument("field", type=int, help="Field number")

    parser_all = subparsers.add_parser("process_all_fields", parents=[parent_parser],
                                       help="Process all fields")
    parser_all.add_argument("--n_threads", type=int, default=4,
                            help="Number of worker processes (default: 4)")

    parser_subset = subparsers.add_parser("process_subset_fields", parents=[parent_parser],
                                          help="Process fields in an inclusive range")
    parser_subset.add_argument("lower_field", type=int, help="Lower field number (inclusive)")
    parser_subset.add_argument("upper_field", type=int, help="Upper field number (inclusive)")
    parser_subset.add_argument("--n_threads", type=int, default=4,
                               help="Number of worker processes (default: 4)")

    args = parser.parse_args()

    # wrap top-level calls so Ctrl+C is handled more predictably
    try:
        if args.command == "process_field":
            process_field(args.input_path, args.output_path, args.field)
        elif args.command == "process_all_fields":
            n = clamp_n_threads(args.n_threads)
            process_all_fields(args.input_path, args.output_path, n)
        elif args.command == "process_subset_fields":
            n = clamp_n_threads(args.n_threads)
            process_subset_fields(args.input_path, args.output_path,
                                  args.lower_field, args.upper_field, n)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user — exiting.", file=sys.stderr)
        # Attempt to terminate child processes in the process group
        try:
            # send TERM to process group; this kills children too
            os.killpg(os.getpgid(0), signal.SIGTERM)
        except Exception:
            pass
        sys.exit(1)