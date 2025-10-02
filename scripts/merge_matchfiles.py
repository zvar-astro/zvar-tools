import h5py
import numpy as np
import glob
import argparse
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

def look_for_files(field, ccd, quad):
    pattern = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_*.h5'
    files = glob.glob(pattern)
    return files

def copy_h5_file(src, dst):
    """
    Copy an HDF5 file from src to dst.
    """
    shutil.copy2(src, dst)

def copy_g(field, ccd, quad):
    g_filename = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zg.h5'
    output_filename = f'/data/zvar/comb_matchfiles/{field:04d}/comb_data_{field:04d}_{ccd:02d}_{quad:01d}.h5'
    copy_h5_file(g_filename, output_filename)

def copy_r(field, ccd, quad):
    r_filename = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zr.h5'
    output_filename = f'/data/zvar/comb_matchfiles/{field:04d}/comb_data_{field:04d}_{ccd:02d}_{quad:01d}.h5'
    copy_h5_file(r_filename, output_filename)

def merge_g_r(field, ccd, quad):
    g_filename = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zg.h5'
    r_filename = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zr.h5'
    output_filename = f'/data/zvar/comb_matchfiles/{field:04d}/comb_data_{field:04d}_{ccd:02d}_{quad:01d}.h5'

    #Load relevant data from each file
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

    #Find unique gaia_ids (psids)
    unique_ids = np.union1d(g_psid, r_psid)  # Returns sorted unique values, dtype preserved
    
    #Combine the data
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

    comb_ra = np.zeros(len(unique_ids), dtype=g_ra.dtype)
    comb_dec = np.zeros(len(unique_ids), dtype=g_dec.dtype)
    comb_mag_ref = np.zeros(len(unique_ids), dtype=g_mag_ref.dtype)
    comb_mag_err_ref = np.zeros(len(unique_ids), dtype=g_mag_err_ref.dtype)
    comb_objtype = np.zeros(len(unique_ids), dtype=g_objtype.dtype)

    comb_flux = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd)), dtype=g_flux.dtype)
    comb_fluxerr = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd)), dtype=g_fluxerr.dtype)
    comb_flag = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd)), dtype=g_flag.dtype)

    g_nexp = len(g_bjd)
    r_nexp = len(r_bjd)
    for i, psid in enumerate(unique_ids):
        if psid in g_psid:
            g_index = np.where(g_psid == psid)[0][0]
            g_flux_data = g_flux[g_index * g_nexp : (g_index + 1) * g_nexp]
            g_fluxerr_data = g_fluxerr[g_index * g_nexp : (g_index + 1) * g_nexp]
            g_flag_data = g_flag[g_index * g_nexp : (g_index + 1) * g_nexp]
            comb_flux[i, :len(g_bjd)] = g_flux_data
            comb_fluxerr[i, :len(g_bjd)] = g_fluxerr_data
            comb_flag[i, :len(g_bjd)] = g_flag_data

            #also add the ra/dec. If the source is in g only, or in both g and r, take the g position
            ra = g_ra[g_index]
            dec = g_dec[g_index]
            mag_ref = g_mag_ref[g_index]
            mag_err_ref = g_mag_err_ref[g_index]
            objtype = g_objtype[g_index]

        if psid in r_psid:
            r_index = np.where(r_psid == psid)[0][0]
            r_flux_data = r_flux[r_index * r_nexp : (r_index + 1) * r_nexp]
            r_fluxerr_data = r_fluxerr[r_index * r_nexp : (r_index + 1) * r_nexp]
            r_flag_data = r_flag[r_index * r_nexp : (r_index + 1) * r_nexp]
            comb_flux[i, len(g_bjd):] = r_flux_data
            comb_fluxerr[i, len(g_bjd):] = r_fluxerr_data
            comb_flag[i, len(g_bjd):] = r_flag_data

        if psid in r_psid and psid not in g_psid: #if the source is only in r-band, take the r-band position
            ra = r_ra[r_index]
            dec = r_dec[r_index]
            mag_ref = r_mag_ref[r_index]
            mag_err_ref = r_mag_err_ref[r_index]
            objtype = r_objtype[r_index]
        comb_ra[i] = ra
        comb_dec[i] = dec
        comb_mag_ref[i] = mag_ref
        comb_mag_err_ref[i] = mag_err_ref
        comb_objtype[i] = objtype

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

def merge_g_r_i(field, ccd, quad):
    g_filename = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zg.h5'
    r_filename = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zr.h5'
    i_filename = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zi.h5'
    output_filename = f'/data/zvar/comb_matchfiles/{field:04d}/comb_data_{field:04d}_{ccd:02d}_{quad:01d}.h5'

    #Load relevant data from each file
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

    #Find unique gaia_ids (psids)
    unique_ids = np.union1d(g_psid, np.union1d(r_psid, i_psid))  # Returns sorted unique values, dtype preserved

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

    g_nexp = len(g_bjd)
    r_nexp = len(r_bjd)
    i_nexp = len(i_bjd)
    for i, psid in enumerate(unique_ids):
        if psid in g_psid:
            g_index = np.where(g_psid == psid)[0][0]
            g_flux_data = g_flux[g_index * g_nexp : (g_index + 1) * g_nexp]
            g_fluxerr_data = g_fluxerr[g_index * g_nexp : (g_index + 1) * g_nexp]
            g_flag_data = g_flag[g_index * g_nexp : (g_index + 1) * g_nexp]
            comb_flux[i, :len(g_bjd)] = g_flux_data
            comb_fluxerr[i, :len(g_bjd)] = g_fluxerr_data
            comb_flag[i, :len(g_bjd)] = g_flag_data

            #also add the ra/dec. If the source is in g only, or in multiple bands, take the g position
            ra = g_ra[g_index]
            dec = g_dec[g_index]
            mag_ref = g_mag_ref[g_index]
            mag_err_ref = g_mag_err_ref[g_index]
            objtype = g_objtype[g_index]

        if psid in r_psid:
            r_index = np.where(r_psid == psid)[0][0]
            r_flux_data = r_flux[r_index * r_nexp : (r_index + 1) * r_nexp]
            r_fluxerr_data = r_fluxerr[r_index * r_nexp : (r_index + 1) * r_nexp]
            r_flag_data = r_flag[r_index * r_nexp : (r_index + 1) * r_nexp]
            comb_flux[i, len(g_bjd): len(g_bjd) + len(r_bjd)] = r_flux_data
            comb_fluxerr[i, len(g_bjd):len(g_bjd) + len(r_bjd)] = r_fluxerr_data
            comb_flag[i, len(g_bjd):len(g_bjd) + len(r_bjd)] = r_flag_data

        if psid in i_psid:
            i_index = np.where(i_psid == psid)[0][0]
            i_flux_data = i_flux[i_index * i_nexp : (i_index + 1) * i_nexp]
            i_fluxerr_data = i_fluxerr[i_index * i_nexp : (i_index + 1) * i_nexp]
            i_flag_data = i_flag[i_index * i_nexp : (i_index + 1) * i_nexp]
            comb_flux[i, len(g_bjd) + len(r_bjd):] = i_flux_data
            comb_fluxerr[i, len(g_bjd) + len(r_bjd):] = i_fluxerr_data
            comb_flag[i, len(g_bjd) + len(r_bjd):] = i_flag_data

        if psid in r_psid and psid not in g_psid: #if the source is not in g-band, take the r-band position
            ra = r_ra[r_index]
            dec = r_dec[r_index]
            mag_ref = r_mag_ref[r_index]
            mag_err_ref = r_mag_err_ref[r_index]
            objtype = r_objtype[r_index]

        if psid in i_psid and psid not in g_psid and psid not in r_psid: #if the source is only in i-band, take the i-band position
            ra = i_ra[i_index]
            dec = i_dec[i_index]
            mag_ref = i_mag_ref[i_index]
            mag_err_ref = i_mag_err_ref[i_index]
            objtype = i_objtype[i_index]

        comb_ra[i] = ra
        comb_dec[i] = dec
        comb_mag_ref[i] = mag_ref
        comb_mag_err_ref[i] = mag_err_ref
        comb_objtype[i] = objtype

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


def which_function(field, ccd, quad):
    files = look_for_files(field, ccd, quad)
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

def process_file(field, ccd, quad):
    print(f"Processing field {field}, ccd {ccd}, quad {quad} on thread {threading.get_ident()}")
    func = which_function(field, ccd, quad)
    if func:
        func(field, ccd, quad)
        print(f"Completed processing for field {field}, ccd {ccd}, quad {quad}.")
    else:
        print(f"No g or r files found for field {field}, ccd {ccd}, quad {quad}. Skipping.")

def process_field(field):
    if field < 245 or field > 881:
        print("Field number must be between 245 and 881.")
        return
    ccds = range(1, 17)      # CCDs 1 to 16
    quads = range(1, 5)      # Quads 1 to 4

    for ccd in ccds:
        for quad in quads:
            process_file(field, ccd, quad)

def process_all_fields(n_threads=4):
    fields = np.arange(245, 882)
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(process_field, field) for field in fields]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge or copy ZTF matchfiles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # process_field command
    parser_field = subparsers.add_parser("process_field", help="Process a single field")
    parser_field.add_argument("field", type=int, help="Field number")

    # process_all_fields command
    parser_all = subparsers.add_parser("process_all_fields", help="Process all fields")
    parser_all.add_argument("--n_threads", type=int, default=4, help="Number of threads (default: 4)")

    args = parser.parse_args()

    if args.command == "process_field":
        process_field(args.field)
    elif args.command == "process_all_fields":
        process_all_fields(args.n_threads)
    else:
        parser.print_help()
        sys.exit(1)