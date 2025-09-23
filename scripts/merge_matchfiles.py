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
        g_psid = f["/data/sources"]["gaia_id"][:]
        g_ra = f["/data/sources"]["ra"][:]
        g_dec = f["/data/sources"]["decl"][:]
        g_bjd = f["/data/exposures"]["bjd"][:]
        g_exptime = f["/data/exposures"]["exptime"][:]
        g_flux = f["/data/sourcedata"]["flux"][:]
        g_fluxerr = f["/data/sourcedata"]["flux_err"][:]
        g_flag = f["/data/sourcedata"]["flag"][:]

    with h5py.File(r_filename, "r") as f:
        r_psid = f["/data/sources"]["gaia_id"][:]
        r_ra = f["/data/sources"]["ra"][:]
        r_dec = f["/data/sources"]["decl"][:]
        r_bjd = f["/data/exposures"]["bjd"][:]
        r_exptime = f["/data/exposures"]["exptime"][:]
        r_flux = f["/data/sourcedata"]["flux"][:]
        r_fluxerr = f["/data/sourcedata"]["flux_err"][:]
        r_flag = f["/data/sourcedata"]["flag"][:]

    #Find unique gaia_ids (psids)
    g_set_psid = set(g_psid)
    r_set_psid = set(r_psid)
    unique_ids = g_set_psid.union(r_set_psid)
    
    #Combine the data
    comb_bjd = np.concatenate((g_bjd, r_bjd))
    comb_flux = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd)))
    comb_fluxerr = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd)))
    comb_flag = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd)))
    comb_ra = np.zeros(len(unique_ids))
    comb_dec = np.zeros(len(unique_ids))
    g_nexp = len(g_bjd)
    r_nexp = len(r_bjd)
    for i, psid in enumerate(unique_ids):
        if psid in g_set_psid:
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

        if psid in r_set_psid:
            r_index = np.where(r_psid == psid)[0][0]
            r_flux_data = r_flux[r_index * r_nexp : (r_index + 1) * r_nexp]
            r_fluxerr_data = r_fluxerr[r_index * r_nexp : (r_index + 1) * r_nexp]
            r_flag_data = r_flag[r_index * r_nexp : (r_index + 1) * r_nexp]
            comb_flux[i, len(g_bjd):] = r_flux_data
            comb_fluxerr[i, len(g_bjd):] = r_fluxerr_data
            comb_flag[i, len(g_bjd):] = r_flag_data

        if psid in r_set_psid and psid not in g_set_psid: #if the source is only in r-band, take the r-band position
            ra = r_ra[r_index]
            dec = r_dec[r_index]
        comb_ra[i] = ra
        comb_dec[i] = dec

    # Argsort everything by time
    sort_idx = np.argsort(comb_bjd)
    comb_bjd_sorted = comb_bjd[sort_idx]
    comb_ra_sorted = comb_ra[sort_idx]
    comb_dec_sorted = comb_dec[sort_idx]
    comb_flux_sorted = comb_flux[:, sort_idx]
    comb_fluxerr_sorted = comb_fluxerr[:, sort_idx]
    comb_flag_sorted = comb_flag[:, sort_idx]

    # Write out the combined data to a new HDF5 file
    with h5py.File(output_filename, "w") as f:
        # Create groups as in the original files
        data_group = f.create_group("data")
        sources_group = data_group.create_group("sources")
        exposures_group = data_group.create_group("exposures")
        sourcedata_group = data_group.create_group("sourcedata")

        # Save sources datasets
        sources_group.create_dataset("gaia_id", data=np.array(list(unique_ids)))
        sources_group.create_dataset("ra", data=comb_ra)
        sources_group.create_dataset("decl", data=comb_dec)

        # Save exposures datasets
        exposures_group.create_dataset("bjd", data=comb_bjd_sorted)

        # Save sourcedata datasets
        sourcedata_group.create_dataset("flux", data=comb_flux_sorted)
        sourcedata_group.create_dataset("flux_err", data=comb_fluxerr_sorted)
        sourcedata_group.create_dataset("flag", data=comb_flag_sorted)

def merge_g_r_i(field, ccd, quad):
    g_filename = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zg.h5'
    r_filename = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zr.h5'
    i_filename = f'/data/zvar/matchfiles/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_zi.h5'
    output_filename = f'/data/zvar/comb_matchfiles/{field:04d}/comb_data_{field:04d}_{ccd:02d}_{quad:01d}.h5'

    #Load relevant data from each file
    with h5py.File(g_filename, "r") as f:
        g_psid = f["/data/sources"]["gaia_id"][:]
        g_ra = f["/data/sources"]["ra"][:]
        g_dec = f["/data/sources"]["decl"][:]
        g_bjd = f["/data/exposures"]["bjd"][:]
        g_exptime = f["/data/exposures"]["exptime"][:]
        g_flux = f["/data/sourcedata"]["flux"][:]
        g_fluxerr = f["/data/sourcedata"]["flux_err"][:]
        g_flag = f["/data/sourcedata"]["flag"][:]

    with h5py.File(r_filename, "r") as f:
        r_psid = f["/data/sources"]["gaia_id"][:]
        r_ra = f["/data/sources"]["ra"][:]
        r_dec = f["/data/sources"]["decl"][:]
        r_bjd = f["/data/exposures"]["bjd"][:]
        r_exptime = f["/data/exposures"]["exptime"][:]
        r_flux = f["/data/sourcedata"]["flux"][:]
        r_fluxerr = f["/data/sourcedata"]["flux_err"][:]
        r_flag = f["/data/sourcedata"]["flag"][:]

    with h5py.File(i_filename, "r") as f:
        i_psid = f["/data/sources"]["gaia_id"][:]
        i_ra = f["/data/sources"]["ra"][:]
        i_dec = f["/data/sources"]["decl"][:]
        i_bjd = f["/data/exposures"]["bjd"][:]
        i_exptime = f["/data/exposures"]["exptime"][:]
        i_flux = f["/data/sourcedata"]["flux"][:]
        i_fluxerr = f["/data/sourcedata"]["flux_err"][:]
        i_flag = f["/data/sourcedata"]["flag"][:]

    #Find unique gaia_ids (psids)
    g_set_psid = set(g_psid)
    r_set_psid = set(r_psid)
    i_set_psid = set(i_psid)
    unique_ids = g_set_psid.union(r_set_psid).union(i_set_psid)

    #Combine the data
    comb_bjd = np.concatenate((g_bjd, r_bjd, i_bjd))
    comb_flux = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd) + len(i_bjd)))
    comb_fluxerr = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd) + len(i_bjd)))
    comb_flag = np.zeros((len(unique_ids), len(g_bjd) + len(r_bjd) + len(i_bjd)))
    comb_ra = np.zeros(len(unique_ids))
    comb_dec = np.zeros(len(unique_ids))
    g_nexp = len(g_bjd)
    r_nexp = len(r_bjd)
    i_nexp = len(i_bjd)
    for i, psid in enumerate(unique_ids):
        if psid in g_set_psid:
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

        if psid in r_set_psid:
            r_index = np.where(r_psid == psid)[0][0]
            r_flux_data = r_flux[r_index * r_nexp : (r_index + 1) * r_nexp]
            r_fluxerr_data = r_fluxerr[r_index * r_nexp : (r_index + 1) * r_nexp]
            r_flag_data = r_flag[r_index * r_nexp : (r_index + 1) * r_nexp]
            comb_flux[i, len(g_bjd): len(g_bjd) + len(r_bjd)] = r_flux_data
            comb_fluxerr[i, len(g_bjd):len(g_bjd) + len(r_bjd)] = r_fluxerr_data
            comb_flag[i, len(g_bjd):len(g_bjd) + len(r_bjd)] = r_flag_data

        if psid in i_set_psid:
            i_index = np.where(i_psid == psid)[0][0]
            i_flux_data = i_flux[i_index * i_nexp : (i_index + 1) * i_nexp]
            i_fluxerr_data = i_fluxerr[i_index * i_nexp : (i_index + 1) * i_nexp]
            i_flag_data = i_flag[i_index * i_nexp : (i_index + 1) * i_nexp]
            comb_flux[i, len(g_bjd) + len(r_bjd):] = i_flux_data
            comb_fluxerr[i, len(g_bjd) + len(r_bjd):] = i_fluxerr_data
            comb_flag[i, len(g_bjd) + len(r_bjd):] = i_flag_data

        if psid in r_set_psid and psid not in g_set_psid: #if the source is not in g-band, take the r-band position
            ra = r_ra[r_index]
            dec = r_dec[r_index]

        if psid in i_set_psid and psid not in g_set_psid and psid not in r_set_psid: #if the source is only in i-band, take the i-band position
            ra = i_ra[i_index]
            dec = i_dec[i_index]
        
        comb_ra[i] = ra
        comb_dec[i] = dec

    # Argsort everything by time
    sort_idx = np.argsort(comb_bjd)
    comb_bjd_sorted = comb_bjd[sort_idx]
    comb_ra_sorted = comb_ra[sort_idx]
    comb_dec_sorted = comb_dec[sort_idx]
    comb_flux_sorted = comb_flux[:, sort_idx]
    comb_fluxerr_sorted = comb_fluxerr[:, sort_idx]
    comb_flag_sorted = comb_flag[:, sort_idx]

    # Write out the combined data to a new HDF5 file
    with h5py.File(output_filename, "w") as f:
        # Create groups as in the original files
        data_group = f.create_group("data")
        sources_group = data_group.create_group("sources")
        exposures_group = data_group.create_group("exposures")
        sourcedata_group = data_group.create_group("sourcedata")

        # Save sources datasets
        sources_group.create_dataset("gaia_id", data=np.array(list(unique_ids)))
        sources_group.create_dataset("ra", data=comb_ra)
        sources_group.create_dataset("decl", data=comb_dec)

        # Save exposures datasets
        exposures_group.create_dataset("bjd", data=comb_bjd_sorted)

        # Save sourcedata datasets
        sourcedata_group.create_dataset("flux", data=comb_flux_sorted)
        sourcedata_group.create_dataset("flux_err", data=comb_fluxerr_sorted)
        sourcedata_group.create_dataset("flag", data=comb_flag_sorted)

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