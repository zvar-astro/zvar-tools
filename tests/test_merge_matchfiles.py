#!/usr/bin/env python3
"""
Compare g/r band matchfiles to a merged combined matchfile for a field/ccd/quad.

Usage:
  python scripts/compare_merge.py --field 245 --ccd 1 --quad 1
  Optional: --input_path, --output_path (defaults match your repo)
"""

import argparse
import numpy as np
import h5py
import sys
import os

def read_band_file(path):
    with h5py.File(path, "r") as f:
        exposures = {k: f["/data/exposures"][k][:] for k in f["/data/exposures"].dtype.names} if hasattr(f["/data/exposures"], 'dtype') else {k: f["/data/exposures"][k][:] for k in f["/data/exposures"].keys()}
        # exposures fields as arrays (jd,bjd,...)
        # sources fields:
        srcs = {k: f["/data/sources"][k][:] for k in f["/data/sources"].dtype.names} if hasattr(f["/data/sources"], 'dtype') else {k: f["/data/sources"][k][:] for k in f["/data/sources"].keys()}
        # sourcedata are structured flattened arrays or compound dataset with fields 'flux','flux_err','flag'
        sd = f["/data/sourcedata"][:]
    return exposures, srcs, sd

def exposures_fields_from_group(group):
    # helper if exposures stored as dataset with named fields or as group of arrays
    if isinstance(group, h5py.Dataset) and group.dtype.names:
        return {n: group[n][:] for n in group.dtype.names}
    else:
        return {n: group[n][:] for n in group.keys()}

def load_band(path):
    with h5py.File(path, "r") as f:
        # exposures
        ex_node = f["/data/exposures"]
        if isinstance(ex_node, h5py.Dataset) and ex_node.dtype.names:
            exposures = {n: ex_node[n][:] for n in ex_node.dtype.names}
        else:
            exposures = {n: ex_node[n][:] for n in ex_node.keys()}
        # sources
        src_node = f["/data/sources"]
        if isinstance(src_node, h5py.Dataset) and src_node.dtype.names:
            sources = {n: src_node[n][:] for n in src_node.dtype.names}
        else:
            sources = {n: src_node[n][:] for n in src_node.keys()}
        # sourcedata
        sd_node = f["/data/sourcedata"]
        if isinstance(sd_node, h5py.Dataset) and sd_node.dtype.names:
            # structured array (flat) or 1D structured
            sourcedata = sd_node[:]
        else:
            # fallback
            sourcedata = sd_node[:]
    return exposures, sources, sourcedata

def reshape_band_sourcedata(sourcedata, n_sources, n_exposures, field_names=("flux","flux_err","flag")):
    """
    Many of your band files store sourcedata as a 1D structured array of length n_sources * n_exposures,
    where the values are ordered by source blocks. We'll reshape into (n_sources, n_exposures) arrays per field.
    """
    # Handle case where sourcedata is structured array with named fields
    if sourcedata.dtype.names:
        out = {}
        for fn in field_names:
            if fn not in sourcedata.dtype.names:
                raise KeyError(f"Field {fn} not found in sourcedata dtype {sourcedata.dtype.names}")
            arr = sourcedata[fn]
            if arr.size != n_sources * n_exposures:
                raise ValueError(f"Unexpected size for field {fn}: {arr.size} != {n_sources}*{n_exposures}")
            out[fn] = arr.reshape((n_sources, n_exposures))
        return out
    else:
        # If sourcedata is plain numeric array, try to infer shapes (unlikely)
        raise RuntimeError("sourcedata is not a structured array with named fields; cannot reshape automatically")

def build_expected_combined(g_expos, g_src, g_sd, r_expos, r_src, r_sd):
    # g_expos/r_expos: dicts containing 'bjd' etc.
    g_bjd = g_expos['bjd']
    r_bjd = r_expos['bjd']
    g_nexp = len(g_bjd)
    r_nexp = len(r_bjd)
    # psids
    g_psid = g_src['gaia_id']
    r_psid = r_src['gaia_id']
    # reshape sourcedata from band files
    g_nsrc = len(g_psid)
    r_nsrc = len(r_psid)

    g_sd_rows = reshape_band_sourcedata(g_sd, g_nsrc, g_nexp)
    r_sd_rows = reshape_band_sourcedata(r_sd, r_nsrc, r_nexp)

    unique_ids = np.union1d(g_psid, r_psid)
    n_sources = len(unique_ids)
    comb_nexp = g_nexp + r_nexp

    # initialize combined arrays
    comb_flux = np.zeros((n_sources, comb_nexp), dtype=g_sd_rows['flux'].dtype)
    comb_fluxerr = np.zeros_like(comb_flux)
    comb_flag = np.zeros((n_sources, comb_nexp), dtype=g_sd_rows['flag'].dtype)

    # positions arrays
    comb_ra = np.zeros(n_sources, dtype=g_src['ra'].dtype)
    comb_dec = np.zeros(n_sources, dtype=g_src['decl'].dtype)
    comb_mag_ref = np.zeros(n_sources, dtype=g_src['mag_ref'].dtype)
    comb_mag_err_ref = np.zeros(n_sources, dtype=g_src['mag_err_ref'].dtype)
    comb_objtype = np.zeros(n_sources, dtype=g_src['objtype'].dtype)

    # fill comb arrays per source
    for i, psid in enumerate(unique_ids):
        ra = dec = mag_ref = mag_err_ref = objtype = 0
        if psid in g_psid:
            g_index = np.where(g_psid == psid)[0][0]
            comb_flux[i, :g_nexp] = g_sd_rows['flux'][g_index]
            comb_fluxerr[i, :g_nexp] = g_sd_rows['flux_err'][g_index]
            comb_flag[i, :g_nexp] = g_sd_rows['flag'][g_index]
            ra = g_src['ra'][g_index]
            dec = g_src['decl'][g_index]
            mag_ref = g_src['mag_ref'][g_index]
            mag_err_ref = g_src['mag_err_ref'][g_index]
            objtype = g_src['objtype'][g_index]
        if psid in r_psid:
            r_index = np.where(r_psid == psid)[0][0]
            comb_flux[i, g_nexp:] = r_sd_rows['flux'][r_index]
            comb_fluxerr[i, g_nexp:] = r_sd_rows['flux_err'][r_index]
            comb_flag[i, g_nexp:] = r_sd_rows['flag'][r_index]
            # if not present in g, take r positions
            if psid not in g_psid:
                ra = r_src['ra'][r_index]
                dec = r_src['decl'][r_index]
                mag_ref = r_src['mag_ref'][r_index]
                mag_err_ref = r_src['mag_err_ref'][r_index]
                objtype = r_src['objtype'][r_index]
        comb_ra[i] = ra
        comb_dec[i] = dec
        comb_mag_ref[i] = mag_ref
        comb_mag_err_ref[i] = mag_err_ref
        comb_objtype[i] = objtype

    # exposures combined and sort idx by bjd (same as merge)
    comb_bjd = np.concatenate((g_bjd, r_bjd))
    sort_idx = np.argsort(comb_bjd)
    comb_jd = np.concatenate((g_expos['jd'], r_expos['jd']))[sort_idx]
    comb_bjd_sorted = comb_bjd[sort_idx]

    # apply column sort to sourcedata
    comb_flux_sorted = comb_flux[:, sort_idx]
    comb_fluxerr_sorted = comb_fluxerr[:, sort_idx]
    comb_flag_sorted = comb_flag[:, sort_idx]

    return {
        "unique_ids": unique_ids,
        "comb_bjd_sorted": comb_bjd_sorted,
        "comb_flux_sorted": comb_flux_sorted,
        "comb_fluxerr_sorted": comb_fluxerr_sorted,
        "comb_flag_sorted": comb_flag_sorted,
        "comb_ra": comb_ra,
        "comb_dec": comb_dec
    }

def compare_band_and_merged(input_path, output_path, field, ccd, quad, verbose=False, n_show=10):
    g_file = os.path.join(input_path, f"{field:04d}", f"data_{field:04d}_{ccd:02d}_{quad:01d}_zg.h5")
    r_file = os.path.join(input_path, f"{field:04d}", f"data_{field:04d}_{ccd:02d}_{quad:01d}_zr.h5")
    merged_file = os.path.join(output_path, f"{field:04d}", f"comb_data_{field:04d}_{ccd:02d}_{quad:01d}.h5")

    print("Files:")
    print("  g:", g_file)
    print("  r:", r_file)
    print("  merged:", merged_file)

    for p in (g_file, r_file, merged_file):
        if not os.path.exists(p):
            print("Missing file:", p)
            return 2

    # load
    g_expos, g_src, g_sd = load_band(g_file)
    r_expos, r_src, r_sd = load_band(r_file)
    m_expos, m_src, m_sd = load_band(merged_file)

    # basic checks
    print("\nBasic counts:")
    print("  g: sources", len(g_src['gaia_id']), "exposures", len(g_expos['bjd']))
    print("  r: sources", len(r_src['gaia_id']), "exposures", len(r_expos['bjd']))
    # merged exposures number should equal g_expos + r_expos
    print("  merged: sources", len(m_src['gaia_id']), "exposures", len(m_expos['bjd']))

    # Rebuild expected combined
    expected = build_expected_combined(g_expos, g_src, g_sd, r_expos, r_src, r_sd)
    # Flatten expected arrays
    n_sources = len(expected['unique_ids'])
    n_expos = expected['comb_flux_sorted'].shape[1]
    n_total = n_sources * n_expos
    exp_flux_flat = expected['comb_flux_sorted'].flatten()
    exp_fluxerr_flat = expected['comb_fluxerr_sorted'].flatten()
    exp_flag_flat = expected['comb_flag_sorted'].flatten()

    # merged file's sourcedata should be structured array of length n_total
    if m_sd.dtype.names is None:
        print("Merged sourcedata does not appear to be a structured array; dtype:", m_sd.dtype)
        return 3
    for fld in ('flux','flux_err','flag'):
        if fld not in m_sd.dtype.names:
            print("Merged sourcedata missing field:", fld)
            return 4

    m_flux = m_sd['flux']
    m_fluxerr = m_sd['flux_err']
    m_flag = m_sd['flag']

    print("\nShapes & sizes:")
    print("  expected (n_sources,n_expos):", expected['comb_flux_sorted'].shape, "flat:", exp_flux_flat.size)
    print("  merged sourcedata length:", m_flux.size)

    ok = True
    if m_flux.size != exp_flux_flat.size:
        print("Size mismatch: merged sourcedata length", m_flux.size, "!= expected", exp_flux_flat.size)
        ok = False
    else:
        # compare arrays
        eq_flux = np.allclose(m_flux.astype(np.float64), exp_flux_flat.astype(np.float64), equal_nan=True)
        eq_fluxerr = np.allclose(m_fluxerr.astype(np.float64), exp_fluxerr_flat.astype(np.float64), equal_nan=True)
        # flag compare as exact ints
        eq_flag = np.array_equal(m_flag, exp_flag_flat)

        print("\nField comparisons:")
        print("  flux equal:", eq_flux)
        print("  flux_err equal:", eq_fluxerr)
        print("  flag equal:", eq_flag)

        if not eq_flux:
            # show first mismatches
            diff_idx = np.where(~np.isclose(m_flux.astype(np.float64), exp_flux_flat.astype(np.float64), equal_nan=True))[0]
            print("  flux mismatch count:", diff_idx.size)
            for j in diff_idx[:n_show]:
                print(f"   idx {j}: merged={m_flux[j]} expected={exp_flux_flat[j]}")
            ok = False
        if not eq_fluxerr:
            diff_idx = np.where(~np.isclose(m_fluxerr.astype(np.float64), exp_fluxerr_flat.astype(np.float64), equal_nan=True))[0]
            print("  flux_err mismatch count:", diff_idx.size)
            for j in diff_idx[:n_show]:
                print(f"   idx {j}: merged={m_fluxerr[j]} expected={exp_fluxerr_flat[j]}")
            ok = False
        if not eq_flag:
            diff_idx = np.where(m_flag != exp_flag_flat)[0]
            print("  flag mismatch count:", diff_idx.size)
            for j in diff_idx[:n_show]:
                print(f"   idx {j}: merged={m_flag[j]} expected={exp_flag_flat[j]}")
            ok = False

    if ok:
        print("\nOK: merged sourcedata matches expected reconstruction.")
        return 0
    else:
        print("\nDiscrepancies found. See outputs above for sample mismatches.")
        return 5

def main():
    parser = argparse.ArgumentParser(description="Compare g/r matchfiles to merged combined matchfile.")
    parser.add_argument("--input_path", default="/data/zvar/matchfiles", help="Input matchfiles base path")
    parser.add_argument("--output_path", default="/data/zvar/comb_matchfiles", help="Output merged path")
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--ccd", type=int, required=True)
    parser.add_argument("--quad", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rc = compare_band_and_merged(args.input_path, args.output_path, args.field, args.ccd, args.quad, verbose=args.verbose)
    sys.exit(rc)

if __name__ == "__main__":
    main()