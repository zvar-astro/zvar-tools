import os
from typing import Tuple, List
import numpy as np
import time
import h5py
from astropy.coordinates import SkyCoord

import paramiko
import requests
from scp import SCPClient, SCPException
from penquins import Kowalski

from zvartools.spatial import get_field_id, great_circle_distance
from zvartools.enums import FILTER2IDX
from zvartools.lightcurves import process_curve, remove_deep_drilling
from zvartools.candidate import (
    VariabilityCandidate,
    add_gaia_xmatch_to_candidates,
    add_ps1_xmatch_to_candidates,
    add_allwise_xmatch_to_candidates,
    add_2mass_xmatch_to_candidates,
)


class BaseDataSource:
    def __init__(self, local_lightcurve_path, local_period_path, **kwargs):
        self.local_lightcurve_path = local_lightcurve_path
        self.local_period_path = local_period_path
        self.kwargs = kwargs
        self.verbose = kwargs.get("verbose", False)
        if (
            kwargs.get("kowalski_protocol")
            and kwargs.get("kowalski_host")
            and kwargs.get("kowalski_port")
        ):
            if kwargs.get("kowalski_token"):
                k = Kowalski(
                    protocol=kwargs.get("kowalski_protocol"),
                    host=kwargs.get("kowalski_host"),
                    port=443,
                    token=kwargs.get("kowalski_token"),
                    verbose=kwargs.get("verbose", False),
                    timeout=600,
                )
            elif kwargs.get("kowalski_user") and kwargs.get("kowalski_password"):
                k = Kowalski(
                    protocol=kwargs.get("kowalski_protocol"),
                    host=kwargs.get("kowalski_host"),
                    port=kwargs.get("kowalski_port"),
                    user=kwargs.get("kowalski_user"),
                    password=kwargs.get("kowalski_password"),
                    verbose=kwargs.get("verbose", False),
                    timeout=600,
                )
            else:
                raise ValueError("Kowalski credentials not provided")
            self.kowalski = k

        self.implements = {
            "check_file_availability",
            "get_files",
        }

    def check_file_availability(
        self, data: Tuple[int, int, int, str], type="lightcurve"
    ) -> Tuple[List, List]:
        base_dir, prefix = None, None
        if type == "lightcurve":
            base_dir, prefix = self.local_lightcurve_path, "data"
        elif type == "period":
            base_dir, prefix = self.local_period_path, "fpw"
        else:
            raise ValueError(f"Type {type} is not supported")

        available, missing = [], []

        for field, ccd, quad, band in data:
            filename = (
                f"{field:04d}/{prefix}_{field:04d}_{ccd:02d}_{quad:01d}_z{band}.h5"
            )
            if not os.path.isfile(f"{base_dir}/{filename}"):
                missing.append((field, ccd, quad, band))
            else:
                available.append((field, ccd, quad, band))

        return available, missing

    def get_files(self, data: Tuple[int, int, int, str], type="lightcurve") -> List:
        available, missing = self.check_file_availability(data, type)
        if len(missing) > 0:
            if "download_files" in self.implements:
                if self.verbose:
                    print(f"Downloading {len(missing)} missing files")

                self.download_files(missing, type)
                available, missing = self.check_file_availability(data, type)

        if len(missing) > 0 and self.verbose:
            print(f"Missing {len(missing)} {type} files")

        return available

    def download_files(
        self, data: Tuple[int, int, int, str], type="lightcurve"
    ) -> List:
        raise NotImplementedError(
            "This method should be implemented in the child classes"
        )

    @staticmethod
    def read_lightcurve_by_idx(idx, sources_data, times, band, zeropoints, exptimes):
        nb_exposures = len(times)
        start, end = idx * nb_exposures, (idx + 1) * nb_exposures

        raw_photometry = sources_data[start:end]

        photometry = []
        for i in range(nb_exposures):
            if (
                int(raw_photometry[i][2]) == 1
            ):  # if the source is not detected in this exposure
                photometry.append(
                    [
                        times[i],
                        np.nan,
                        float(raw_photometry[i][1]),
                        FILTER2IDX[band],
                    ]
                )
            else:  # if the source is detected in this exposure
                flux_jy, flux_err_jy = BaseDataSource.adu_to_jansky(
                    float(raw_photometry[i][0]),
                    float(raw_photometry[i][1]),
                    exptimes[i],
                    zeropoints[i],
                )
                # we convert to microJy
                photometry.append(
                    [
                        times[i],
                        flux_jy * 1e6,
                        flux_err_jy * 1e6,
                        FILTER2IDX[band],
                    ]
                )

        time, flux, flux_err = (
            np.array([x[0] for x in photometry]),
            np.array([x[1] for x in photometry]),
            np.array([x[2] for x in photometry]),
        )
        photometry = list(zip(time, flux, flux_err, [x[3] for x in photometry]))

        return photometry

    @staticmethod
    def read_period_by_idx(idx, dataset):
        try:
            freqs = dataset["bestFreqs"][idx * 50 * 3 : (idx + 1) * 50 * 3]
            sigs = dataset["significance"][idx * 50 * 3 : (idx + 1) * 50 * 3]
            valid = dataset["valid"][idx]
            max_sig = np.max(sigs)
            max_sig_idx = np.argmax(sigs)
            best_M = 20 if max_sig_idx < 50 else 10 if max_sig_idx < 100 else 5
            best_freq = freqs[max_sig_idx]
            return best_freq, max_sig, best_M, valid
        except Exception as e:
            print(f"Error reading period for {idx}: {e}")
            return None, None, None, None

    @staticmethod
    def merge_lightcurves(lightcurve_per_psid):
        for id, photometry in lightcurve_per_psid.items():
            if not photometry:
                print(f"No photometry for {id} found in any file")
                continue
            lightcurve_per_psid[id] = np.array(sorted(photometry, key=lambda x: x[0])).T
            lightcurve_per_psid[id] = process_curve(*lightcurve_per_psid[id])
            lightcurve_per_psid[id] = remove_deep_drilling(*lightcurve_per_psid[id])

            lightcurve_per_psid[id] = np.array(lightcurve_per_psid[id], order="C")

        return lightcurve_per_psid

    def cone_search(self, targets, radius=2.0, max_matches=None):
        fieldccdquad2radec = {}
        if isinstance(targets, (list, tuple)) and all(
            isinstance(t, (list, tuple)) and len(t) == 2 for t in targets
        ):
            positions = targets
        elif isinstance(targets, (list, tuple)) and all(
            isinstance(t, SkyCoord) for t in targets
        ):
            positions = [(t.ra.deg, t.dec.deg) for t in targets]
        elif isinstance(targets, (list, tuple)) and all(
            isinstance(t, dict) and "ra" in t and "dec" in t for t in targets
        ):
            positions = [(t["ra"], t["dec"]) for t in targets]
        # if its a python class with ra and dec attributes
        elif isinstance(targets, (list, tuple)) and all(
            isinstance(t, object)
            and isinstance(getattr(t, "ra", None), float)
            and isinstance(getattr(t, "dec", None), float)
            for t in targets
        ):
            positions = [(t.ra, t.dec) for t in targets]
        else:
            raise ValueError(
                "Targets should be a list of tuples, lists, SkyCoord objects, dicts or objects with ra and dec attributes"
            )

        for ra, dec in positions:
            field_ccd_quads = get_field_id(ra, dec)
            # only keep the primary fields, so where field < 1000
            field_ccd_quads = [f for f in field_ccd_quads if f[0] < 1000]

            for field, ccd, quad in field_ccd_quads:
                if (field, ccd, quad) not in fieldccdquad2radec:
                    fieldccdquad2radec[(field, ccd, quad)] = []
                fieldccdquad2radec[(field, ccd, quad)].append((ra, dec))

        data_to_recover = []
        for (field, ccd, quad), radecs in fieldccdquad2radec.items():
            for ra, dec in radecs:
                for band in ["g", "r"]:
                    data_to_recover.append((field, ccd, quad, band))

        available_lightcurves = self.get_files(data_to_recover, type="lightcurve")
        available_periods = self.get_files(data_to_recover, type="period")

        radec2sources = {(ra, dec): [] for ra, dec in positions}
        lightcurve_per_psid = {}
        periods_per_psid = {}
        for field, ccd, quad, band in available_lightcurves:
            matched_idx = set()
            lightcurve_file_path = f"{self.local_lightcurve_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_z{band}.h5"
            with h5py.File(lightcurve_file_path, "r") as f:
                data = f["data"]
                sources: h5py.Dataset = data["sources"]

                for ra, dec in fieldccdquad2radec[(field, ccd, quad)]:
                    distances = great_circle_distance(
                        ra, dec, sources["ra"], sources["decl"]
                    )
                    # TODO: if max_matches is defined, limit the nb of results returned.
                    matches = distances < radius / 3600
                    if max_matches:
                        # sort by distance and take the first max_matches
                        matches = np.argsort(distances)[0:max_matches]
                    for i in np.where(matches)[0]:
                        psid = sources["gaia_id"][i]
                        radec2sources[(ra, dec)].append(
                            {
                                "psid": psid,
                                "ra": sources["ra"][i],
                                "dec": sources["decl"][i],
                                "field": field,
                                "ccd": ccd,
                                "quad": quad,
                                "band": band,
                            }
                        )
                        if psid not in lightcurve_per_psid:
                            lightcurve_per_psid[psid] = []
                        matched_idx.add((i, psid))

                for i, psid in matched_idx:
                    lc = self.read_lightcurve_by_idx(
                        i,
                        data["sourcedata"],
                        data["exposures"]["bjd"],
                        band,
                        data["exposures"]["mzpsci"],
                        data["exposures"]["exptime"],
                    )
                    lightcurve_per_psid[psid].extend(lc)

            if (field, ccd, quad, band) not in available_periods:
                print(
                    f"Period file for {field:04d}_{ccd:02d}_{quad:01d}_z{band} not found"
                )
                continue
            periods_file_path = f"{self.local_period_path}/{field:04d}/fpw_{field:04d}_{ccd:02d}_{quad:01d}_z{band}.h5"
            with h5py.File(periods_file_path, "r") as dataset:
                psids = np.array(dataset["psids"])
                assert len(dataset["bestFreqs"]) == len(dataset["psids"]) * 50 * 3

                for _, psid in matched_idx:
                    idx = np.where(psids == psid)[0]
                    if len(idx) == 0:
                        continue

                    freq, sig, best_M, valid = self.read_period_by_idx(idx[0], dataset)
                    if (psid, band) not in periods_per_psid:
                        periods_per_psid[(psid, band)] = []
                    periods_per_psid[(psid, band)].append(
                        {
                            "freq": freq,
                            "significance": sig,
                            "best_M": best_M,
                            "valid": valid,
                        }
                    )

        radec2sources = {
            (ra, dec): list(match) for (ra, dec), match in radec2sources.items()
        }
        for key in radec2sources:
            radec2sources[key] = sorted(radec2sources[key], key=lambda x: x["psid"])
            # for each, add the period data if available
            for i in range(len(radec2sources[key])):
                psid = radec2sources[key][i]["psid"]
                if (psid, radec2sources[key][i]["band"]) in periods_per_psid:
                    radec2sources[key][i] = {
                        **radec2sources[key][i],
                        **periods_per_psid[(psid, radec2sources[key][i]["band"])][0],
                    }
                # convert them into VariabilityCandidate objects
                radec2sources[key][i] = VariabilityCandidate(
                    psid=psid,
                    ra=radec2sources[key][i]["ra"],
                    dec=radec2sources[key][i]["dec"],
                    band=radec2sources[key][i].get("band", None),
                    freq=radec2sources[key][i].get("freq", None),
                    fap=radec2sources[key][i].get("significance", None),
                    best_M=radec2sources[key][i].get("best_M", None),
                    field=radec2sources[key][i].get("field", None),
                    ccd=radec2sources[key][i].get("ccd", None),
                    quad=radec2sources[key][i].get("quad", None),
                    valid=radec2sources[key][i].get("valid", None),
                )

        lightcurve_per_psid = self.merge_lightcurves(lightcurve_per_psid)

        return radec2sources, lightcurve_per_psid

    def psid_search(self, pstargets):
        if isinstance(pstargets, (list, tuple)) and all(
            isinstance(t, (list, tuple)) and len(t) == 4 for t in pstargets
        ):
            targets = pstargets
        elif isinstance(pstargets, (list, tuple)) and all(
            isinstance(t, dict)
            and "psid" in t
            and "field" in t
            and "ccd" in t
            and "quad" in t
            for t in pstargets
        ):
            targets = [(t["psid"], t["field"], t["ccd"], t["quad"]) for t in pstargets]
        elif isinstance(pstargets, (list, tuple)) and all(
            isinstance(t, object)
            and hasattr(t, "psid")
            and hasattr(t, "field")
            and hasattr(t, "ccd")
            and hasattr(t, "quad")
            for t in pstargets
        ):
            targets = [(t.psid, t.field, t.ccd, t.quad) for t in pstargets]

        lightcurve_per_psid = {}
        metadata_per_psidband = {}
        # create a mapper from field, ccd, quad to psid
        fieldccdquad2psid = {}
        data_to_recover = []
        for psid, field, ccd, quad in targets:
            if (field, ccd, quad) not in fieldccdquad2psid:
                fieldccdquad2psid[(field, ccd, quad)] = []
                for band in ["g", "r"]:
                    data_to_recover.append((field, ccd, quad, band))
            fieldccdquad2psid[(field, ccd, quad)].append(psid)

        available_lightcurves = self.get_files(data_to_recover, type="lightcurve")
        available_periods = self.get_files(data_to_recover, type="period")

        for field, ccd, quad, band in available_lightcurves:
            lightcurve_file_path = f"{self.local_lightcurve_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_z{band}.h5"
            with h5py.File(lightcurve_file_path, "r") as f:
                data = f["data"]
                sources: h5py.Dataset = data["sources"]
                for psid in fieldccdquad2psid[(field, ccd, quad)]:
                    idx = np.where(sources["gaia_id"][:] == psid)[0]
                    if len(idx) == 0:
                        continue
                    idx = idx[0]
                    lc = self.read_lightcurve_by_idx(
                        idx,
                        data["sourcedata"],
                        data["exposures"]["bjd"],
                        band,
                        data["exposures"]["mzpsci"],
                        data["exposures"]["exptime"],
                    )
                    if psid not in lightcurve_per_psid:
                        lightcurve_per_psid[psid] = []
                    if (psid, band) not in metadata_per_psidband:
                        metadata_per_psidband[(psid, band)] = {
                            "psid": psid,
                            "ra": sources["ra"][idx],
                            "dec": sources["decl"][idx],
                            "field": field,
                            "ccd": ccd,
                            "quad": quad,
                            "band": band,
                        }
                    lightcurve_per_psid[psid].extend(lc)

            if (field, ccd, quad, band) not in available_periods:
                print(
                    f"Period file for {field:04d}_{ccd:02d}_{quad:01d}_z{band} not found"
                )
                continue
            periods_file_path = f"{self.local_period_path}/{field:04d}/fpw_{field:04d}_{ccd:02d}_{quad:01d}_z{band}.h5"
            with h5py.File(periods_file_path, "r") as dataset:
                psids = np.array(dataset["psids"])
                assert len(dataset["bestFreqs"]) == len(dataset["psids"]) * 50 * 3

                for psid in fieldccdquad2psid[(field, ccd, quad)]:
                    idx = np.where(psids == psid)[0]
                    if len(idx) == 0:
                        continue

                    freq, sig, best_M, valid = self.read_period_by_idx(idx[0], dataset)
                    metadata_per_psidband[(psid, band)] = {
                        **metadata_per_psidband[(psid, band)],
                        "freq": freq,
                        "significance": sig,
                        "best_M": best_M,
                        "valid": valid,
                    }

        lightcurve_per_psid = self.merge_lightcurves(lightcurve_per_psid)

        # then create the VariabilityCandidate objects
        candidates = {}
        for (psid, band), metadata in metadata_per_psidband.items():
            if psid not in candidates:
                candidates[psid] = []
            candidates[psid].append(
                VariabilityCandidate(
                    psid=psid,
                    ra=metadata["ra"],
                    dec=metadata["dec"],
                    band=metadata["band"],
                    freq=metadata["freq"],
                    fap=metadata["significance"],
                    best_M=metadata["best_M"],
                    field=metadata["field"],
                    ccd=metadata["ccd"],
                    quad=metadata["quad"],
                    valid=metadata["valid"],
                )
            )

        return candidates, lightcurve_per_psid

    def get_candidates_lightcurves(self, candidates):
        if not isinstance(candidates, (list, tuple)) and all(
            isinstance(c, VariabilityCandidate) for c in candidates
        ):
            raise ValueError(
                "Candidates should be a list of VariabilityCandidate objects"
            )

        data_to_recover = []
        for candidate in candidates:
            if candidate.band is None:
                for band in ["g", "r"]:
                    data_to_recover.append(
                        (candidate.field, candidate.ccd, candidate.quad, band)
                    )
            else:
                data_to_recover.append(
                    (candidate.field, candidate.ccd, candidate.quad, candidate.band)
                )

        available_lightcurves = self.get_files(data_to_recover, type="lightcurve")

        lightcurve_per_psid = {}

        for field, ccd, quad, band in available_lightcurves:
            lightcurve_file_path = f"{self.local_lightcurve_path}/{field:04d}/data_{field:04d}_{ccd:02d}_{quad:01d}_z{band}.h5"
            with h5py.File(lightcurve_file_path, "r") as f:
                data = f["data"]
                sources: h5py.Dataset = data["sources"]
                for candidate in candidates:
                    idx = np.where(sources["gaia_id"][:] == candidate.psid)[0]
                    if len(idx) == 0:
                        continue
                    idx = idx[0]
                    lc = self.read_lightcurve_by_idx(
                        idx,
                        data["sourcedata"],
                        data["exposures"]["bjd"],
                        band,
                        data["exposures"]["mzpsci"],
                        data["exposures"]["exptime"],
                    )
                    if candidate.psid not in lightcurve_per_psid:
                        lightcurve_per_psid[candidate.psid] = []
                    lightcurve_per_psid[candidate.psid].extend(lc)

        lightcurve_per_psid = self.merge_lightcurves(lightcurve_per_psid)

        return lightcurve_per_psid

    def add_gaia_data_to_candidates(self, candidates, radius=3.0):
        if not isinstance(candidates, (list, tuple)) and all(
            isinstance(c, VariabilityCandidate) for c in candidates
        ):
            raise ValueError(
                "Candidates should be a list of VariabilityCandidate objects"
            )

        if self.kowalski is None:
            raise ValueError(
                "Kowalski instance not provided, required for adding Gaia xmatch to candidates"
            )

        return add_gaia_xmatch_to_candidates(self.kowalski, candidates, radius)

    def add_ps1_data_to_candidates(self, candidates):
        if not isinstance(candidates, (list, tuple)) and all(
            isinstance(c, VariabilityCandidate) for c in candidates
        ):
            raise ValueError(
                "Candidates should be a list of VariabilityCandidate objects"
            )

        if self.kowalski is None:
            raise ValueError(
                "Kowalski instance not provided, required for adding Pan-STARRS xmatch to candidates"
            )

        return add_ps1_xmatch_to_candidates(self.kowalski, candidates)

    def add_allwise_data_to_candidates(self, candidates, radius=3.0):
        if not isinstance(candidates, (list, tuple)) and all(
            isinstance(c, VariabilityCandidate) for c in candidates
        ):
            raise ValueError(
                "Candidates should be a list of VariabilityCandidate objects"
            )

        if self.kowalski is None:
            raise ValueError(
                "Kowalski instance not provided, required for adding AllWISE xmatch to candidates"
            )

        return add_allwise_xmatch_to_candidates(self.kowalski, candidates, radius)

    def add_2mass_data_to_candidates(self, candidates, radius=3.0):
        if not isinstance(candidates, (list, tuple)) and all(
            isinstance(c, VariabilityCandidate) for c in candidates
        ):
            raise ValueError(
                "Candidates should be a list of VariabilityCandidate objects"
            )

        if self.kowalski is None:
            raise ValueError(
                "Kowalski instance not provided, required for adding 2MASS xmatch to candidates"
            )

        return add_2mass_xmatch_to_candidates(self.kowalski, candidates, radius)


class RemoteDataSource(BaseDataSource):
    def __init__(
        self,
        local_lightcurve_path,
        local_period_path,
        remote_lightcurve_path,
        remote_period_path,
        ssh_host,
        ssh_user,
        ssh_password=None,
        ssh_key_filename=None,
        **kwargs,
    ):
        super().__init__(local_lightcurve_path, local_period_path, **kwargs)
        self.remote_lightcurve_path = remote_lightcurve_path
        self.remote_period_path = remote_period_path

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.load_system_host_keys()
        if ssh_key_filename:
            self.ssh_client.connect(
                ssh_host, username=ssh_user, key_filename=ssh_key_filename
            )
        else:
            self.ssh_client.connect(ssh_host, username=ssh_user, password=ssh_password)
        self.implements.add("download_files")

    def download_files(
        self, data: Tuple[int, int, int, str], type="lightcurve"
    ) -> List:
        scp_client = SCPClient(self.ssh_client.get_transport())

        local_base_dir, remote_base_dir, prefix = None, None, None
        if type == "lightcurve":
            local_base_dir, remote_base_dir, prefix = (
                self.local_lightcurve_path,
                self.remote_lightcurve_path,
                "data",
            )
        elif type == "period":
            local_base_dir, remote_base_dir, prefix = (
                self.local_period_path,
                self.remote_period_path,
                "fpw",
            )
        else:
            raise ValueError(f"Type {type} is not supported")

        downloaded = []

        for field, ccd, quad, band in data:
            filename = (
                f"{field:04d}/{prefix}_{field:04d}_{ccd:02d}_{quad:01d}_z{band}.h5"
            )
            remote_filename = f"{remote_base_dir}/{filename}"
            outdir = f"{local_base_dir}/{field:04d}"
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            # if we already have the file, we skip it
            if os.path.isfile(f"{local_base_dir}/{filename}"):
                continue

            start = time.time()

            # check if the file exists on the remote server
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"ls {remote_filename}"
            )
            if not stdout.read():
                continue
            if stderr.read():
                if self.verbose:
                    print(
                        f"Could not check if {remote_filename} exists: {stderr.read()}"
                    )
                continue

            if self.verbose:
                print(f"Downloading {remote_filename} to {outdir}")
            try:
                scp_client.get(remote_filename, outdir)
            except SCPException as e:
                if self.verbose:
                    print(f"Could not download {remote_filename}: {e}")

            if self.verbose:
                print(f"Downloaded {filename} in {time.time() - start:.2f} s")
                # print the size of the file in MB
                print(
                    f"File size: {os.path.getsize(f'{local_base_dir}/{filename}') / 1024 / 1024:.2f} MB"
                )

            downloaded.append((field, ccd, quad, band))

        scp_client.close()

        return downloaded


class LocalDataSource(BaseDataSource):
    def __init__(self, local_lightcurve_path, local_period_path, **kwargs):
        super().__init__(local_lightcurve_path, local_period_path, **kwargs)

    def download_files(
        self, data: Tuple[int, int, int, str], type="lightcurve"
    ) -> List:
        raise NotImplementedError("LocalDataSource does not support downloading files")


class APIDataSource(BaseDataSource):
    def __init__(
        self,
        local_lightcurve_path,
        local_period_path,
        api_url,
        api_user,
        api_password,
        **kwargs,
    ):
        super().__init__(local_lightcurve_path, local_period_path, **kwargs)
        self.api_url = api_url
        self.api_user = api_user
        self.api_password = api_password
        self.implements.add("download_files")

    def download_files(
        self, data: Tuple[int, int, int, str], type="lightcurve"
    ) -> List:
        # the API is like protocol://host:port/api/type?field=field&ccd=ccd&quad=quad&band=band
        # we need to make a request for each file
        # the API should return the file content
        # we should save it in the local filesystem

        local_base_dir, prefix = None, None
        if type == "lightcurve":
            local_base_dir, prefix, api_endpoint = (
                self.local_lightcurve_path,
                "data",
                "lightcurves",
            )
        elif type == "period":
            local_base_dir, prefix, api_endpoint = (
                self.local_period_path,
                "fpw",
                "periods",
            )
        else:
            raise ValueError(f"Type {type} is not supported")

        downloaded = []

        for field, ccd, quad, band in data:
            filename = (
                f"{field:04d}/{prefix}_{field:04d}_{ccd:02d}_{quad:01d}_z{band}.h5"
            )
            outdir = f"{local_base_dir}/{field:04d}"
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            # if we already have the file, we skip it
            if os.path.isfile(f"{local_base_dir}/{filename}"):
                continue

            # make the request to the API
            # save the content to the local filesystem
            if self.verbose:
                print(f"Downloading {filename} to {outdir}")

            # query it with python request, streaming the data in chunks to save memory
            url = f"{self.api_url}/{api_endpoint}"
            print(url)
            start = time.time()
            response = requests.get(
                url,
                params={"field": field, "ccd": ccd, "quad": quad, "filter": band},
                stream=True,
                auth=(self.api_user, self.api_password),
            )
            if response.status_code != 200:
                if self.verbose:
                    if response.status_code == 404:
                        print(f"File {filename} not found")
                    elif response.status_code == 401:
                        print(f"Unauthorized access to {filename}")
                    else:
                        print(f"Could not download {filename}: {response.text}")
                continue

            with open(f"{local_base_dir}/{filename}", "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

            if self.verbose:
                print(f"Downloaded {filename} in {time.time() - start:.2f} s")
                # print the size of the file in MB
                print(
                    f"File size: {os.path.getsize(f'{local_base_dir}/{filename}') / 1024 / 1024:.2f} MB"
                )

            downloaded.append((field, ccd, quad, band))

        return downloaded
