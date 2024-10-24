import os

import h5py
import numpy as np
import paramiko
from scp import SCPClient, SCPException

from zvar_utils.enums import FILTERS, FILTER2IDX
from zvar_utils.spatial import get_field_id


def get_files_list(ra, dec, prefix="data"):
    field_ccd_quads = get_field_id(ra, dec)
    files_list = []
    for field, ccd, quad in field_ccd_quads:
        field = f"{field:04d}"
        ccd = f"{ccd:02d}"
        quad = f"{quad:01d}"
        for f in FILTERS:
            filename = f"{field}/{prefix}_{field}_{ccd}_{quad}_z{f}.h5"
            files_list.append(filename)
    return files_list


def get_files(
    files, local_path, ssh_client: paramiko.SSHClient = None, remote_path=None
):
    available_files = []
    missing_files = []
    for file in files:
        if not os.path.isfile(f"{local_path}/{file}"):
            missing_files.append(file)
        else:
            available_files.append(file)

    if not missing_files:
        return

    if ssh_client is None or remote_path is None:
        return available_files

    scp_client = SCPClient(ssh_client.get_transport())

    for file in missing_files:
        outdir = f'{local_path}/{file.split("/")[-2]}'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        remote_filename = f"{remote_path}/{file}"

        # check if the file exists on the remote server
        stdin, stdout, stderr = ssh_client.exec_command(f"ls {remote_filename}")
        if not stdout.read():
            continue
        print(f"Downloading {remote_filename} to {outdir}")
        try:
            scp_client.get(remote_filename, outdir)
            available_files.append(file)
        except SCPException as e:
            print(f"Could not download {remote_filename}: {e}")

    scp_client.close()

    return available_files


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
        # print(f'Reading {file_path}')
        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist")
            continue
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
                # photometry is a list of tuples: flux, flux_err, flag (where flag = 1 means flux is NaN)
                # to which we want to add a fourth element: the filter
                # we get the filter simply by looking at the last character of the file name (before the extension)
                filter = file.split(".")[-2][-1]
                if filter not in FILTERS:
                    continue

                photometry = []
                # also just make the flux = NaN where flag = 1
                for i in range(rows):
                    if int(raw_photometry[i][2]) == 1:
                        photometry.append(
                            [np.nan, float(raw_photometry[i][1]), FILTER2IDX[filter]]
                        )
                    else:
                        photometry.append(
                            [
                                float(raw_photometry[i][0]),
                                float(raw_photometry[i][1]),
                                FILTER2IDX[filter],
                            ]
                        )

                all_photometry[id] = all_photometry[id] + photometry

    return all_photometry


def get_ssh_client(host, username=None, password=None, pkey_path=None):
    """
    Create an scp client
    """
    if not username and not password and not pkey_path:
        raise ValueError("Either username and password or pkey_path must be provided")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if pkey_path:
        k = paramiko.RSAKey.from_private_key_file(pkey_path)
    else:
        k = None

    ssh.connect(host, username=username, password=password, pkey=k)

    return ssh
