import os

import paramiko
from scp import SCPClient, SCPException

from zvar_utils.enums import FILTERS
from zvar_utils.spatial import get_field_id


def get_files_list(ra, dec, prefix="data", bands=FILTERS):
    if not set(bands).issubset(FILTERS):
        raise ValueError(f"Allowed bands are {FILTERS}, got {bands}")
    field_ccd_quads = get_field_id(ra, dec)
    files_list = []
    for field, ccd, quad in field_ccd_quads:
        field = f"{field:04d}"
        ccd = f"{ccd:02d}"
        quad = f"{quad:01d}"
        for f in bands:
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
