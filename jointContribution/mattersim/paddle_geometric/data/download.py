import os
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional

import fsspec
from paddle_geometric.io import fs


def download_url(
    url: str,
    folder: str,
    log: bool = True,
    filename: Optional[str] = None,
):
    """
    Downloads the content of a URL to a specific folder.

    Args:
        url (str): The URL to download the file from.
        folder (str): The destination folder where the file will be saved.
        log (bool, optional): If False, no logs will be printed to the console.
            Default is True.
        filename (str, optional): The desired filename for the downloaded file.
            If None, the filename is inferred from the URL. Default is None.
    """
    if filename is None:
        # Extract the filename from the URL.
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    # If the file already exists, return its path.
    if fs.exists(path):  # pragma: no cover
        if log and 'pytest' not in sys.modules:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log and 'pytest' not in sys.modules:
        print(f'Downloading {url}', file=sys.stderr)

    # Ensure the folder exists.
    os.makedirs(folder, exist_ok=True)

    # Create an unverified SSL context for downloading the file.
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with fsspec.open(path, 'wb') as f:
        # Write data in chunks to avoid memory issues.
        while True:
            chunk = data.read(10 * 1024 * 1024)  # Read 10 MB chunks.
            if not chunk:
                break
            f.write(chunk)

    return path


def download_google_url(
    id: str,
    folder: str,
    filename: str,
    log: bool = True,
):
    """
    Downloads the content of a Google Drive file to a specific folder.

    Args:
        id (str): The Google Drive file ID.
        folder (str): The destination folder where the file will be saved.
        filename (str): The desired filename for the downloaded file.
        log (bool, optional): If False, no logs will be printed to the console.
            Default is True.
    """
    # Construct the Google Drive download URL using the file ID.
    url = f'https://drive.usercontent.google.com/download?id={id}&confirm=t'
    return download_url(url, folder, log, filename)
