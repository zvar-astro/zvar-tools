# ZVAR Utilities

## Description

This repository contains a collection of utilities for working with the ZVAR data products. The utilities are written in Python and can be used to retrieve variability candidates. So far, we mostly rely on the FPW periods and Gaia DR3 data to extract interesting candidates. Said candidates are then stored in a CSV file, and the notebooks in the `notebooks` directory can be used to further analyze the candidates.

## Installation

To get started with this project, you need to have Python installed on your machine. Follow the steps below to set up the project:

1. Clone the repository:

   ```sh
   git clone https://github.com/zvar-astro/ZVAR-Utilities
   cd ZVAR-Utilities
   ```

2. Fetch the demo_data submodule:

   ```sh
   git submodule update --init --recursive
   ```

   _If this command isn't recognized, you may need to update your git version or install [git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)._

3. Create a virtual environment, with [uv](https://docs.astral.sh/uv/getting-started/installation/):

   ```sh
   uv venv env --python=python3.10
   source env/bin/activate
   ```

   **Note:** You can also use conda, virtualenv, or any other virtual environment manager of your choice.

4. Install the required dependencies:

   ```sh
   uv sync
   ```

   _You can rerun this command whenever you want to update the dependencies._

## Usage

To generate a CSV file with variability candidates, you will first need to fetch FPW periods from magnetar (**subject to change**). You can save these anywhere in your machine, simply make sure to follow this directory structure:

```
whatever_directory/
└── field/
    └── fpw_field_ccd_quad_zfilter.h5
    └── ...
```

where:

- The field should be padded with zeros to have a length of 4. For example, field 1 should be `0001`.
- The `ccd` should be padded with zeros to have a length of 2. For example, CCD 1 should be `01`.
- The `quad` should be of length 1.
- The `filter` must be one of `g`, `r`, or `i`.

Example: `fpw_0001_01_1_zfilter.h5`.

Once you have the FPW periods, you can run the following command to generate the CSV file:

```sh
PYTHONPATH=. python scripts/retrieve_variability_candidates.py --field_min=279 --field_max=279 --bands=r,g --radius=3.0 --periods_path=/path/to/periods/directory --output_path=/path/to/output/directory --credentials_path=/path/to/credentials/file
```

where:

- `field_min` and `field_max` are the minimum and maximum fields to consider.
- `bands` is a comma-separated list of bands to consider. The possible values are `g`, `r`, and `i`.
- `radius` is the radius in arcseconds used to query for xmatches from Kowalski's Gaia DR3 catalog.
- `periods_path` is the path to the directory containing the FPW periods.
- `output_path` is the path to the directory where the CSV file will be saved.
- `credentials_path` is the path to the file containing the credentials to access Kowalski.

A default credentials file is provided as `credentials.default.json`. You can use this file as a template to create your own `credentials.json` file. The file should have the following structure:

```json
{
  "melman": {
    "username": "your_username_here",
    "password": "your_password_here"
  }
}
```

where `melman` is the name of the Kowalski instance you want to access, and `username` and `password` are your credentials to access the instance.

## Notebooks

The `notebooks` directory contains Jupyter notebooks that can be used to further analyze the variability candidates. For now, only one notebook is available, `candidates_analysis.ipynb`, in which we:

- Load the CSV file with the variability candidates.
- Plot the candidates on an HR diagram using Gaia DR3 data.
- Plot the light curves of the candidates.
- Compute a period using the Lomb-Scargle periodogram, and plot lightcurve vs periodogram vs phased lightcurve.
- Compute a period using the FPW algorithm, and plot lightcurve vs periodogram vs phased lightcurve.
