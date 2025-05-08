# Background

This is a branch of https://github.com/eulerlab/djimaging.
For the latest version of the code, please visit the original repository.

# Usage

## Setup
Install the requirements, including datajoint, and djimaging using pip:
```bash
cd code/AlphaDjimaging
pip install -e .
```

For details on how to use datajoint visit: https://datajoint.com/docs/core/datajoint-python/0.14/

## Running the notebooks
Go to ``djimaging/djimaging/user/alpha/notebooks`` to populate the databases.

``djimaging`` is a general preprocessing pipeline, the project specific code including jupyter notebooks is in the `djimaging/djimaging/user/alpha` directory.

Set CONFIG_FILE and SCHEMA_PREFIX in `djimaging/djimaging/user/alpha/utils/populate_alpha.py` according to your local settings.

