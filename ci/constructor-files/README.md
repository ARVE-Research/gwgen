Constructor information for gwgen_conda
=======================================

The files in this directory are intended to construct a conda installer which
to be ready to use with the gwgen installation. The environments do not include
GWGEN itself, nor gfortran. GWGEN can be installed through the github sources
via `python setup.py install`, gfortran can be installed, e.g., via
`sudo apt-get install gfortran` (on Debian).

The created environment contain the following packages, downloaded from the
conda-forge channel:

- python
- scipy
- matplotlib
- dask
- xarray
- netcdf4
- cartopy
- seaborn
- bottleneck
- statsmodels
- psyplot

This constructor informations are automatically used to create executables when
a new release is published and the executables are then uploaded to github.
