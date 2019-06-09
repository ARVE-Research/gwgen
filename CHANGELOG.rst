v1.0.3
======
Compatibility fixes for pandas and dask, and bug fix for random number generator based on https://github.com/ARVE-Research/gwgen_f90/pull/1

v1.0.2
======
Final version for Geoscientific Model Development

Changed
-------
* Minor changes in LICENSE file

v1.0.1
======
Added
-----
* Added changelog

Changed
-------
* Changed parameterization of temperatures and winds standard deviation
* Implemented default parameterization based on `Sommer and Kaplan, 2017`_
* Several bug fixes in ``'mo_parseghcnrow.f90'``
* To decrease the size of the github repository, the old repository of
  GWGEN v1.0.0 has been  deleted and recreated. These files can be accessed
  through the gwgen_old_ repository
* The FORTRAN src files have been moved to an own repository, the gwgen_f90_
  repository, and are not implemented as a submodule. Hence, when cloning the
  gwgen repository from github, one has to execute::

    git submodule update --init gwgen/src

.. _Sommer and Kaplan, 2017: https://doi.org/10.5194/gmd-2017-42
.. _gwgen_old: https://github.com/ARVE-Research/gwgen_old
.. _gwgen_f90: https://github.com/ARVE-Research/gwgen_f90
