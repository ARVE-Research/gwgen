.. _fortran_api:

Fortran API Reference
=====================
This section documents the real weather generator which is written as a
FORTRAN program. All the FORTRAN files, including the Makefile are in the
``src`` directory of the ``gwgen`` package. If you initialize a new project,
those files are copied to the ``'<project-dir>/src'`` directory.

To compile the source code manually, you may go to the source directory and
simply run::

    make all

This will create an executable called ``'weathergen'`` in the same directory.
However, make sure that the right compiler is chosen in the ``Makefile`` (
by default: ``gfortran``).

This API reference is created automatically from the FORTRAN source code using
sphinx-fortran_ and is subject to further improvements.

.. _sphinx-fortran: https://github.com/VACUMM/sphinx-fortran


Submodules
----------

.. toctree::

    gwgen.src.main
    gwgen.src.weathergenmod
    gwgen.src.randomdistmod
    gwgen.src.csv_file
    gwgen.src.geohashmod
    gwgen.src.parametersmod
