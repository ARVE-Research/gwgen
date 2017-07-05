.. _install:


Installation
============


.. _install_fortran:

Installing only the FORTRAN program
-----------------------------------
If you are only interested in the FORTRAN source files for the weather
generator, checkout the gwgen_f90_ repository.

It is implemented as a submodule in this repository in the src_ directory

.. _src: https://github.com/ARVE-Research/gwgen/blob/master/gwgen/src
.. _gwgen_f90: https://github.com/ARVE-Research/gwgen_f90
.. _Github: https://github.com/ARVE-Research/gwgen

.. _install_full:

Installing the full GWGEN
-------------------------
If you not only want the source code, but also the
:ref:`parameterization <parameterization>` or the experiment organization
features of the model, you need the full python package which includes the
FORTRAN source files.

The code is hosted open-source on Github_ and can be downloaded via

.. code-block:: bash

    git clone https://github.com/ARVE-Research/gwgen.git

or as a zipped archive directly from Github_.

Installing the requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~
You need a FORTRAN 95 compiler (see :ref:`install_fortran`). Furthermore, you
need python and the  following python packages:

- model-organization_: For the command line utility and the experiments
  organization
- matplotlib_, seaborn_ and psyplot_: For the visualization
- xarray_, statsmodels_, `numpy and scipy`_: For the calculations
- dask_ (only necessary for the
  :class:`cross correlation <gwgen.parameterization.CrossCorrelation>`
  parameterization task)
- cartopy_: For the Kolmogorov-Smirnoff
  (:class:`ks <gwgen.evaluation.KSEvaluation>`) evaluation task.

The recommended way to install these requirements is using conda_. We provide
two methodologies here

1. Installing into an existing conda distribution
*************************************************
If you already have conda_ installed, we recommend you just use
:download:`this conda environment file <gwgen_environment.yml>` and create a
new virtual environment via

.. code-block:: bash

    conda env create -f gwgen_environment.yml
    source activate gwgen

2. Installation including conda
*******************************
With every new ``gwgen`` release, we also provide executables to install
conda and gwgen on our releases_ page. This is probably the easiest way to
install it.

After selecting the right file for your operating system (MacOS or Linux), you
can simply install it via

.. code-block:: bash

    bash <downloaded-file.sh>

and follow the instructions.

.. _model-organization: http://model-organization.readthedocs.io/en/latest/
.. _psyplot: http://psyplot.readthedocs.io/en/latest/
.. _numpy and scipy: https://docs.scipy.org/doc/
.. _statsmodels: http://statsmodels.sourceforge.net/
.. _matplotlib: http://matplotlib.org/
.. _xarray: http://xarray.pydata.org/en/stable/
.. _seaborn: http://seaborn.pydata.org/
.. _dask: http://dask.pydata.org/en/latest/
.. _cartopy: http://scitools.org.uk/cartopy/
.. _conda: https://www.continuum.io/downloads
.. _releases: https://github.com/ARVE-Research/gwgen/releases


Installing GWGEN
~~~~~~~~~~~~~~~~
After having successfully installed python, just install the gwgen package via

.. code-block:: bash

    python setup.py install

You can test whether it was successfully installed by typing::

    gwgen -h

.. note::

    If you download the repository from Github_, you have to initialize the
    ``src`` submodule via::

        git submodule update --init gwgen/src
