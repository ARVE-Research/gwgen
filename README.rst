GWGEN: A global weather generator for daily data
================================================


.. start-badges

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - tests
      - |travis| |coveralls|
    * - package
      - |version| |supported-versions| |supported-implementations|

.. |travis| image:: https://travis-ci.org/ARVE-Research/gwgen.svg?branch=master
    :alt: Travis
    :target: https://travis-ci.org/ARVE-Research/gwgen

.. |coveralls| image:: https://coveralls.io/repos/github/ARVE-Research/gwgen/badge.svg?branch=master
    :alt: Coverage
    :target: https://coveralls.io/github/ARVE-Research/gwgen?branch=master

.. |version| image:: https://img.shields.io/pypi/v/gwgen.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/gwgen

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/gwgen.svg?style=flat
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/gwgen

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/gwgen.svg?style=flat
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/gwgen

.. end-badges


Welcome! This synthesis of FORTRAN and Python is a globally applicable
weather generator inspired by the original WGEN weather generator of
[Richardson1981]_ and parameterized through a global dataset of [GHCN]_ and
[EECRA]_ data. A paper with the scientific documentation is in progress, the
`technical documentation`_ with more information is hosted on github.io.

.. _technical documentation: https://arve-research.github.io/gwgen/


Authors
-------
This package has been developped by `Philipp S. Sommer`_ and `Jed O. Kaplan`_.

Citing GWGEN
------------
When using GWGEN, we kindly ask you to provide a reference to the corresponding
paper published in Geoscientific Model Development:

    Sommer, P. S. and Kaplan, J. O.: *A globally calibrated scheme for
    generating daily meteorology from monthly statistics: Global-WGEN (GWGEN)
    v1.0*, Geosci. Model Dev., 10, 3771-3791, DOI: `10.5194/gmd-10-3771-2017`_,
    2017.

.. _10.5194/gmd-10-3771-2017: https://doi.org/10.5194/gmd-10-3771-2017


Acknowledgements
----------------
This work was supported by the European Research Council (COEVOLVE, 313797) and
the Swiss National Science Foundation (ACACIA, CR10I2\_146314). We thank
`Shawn Koppenhoefer`_ for assistance compiling and querying the weather databases and
Alexis Berne and Grégoire Mariéthoz for helpful suggestions on the analyses. We
are grateful to NOAA NCDC and the University of Washingtion for providing free
of charge the GHCN-Daily and EECRA databases, respectively.

.. _Philipp S. Sommer: https://github.com/Chilipp
.. _Jed O. Kaplan: https://github.com/jedokaplan
.. _Shawn Koppenhoefer: http://arve.unil.ch/people/shawn-koppenhoefer/


References
----------
.. [Richardson1981] Richardson, C. W.: *Stochastic simulation of daily
    precipitation, temperature, and solar radiation*, Water Resources Research,
    17, 182–190, doi:10.1029/WR017i001p00182, 1981.
.. [GHCN] T. G.: Global Historical Climatology Network - Daily (GHCN-Daily),
    Version 3.22, doi:10.7289/V5D21VHZ, doi:10.7289/V5D21VHZ, 2012
.. [EECRA] Hahn, C. and Warren, S.: *Extended Edited Synoptic Cloud Reports from
    Ships and Land Stations Over the Globe*, 1952-1996 (with Ship data
    updated through 2008), doi:10.3334/CDIAC/cli.ndp026c, 1999
