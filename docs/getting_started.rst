.. _getting_started:

Getting started
===============

.. _running_raw:

Running the raw ``weathergen``
------------------------------
After successfully :ref:`installing <install_fortran>` the FORTRAN files,
you can run the weather generator by typing::

    ./weathergen <input-file.csv> <output-file.csv>

``<input-file.csv>`` thereby is a csv-file (with header) containing the columns
(the order is important!)

station id (character string of 11 characters)
    a unique identifier of the weather station
lon (float)
    the longitude of the weather station (only necessary if the
    ``use_geohash``  namelist parameter in the main namelist of
    ``weathergen.nml`` is True (the default))
lat (float)
    the latitude of the weather station (see ``lon``)
year (int)
    the year of the month
month (int)
    the month
min. temperature (float)
    the minimum temperature degrees Celsius
max. temperature (float)
    the maximum temperature in degrees Celsius
cloud fraction (float)
    the mean cloud fraction during the month between 0 and 1
wind speed (float)
    the mean wind speed during the month in m/s
precipitation (float)
    the total precipitation in the month in mm/day
wet (int)
    the number of wet days in the month

The output file will contain the same columns except lon and lat but with an
additional day column


Using the python package
------------------------
The GWGEN package uses the model-organization_ framework and thus can be used
from the command line. The corresponding subclass of the
:class:`model_organization.ModelOrganizer` is the
:class:`gwgen.main.GWGENOrganizer` class.

After :ref:`having installed the full python package <install_full>` you can
setup a new project with the :ref:`gwgen.setup` command via

.. ipython::

    @suppress
    In [1]: import os
       ...: os.environ['PYTHONWARNINGS'] = "ignore"

    In [1]: !gwgen setup . -p my_first_project

This will copy the fortran source files into `my_first_project/src` where you
can then modify them according to your needs. To compile the model, use the
:ref:`gwgen.compile` command:

.. ipython::

    In [2]: !gwgen compile -p my_first_project

Note that you can also omit the ``-p`` option. If you do so, it uses the last
created project.

To create a new experiment inside the project, use the :ref:`gwgen.init`
command:

.. ipython::

    In [3]: !gwgen -id my_first_experiment init -p my_first_project

To run the weather generator, use the :ref:`gwgen.run` command

.. ipython::
    :verbatim:

    In [4]: !gwgen -id my_first_experiment run -i <input-file.csv>

(see :ref:`running_raw` for the format of ``<input-file.csv>``). Note that you
can as well omit the ``-id`` option. By doing so, it uses the last created
experiment.

.. _parameterization:

Parameterizing an experiment
----------------------------
The default parameterization of the weather generator uses about 8500 stations
world wide from the [GHCN]_ database and 8500 stations from the [EECRA]_
database. I you however want to have your own parameterization, you can use the
:ref:`gwgen.param` command.

The parameterization is split up into :class:`tasks <gwgen.utils.TaskBase>`,
where each :class:`task <gwgen.utils.TaskBase>` processed a given set of GHCN
or EECRA stations. Each task that is used for the parameterization, requires
some intermediate tasks. For example, the :ref:`prcp <gwgen.param.prcp>` task
that determines the relationship between mean precipitation, number of wet days
and the precipitation distribution parameters, requires the
:ref:`reading and downloading of the daily GHCN data <gwgen.param.day>`, the
:ref:`calculation of the monthly averages <gwgen.param.month>` and the
:ref:`extraction of the complete months <gwgen.param.cmonth>`. However, these
dependencies are specified in the corresponding
:class:`~gwgen.parametization.Parameterizer` subclass
(e.g. :class:`gwgen.parameterization.PrcpDistParams`) and the only question you
have to take care about is: What stations do you want to use for the
parameterization? You can use the climap_ to select the stations you need for
your region but note that you should have as many weather stations as you can.

For our demonstration, we only use two weather stations from Hamburg:

GM000010147
    Hamburg-Fuhlsbüttel
GM000003865
    Hamburg-Bergedorf

and save the corresponding IDs in a file

.. ipython::

    In [5]: with open('hamburg_stations.dat', 'w') as f:
       ...:     f.write('GM000010147\n')
       ...:     f.write('GM000003865')

then, we use the :ref:`day <gwgen.param.day>` task to download the necessary
data files and run our :ref:`prcp <gwgen.param.prcp>` parameterization:

.. ipython::

    In [6]: !gwgen param -s hamburg_stations.dat day --download single prcp

    @suppress
    In [6]: !echo 'bins: 100\ndensity: "kde"\nxrange: [3, 6]' > fmt.yml
       ...: !psyplot -p my_first_project/experiments/my_first_experiment/parameterization/prcp.pkl -o _static/prcp%i.png -fmt fmt.yml

This then also creates a plot that shows the relation ship between the mean
precipitation on wet days (as it is calculated in the weather generator) and
the gamma shape parameter

.. image:: ./_static/prcp1.png
    :alt: Relation ship of mean precipitation and gamma shape parameter

This procedure now modified the namelist of our experiment and added two
namelist parameters

.. ipython::

    In [7]: !gwgen get-value namelist

Otherwise you can of course always modify the namelist of your experiment using
the :ref:`set-value <gwgen.set-value>` and :ref:`del-value <gwgen.del-value>`
commands or by modifying the configuration file (``gwgen info -ep``) by hand

.. note::
    Since we use the psyplot_ package, you can also easily change the
    above created plot after the parameterization. For example, to change the
    number of bins in the density plot of the above plot, just load the created
    psyplot project and update the plot:

    .. ipython::
        :verbatim:

        In [8]: import psyplot.project as psy

        In [9]: p = psy.Project.load_project(
           ...:     'my_first_project/experiments/my_first_experiment/'
           ...:     'parameterization/prcp.pkl')

        In [10]: p.update(bins=20)

.. ipython::
    :suppress:

    In [10]: !gwgen remove -p my_first_project -ay

    In [11]: !rm hamburg_stations.dat fmt.yml


.. _model-organization: http://model-organization.readthedocs.io/en/latest/
.. _psyplot: http://psyplot.readthedocs.io/en/latest/
.. _climap: http://arve.unil.ch/climap/
