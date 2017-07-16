from __future__ import print_function, division
import os
import os.path as osp
import six
import re
import shutil
import sys
import datetime as dt
from itertools import repeat
from argparse import Namespace, RawTextHelpFormatter
import logging
import numpy as np
import gwgen.utils as utils
from gwgen.utils import docstrings
from model_organization import ModelOrganizer
from model_organization.config import ordered_yaml_dump

from collections import OrderedDict


class GWGENOrganizer(ModelOrganizer):
    """
    A class for organizing a model

    This class is indended to have hold the basic functions for organizing a
    model. You can subclass the functions ``setup, init`` to fit to your model.
    When using the model from the command line, you can also use the
    :meth:`setup_parser` method to create the argument parsers"""

    commands = ModelOrganizer.commands
    commands.insert(commands.index('init'), 'compile_model')
    commands.insert(commands.index('archive'), 'preproc')
    commands.insert(commands.index('archive'), 'param')
    commands.insert(commands.index('archive'), 'run')
    commands.insert(commands.index('archive'), 'evaluate')
    commands.insert(commands.index('archive'), 'bias_correction')
    commands.insert(commands.index('archive'), 'sensitivity_analysis')

    #: mapping from the name of the parser command to the method name
    parser_commands = {'compile_model': 'compile',
                       'sensitivity_analysis': 'sens',
                       'bias_correction': 'bias'}

    #: list of str. The keys describing paths for the model
    paths = ['expdir', 'src', 'data', 'param_stations', 'eval_stations',
             'indir', 'input', 'outdir', 'outdata', 'nc_file',  'project_file',
             'plot_file', 'reference', 'evaldir', 'paramdir', 'workdir',
             'param_grid', 'grid', 'eval_grid']

    name = 'gwgen'

    # -------------------------------------------------------------------------
    # --------------------------- Infrastructure ------------------------------
    # ---------- General parts for organizing the model infrastructure --------
    # -------------------------------------------------------------------------

    docstrings.get_sectionsf('ModelOrganizer.setup')(
        docstrings.dedent(ModelOrganizer.setup))

    @docstrings.dedent
    def setup(self, root_dir, projectname=None, link=False, src_project=None,
              compiler=None, **kwargs):
        """
        Perform the initial setup for the model

        Parameters
        ----------
        %(ModelOrganizer.setup.parameters)s
        link: bool
            If set, the source files are linked to the original ones instead
            of copied
        src_project: str
            Another model name to use the source model files from
        compiler: str
            The path to the compiler to use. If None, the global compiler
            option is used
        """
        root_dir = super(GWGENOrganizer, self).setup(
            root_dir, projectname=projectname, **kwargs)
        self.config.projects[self.projectname]['src'] = src_dir = osp.join(
            root_dir, 'src')
        if not osp.exists(src_dir):
            os.makedirs(src_dir)
        if src_project:
            module_src = self.config.projects[src_project]['src']
        else:
            module_src = osp.join(osp.dirname(__file__), 'src')
        for f in os.listdir(module_src):
            target = osp.join(src_dir, f)
            if osp.exists(target):
                os.remove(target)
            if link:
                self._link(osp.join(module_src, f), target)
            else:
                shutil.copyfile(osp.join(module_src, f), target)
        compiler = compiler or self.global_config.get('compiler')
        if compiler is not None:
            with open(osp.join(src_dir, 'Makefile')) as f:
                make_file = f.read()
            make_file = re.sub('^\s*FC\s*=\s*.*$', 'FC = ' + compiler,
                               make_file, flags=re.MULTILINE)
            with open(osp.join(src_dir, 'Makefile'), 'w') as f:
                f.write(make_file)
        return root_dir

    def _modify_setup(self, parser):
        parser.setup_args(ModelOrganizer.setup)
        self._modify_app_main(parser)
        parser.update_arg('src_project', short='src')
        parser.update_arg('compiler', short='c')

    @docstrings.dedent
    def compile_model(self, projectname=None, **kwargs):
        """
        Compile the model

        Parameters
        ----------
        projectname: str
            The name of the project. If None, use the last one or the one
            specified by the current experiment
        ``**kwargs``
            Keyword arguments passed to the :meth:`app_main` method
        """
        import subprocess as spr
        self.app_main(**kwargs)
        projectname = projectname or self.projectname
        self.projectname = projectname
        self.logger.info("Compiling %s", projectname)
        pdict = self.config.projects[projectname]
        pdict['bindir'] = bin_dir = osp.join(pdict['root'], 'bin')
        pdict['bin'] = osp.join(bin_dir, 'weathergen')
        src_dir = self.abspath(pdict['src'])
        if not os.path.exists(bin_dir):
            self.logger.debug("    Creating bin directory %s", bin_dir)
            os.makedirs(bin_dir)
        for f in os.listdir(src_dir):
            self.logger.debug("    Linking %s...", f)
            target = osp.join(bin_dir, f)
            if osp.exists(target):
                os.remove(target)
            self._link(osp.join(src_dir, f), target)
        spr.check_call(['make', '-C', bin_dir, 'all'], stdout=sys.stdout,
                       stderr=sys.stderr)
        self.logger.debug('Compilation done.')
        ts = self.project_config['timestamps']
        ts['compile'] = ts['compile_model '] = dt.datetime.now()

    def _modify_compile_model(self, parser):
        """Does nothing since compile takes no special arguments"""
        self._modify_app_main(parser)

    # -------------------------------------------------------------------------
    # -------------------------- Configuration --------------------------------
    # ------------------ Parts for configuring the organizer ------------------
    # -------------------------------------------------------------------------

    docstrings.get_sectionsf("ModelOrganizer.configure")(docstrings.dedent(
        ModelOrganizer.configure))

    @docstrings.dedent
    def configure(self, update_nml=None, max_stations=None,
                  datadir=None, database=None, user=None, host=None, port=None,
                  chunksize=None, compiler=None, **kwargs):
        """
        Configure the projects and experiments

        Parameters
        ----------
        %(ModelOrganizer.configure.parameters)s
        update_nml: str or dict
            A python dict or path to a namelist to use for updating the
            namelist of the experiment
        max_stations: int
            The maximum number of stations to process in one parameterization
            process. Does automatically impact global settings
        datadir: str
            Path to the data directory to use (impacts the project
            configuration)
        database: str
            The name of a postgres data base to write the data to
        user: str
            The username to use when logging into the database
        host: str
            the host which runs the database server
        port: int
            The port to use to log into the the database
        chunksize: int
            The chunksize to use for the parameterization and evaluation
        compiler: str
            The path to the fortran compiler to use"""
        super(GWGENOrganizer, self).configure(**kwargs)
        exp_config = self.exp_config
        if update_nml is not None:
            import f90nml
            with open(update_nml) as f:
                ref_nml = f90nml.read(f)
            nml2use = exp_config.setdefault('namelist', OrderedDict())
            for key, nml in ref_nml.items():
                nml2use.setdefault(key, OrderedDict()).update(dict(nml))
        gconf = self.config.global_config
        if max_stations:
            gconf['max_stations'] = max_stations
        if datadir:
            datadir = osp.abspath(datadir)
            self.project_config['data'] = datadir
        if database is not None:
            exp_config['database'] = database
        if user is not None:
            gconf['user'] = user
        if port is not None:
            gconf['port'] = port
        if host is not None:
            gconf['host'] = '127.0.0.1'
        if chunksize is not None:
            gconf['chunksize'] = chunksize
        if compiler is not None:
            gconf['compiler'] = compiler

    def _modify_configure(self, parser):
        parser.setup_args(super(GWGENOrganizer, self).configure)
        super(GWGENOrganizer, self)._modify_configure(parser)
        parser.update_arg('datadir', short='d')
        parser.update_arg('update_nml', short='u')
        parser.update_arg('max_stations', short='max', type=int)
        parser.update_arg('database', short='db')
        parser.update_arg('compiler', short='c')

    # -------------------------------------------------------------------------
    # -------------------------- Preprocessing --------------------------------
    # -------------- Preprocessing functions for the experiment ---------------
    # -------------------------------------------------------------------------

    @property
    def preproc_funcs(self):
        """A mapping from preproc commands to the corresponding function"""
        return {'select': self.select,
                'cloud': self.cloud_preproc,
                'test': self.create_test_sample}

    @docstrings.dedent
    def preproc(self, **kwargs):
        """
        Preprocess the data

        Parameters
        ----------
        ``**kwargs``
            Any keyword from the :attr:`preproc` attribute with kws for the
            corresponding function, or any keyword for the :meth:`main` method
        """
        funcs = self.preproc_funcs
        sp_kws = {key: kwargs.pop(key) for key in set(kwargs).intersection(
            funcs)}
        self.app_main(**kwargs)
        exp_config = self.fix_paths(self.exp_config)
        outdir = exp_config.setdefault('indir', osp.join(
            exp_config['expdir'], 'input'))
        if not osp.exists(outdir):
            os.makedirs(outdir)

        preproc_config = exp_config.setdefault('preproc', OrderedDict())

        for key, val in sp_kws.items():
            if isinstance(val, Namespace):
                val = vars(val)
            info = funcs[key](**val)
            if info:
                preproc_config[key] = info

    def _modify_preproc(self, parser):
        from gwgen.preproc import CloudPreproc
        self._modify_app_main(parser)
        sps = parser.add_subparsers(title='Preprocessing tasks', chain=True)

        # select
        sp = sps.add_parser(
            'select', help='Select stations based upon a regular grid')
        sp.setup_args(self.select)
        sp.update_arg('grid', short='g')
        sp.update_arg('grid_output', short='og')
        sp.update_arg('stations_output', short='os')
        sp.update_arg('igrid_key', short='k')
        sp.update_arg('grid_key', short='ok')
        sp.update_arg('grid_db', short='gdb')
        sp.update_arg('stations_db', short='sdb')
        sp.update_arg('no_prcp_check', short='nc')
        sp.update_arg('setup_from', short='f', long='from',
                      dest='setup_from')
        sp.update_arg('download', short='d', choices=['single', 'all'])

        # cloud preprocessing
        sp = sps.add_parser('cloud', help='Cloud preprocessing')
        sp.setup_args(self.cloud_preproc)
        sp.update_arg('max_files', short='mf', type=int)
        sp.pop_arg('return_manager')
        self._modify_task_parser(sp, CloudPreproc)

        # test samples
        sp = sps.add_parser(
            'test', help='Create a test sample for selected GHCN stations')
        sp.setup_args(self.create_test_sample)
        sp.update_arg('no_cloud', short='nc')
        sp.update_arg('reduce_eecra', short='re', type=float)
        sp.update_arg('keep_all', short='a')

        return parser

    # ------------------------------- Selection -------------------------------

    def _prcp_check(self, series):
        try:
            return 11 == len(series.to_frame().set_index('prcp').join(
                self._prcp_test, how='inner').prcp.unique())
        except:
            return None

    def _select_best_df(self, df, test_series, kws):
        from gwgen.parameterization import DailyGHCNData
        # disable logging for the DailyGHCNData task
        task_logger = DailyGHCNData([], self.exp_config, self.project_config,
                                    self.global_config).logger
        orig_level = task_logger.level
        task_logger.setLevel(logging.WARNING)
        self._test_series = test_series
        self._select_kws = kws
        self._select_task = DailyGHCNData
        g = df.sort_values('nyrs', ascending=False).groupby(
            level=['clon', 'clat'])
        ret = g.id.agg(self._select_best)
        task_logger.setLevel(orig_level)
        return ret

    def _select_best(self, series):
        test_series = self._test_series
        for station in series.values:
            task = self._select_task(
                np.array([station]), self.exp_config, self.project_config,
                self.global_config, **self._select_kws)
            try:
                task.init_task()
            except FileNotFoundError as e:
                task.logger.warn(e)
            else:
                task.setup()
                if len(test_series) == len(
                    task.data.set_index('prcp').join(
                        test_series, how='inner').prcp.unique()):
                    return station
        return series.values[0]

    @staticmethod
    def _parallel_select(l):
        organizer, df, test_series, kws = l
        return organizer._select_best_df(df, test_series, kws)

    @docstrings.dedent
    def select(self, grid=None, grid_output=None, stations_output=None,
               igrid_key=None, grid_key=None, grid_db=None, stations_db=None,
               no_prcp_check=False, setup_from=None, download=None, **kwargs):
        """
        Select stations based upon a regular grid

        Parameters
        ----------
        grid: str
            The path to a csv-file containing a lat and a lon column with the
            information on the centers of the grid. If None, `igrid_key` must
            not be None and point to a key in the configuration (either the one
            of the experiment, or the project, or the global configuration)
            specifying the path
        grid_output: str
            The path to the csv-file where to store the mapping from grid
            lat-lon to station id.
        stations_output: str
            The path to the csv-file where to store the mapping from station
            to grid center point
        igrid_key: str
            The key in the configuration where to store the path of the `grid`
            input file
        grid_key: str
            The key in the configuration where to store the name of the
            `grid_output` file.
        grid_db: str
            The name of a data table to store the data of `stations_output` in
        stations_db: str
            The name of a data table to store the data for `stations_output` in
        no_prcp_check: bool
            If True, we will not check for the values between 0.1 and 1.0 for
            precipitation and save the result in the ``'best'`` column
        setup_from: { 'scratch' | 'file' | 'db' }
            The setup method for the daily data for the prcp check
        download: { 'single' | 'all' }
            Handles how to manage missing files for the prcp check. If None
            (default), an warning is printed and the file is ignored, if
            ``'single'``, the missing file is downloaded, if ``'all'``, the
            entire tarball is downloaded (strongly not recommended for this
            function)

        Other Parameters
        ----------------
        ``**kwargs``
            are passed to the :meth:`main` method

        Notes
        -----
        for `igrid_key` and `ogrid_key` we recommend one of
        ``{'grid', 'param_grid', 'eval_grid'`` because that implies a
        correct path management
        """
        from gwgen.evaluation import EvaluationPreparation
        import numpy as np
        import scipy.spatial
        import pandas as pd

        logger = self.logger

        if grid is None:
            if igrid_key is not None:
                grid = self.exp_config.get(igrid_key, self.project_config.get(
                    igrid_key, self.global_config.get(igrid_key)))
            else:
                raise ValueError(
                    "No grid file or configuration key specified!")
        if grid is None:
            raise ValueError(
                    "No grid file specified and '%s' could not be found in "
                    "the configuration!" % igrid_key)
        t = EvaluationPreparation(np.array([]), self.exp_config,
                                  self.project_config, self.global_config)
        # get inventory
        t.download_src()
        df_stations = t.station_list
        df_stations = df_stations[df_stations.vname == 'PRCP'].drop(
            'vname', 1).reset_index()  # reset_index required due to filtering
        df_stations['nyrs'] = df_stations.lastyr - df_stations.firstyr

        # read 1D grid information
        df_centers = pd.read_csv(grid)
        df_centers.rename(columns={'lon': 'clon', 'lat': 'clat'}, inplace=True)

        # concatenate lat and lon values into x-y points
        center_points = np.dstack(
            [df_centers.clat.values, df_centers.clon.values])[0]
        station_points = np.dstack([df_stations.lat, df_stations.lon])[0]

        # look up the nearest neighbor
        logger.debug('Searching neighbors...')
        kdtree = scipy.spatial.cKDTree(center_points)
        dist, indexes = kdtree.query(station_points)
        logger.debug('Done.')

        # store the lat and longitude of, and the distance to the center grid
        # point in the stations table
        df_stations['clon'] = df_centers.clon.values[indexes]
        df_stations['clat'] = df_centers.clat.values[indexes]
        df_stations['dist'] = dist

        # --------- stations with the closest distance to grid center ---------
        # group by the center coordinates and look for the index with the
        # smallest distance
        g = df_stations.sort_index().groupby(['clon', 'clat'])
        indices_closest = g.dist.idxmin()
        indices_longest = g.nyrs.idxmax()
        # merge the nearest stations into the centers table
        df_centers.set_index(['clon', 'clat'], inplace=True)
        df_stations.set_index(['clon', 'clat'], inplace=True)
        merged = df_centers.merge(
            df_stations.ix[indices_closest][['id']].rename(
                columns={'id': 'nearest_station'}),
            left_index=True, right_index=True, how='outer')
        merged = merged.merge(
            df_stations.ix[indices_longest][['id']].rename(
                columns={'id': 'longest_record'}),
            left_index=True, right_index=True, how='outer')

        if not no_prcp_check:
            test_series = pd.Series(
                np.arange(0.1, 1.05, 0.1), name='prcp')
            logger.debug('Performing best station check with %s',
                         test_series.values)
            kws = dict(download=download, setup_from=setup_from)
            if not self.global_config.get('serial'):
                import multiprocessing as mp
                nprocs = self.global_config.get('nprocs', 'all')
                lonlats = np.unique(df_stations.dropna(0).index.values)
                if nprocs == 'all':
                    nprocs = mp.cpu_count()
                splitted = np.array_split(lonlats, nprocs)
                try:
                    nprocs = list(map(len, splitted)).index(0)
                except ValueError:
                    pass
                else:
                    splitted = splitted[:nprocs]
                dfs = [df_stations.loc[list(arr)] for arr in splitted]
                # initializing pool
                logger.debug('Start %i processes', nprocs)
                pool = mp.Pool(nprocs)
                args = list(zip(repeat(self), dfs, repeat(test_series),
                                repeat(kws)))
                res = pool.map_async(self._parallel_select, args)
                best = pd.concat(res.get())
                pool.close()
                pool.join()
                pool.terminate()
            else:
                best = self._select_best_df(
                    df_stations.dropna(0), test_series, kws)
            merged = merged.merge(
                best.to_frame().rename(columns={'id': 'best'}),
                left_index=True, right_index=True, how='outer')

        if igrid_key:
            self.exp_config[igrid_key] = grid
        if stations_output:
            logger.debug('Dumping to%s %s',
                         ' exisiting' if osp.exists(stations_output) else '',
                         stations_output)
            utils.safe_csv_append(df_stations, stations_output)

        if grid_output:
            logger.debug('Dumping to%s %s',
                         ' exisiting' if osp.exists(grid_output) else '',
                         grid_output)
            utils.safe_csv_append(merged, grid_output)
            if grid_key is not None:
                self.exp_config[grid_key] = grid_output
        if stations_db or grid_db:
            conn = t.engine.connect()
            if stations_db:
                logger.info('Writing %i lines into %s', len(df_stations),
                            stations_db)
                df_stations.to_sql(stations_db, conn, if_exists='append')
            if grid_db:
                logger.info('Writing %i lines into %s', len(merged),
                            grid_db)
                merged.to_sql(grid_db, conn, if_exists='append')
            conn.close()

        return df_stations, merged

    # --------------------------- Cloud inventory -----------------------------

    @docstrings.dedent
    def cloud_preproc(self, max_files=None, return_manager=False, **kwargs):
        """
        Extract the inventory of EECRA stations

        Parameters
        ----------
        max_files: int
            The maximum number of files to process during one process. If None,
            it is determined by the global ``'max_stations'`` key
        ``**kwargs``
            Any task in the :class:`gwgen.preproc.CloudPreproc` framework
        """
        from gwgen.preproc import CloudPreproc
        from gwgen.parameterization import HourlyCloud
        stations_orig = self.global_config.get('max_stations')
        if max_files is not None:
            self.global_config['max_stations'] = max_files
        files = HourlyCloud.from_organizer(self, []).raw_src_files
        manager = CloudPreproc.get_manager(config=self.global_config)
        for key, val in kwargs.items():
            if isinstance(val, Namespace):
                kwargs[key] = val = vars(val)
                val.pop('max_files', None)
        self._setup_manager(manager, stations=list(files.values()),
                            base_kws=kwargs)
        d = {}
        manager.run(d)
        if stations_orig:
            self.global_config['max_stations'] = stations_orig
        else:
            self.global_config.pop('max_stations', None)
        if return_manager:
            return d, manager
        else:
            return d

    # --------------------------- Parameterization ----------------------------

    @docstrings.get_sectionsf('GWGENOrganizer.param')
    @docstrings.dedent
    def param(self, complete=False, stations=None, other_exp=None,
              setup_from=None, to_db=None, to_csv=None, database=None,
              norun=False, to_return=None, **kwargs):
        """
        Parameterize the experiment

        Parameters
        ----------
        stations: str or list of str
            either a list of stations to use or a filename containing a
            1-row table with stations
        other_exp: str
            Use the configuration from another experiment
        setup_from: str
            Determine where to get the data from. If `scratch`, the
            data will be calculated from the raw data. If `file`,
            the data will be loaded from a file, if `db`, the data
            will be loaded from a postgres database (Note that the
            `database` argument must be provided!).
        to_db: bool
            Save the data into a postgresql database (Note that the
            `database` argument must be provided!)
        to_csv: bool
            Save the data into a csv file
        database: str
            The name of a postgres data base to write the data to
        norun: bool, list of str or ``'all'``
            If True, only the data is set up and the configuration of the
            experiment is not affected. It can be either a list of  tasks or
            True or ``'all'``
        to_return: list of str or ``'all'``
            The names of the tasks to return. If None, only the ones with an
            :attr:`gwgen.utils.TaskBase.has_run` are returned.
        complete: bool
            If True, setup and run all possible tasks
        """
        from gwgen.parameterization import Parameterizer
        task_names = [task.name for task in Parameterizer._registry]
        parameterizer_kws = {
            key: vars(val) if isinstance(val, Namespace) else dict(val)
            for key, val in kwargs.items() if key in task_names}
        main_kws = {key: val for key, val in kwargs.items()
                    if key not in task_names}
        self.app_main(**main_kws)
        experiment = self.experiment
        exp_dict = self.fix_paths(self.config.experiments[experiment])
        param_dir = exp_dict.setdefault(
            'paramdir', osp.join(exp_dict['expdir'], 'parameterization'))
        if not osp.exists(param_dir):
            os.makedirs(param_dir)
        projectname = self.projectname
        logger = self.logger
        logger.info("Parameterizing experiment %s of project %s",
                    experiment, projectname)
        stations = self._get_stations(stations, other_exp, param_dir,
                                      'param_stations')
        global_conf = self.config.global_config
        # choose keywords for data processing
        manager = Parameterizer.get_manager(config=global_conf)
        self._setup_manager(manager, stations, other_exp, setup_from, to_db,
                            to_csv, database, to_return, complete,
                            parameterizer_kws)
        # update experiment namelist and configuration
        if not norun:
            manager.run(exp_dict.setdefault('parameterization', OrderedDict()),
                        exp_dict.setdefault('namelist', OrderedDict()))
        return manager

    def _modify_param(self, parser, *args, **kwargs):
        from gwgen.parameterization import Parameterizer
        self._modify_task_parser(parser, Parameterizer, *args, **kwargs)

    # --------------------------------- Test ----------------------------------

    @docstrings.dedent
    def create_test_sample(self, test_dir, stations, no_cloud=False,
                           reduce_eecra=0, keep_all=False):
        """
        Create a test sample for the given GHCN stations

        Parameters
        ----------
        test_dir: str
            The path to the directory containing the test files from Github
        stations: str or list of str
            either a list of GHCN stations to use or a filename containing a
            1-row table with GHCN stations
        no_cloud: bool
            If True, no cloud stations are extracted
        reduce_eecra: float
            The percentage by which to reduce the EECRA data
        keep_all: bool
            If True all years of the EECRA data are used. Otherwise, only the
            years with complete temperature and cloud are kept. Note
            that this has only an effect if `reduce_eecra` is not 0
        """
        import calendar
        import pandas as pd
        from gwgen.parameterization import DailyGHCNData, HourlyCloud

        def is_complete(s):
            ndays = 366 if calendar.isleap(s.name[1]) else 365
            s[:] = s.ix[~s.index.duplicated()].count() == ndays
            return s

        stations = self._get_stations(stations)
        np.savetxt(osp.join(test_dir, 'test_stations.dat'), stations, fmt='%s')
        # download the GHCN data
        ghcn_task = DailyGHCNData.from_organizer(self, stations,
                                                 download='single')
        ghcn_task.init_from_scratch()
        data_dir = super(DailyGHCNData, ghcn_task).data_dir

        if not no_cloud:
            eecra_task = HourlyCloud.from_organizer(self, stations)
            if len(eecra_task.stations) == 0:
                raise ValueError(
                    "Could not find any station in the given stations %s!",
                    ', '.join(stations))
            np.savetxt(osp.join(test_dir, 'eecra_test_stations.dat'),
                       eecra_task.eecra_stations, fmt='%i')
            eecra_task.init_from_scratch()

        for fname in ghcn_task.raw_src_files:
            target = fname.replace(osp.join(data_dir, ''),
                                   osp.join(test_dir, ''))
            if not osp.samefile(fname, target):
                shutil.copyfile(fname, target)
            shutil.make_archive(osp.join(test_dir, 'ghcn', 'ghcnd_all'),
                                'gztar',
                                root_dir=osp.join(test_dir, 'ghcn'),
                                base_dir='ghcnd_all')

        if not no_cloud:
            for fname in eecra_task.src_files:
                target = fname.replace(osp.join(data_dir, ''),
                                       osp.join(test_dir, ''))
                if not reduce_eecra and not osp.samefile(fname, target):
                    shutil.copyfile(fname, target)
                else:
                    df = pd.read_csv(fname)
                    if not keep_all:
                        df_bool = df.set_index(
                            ['station_id', 'year', 'month', 'day'])[[
                                'ww', 'AT', 'N']]
                        for col in df_bool.columns:
                            df_bool[col] = df_bool[col].astype(bool)
                        g = df_bool.groupby(level=['station_id', 'year'])
                        mask = g.transform(is_complete).values.any(axis=1)
                        df = df.ix[mask]

                    g = df.groupby(['station_id', 'year'],
                                   as_index=False)
                    tot = g.ngroups
                    n = np.ceil(tot * (100 - reduce_eecra) / 100)
                    idx_groups = iter(sorted(np.random.permutation(tot)[:n]))
                    self.logger.debug(
                        'Saving EECRA test sample with %i years from %i to '
                        '%s', n, tot, target)
                    df.ix[1:0].to_csv(target, index=False)
                    igrp = next(idx_groups)
                    for i, (key, group) in enumerate(g):
                        if i == igrp:
                            group.to_csv(target, header=False, mode='a',
                                         index=False)
                            igrp = next(idx_groups, -1)
    # -------------------------------------------------------------------------
    # ------------------------------- Run -------------------------------------
    # --------------------------- Run the experiment --------------------------
    # -------------------------------------------------------------------------

    @docstrings.get_sectionsf('GWGENOrganizer.run')
    @docstrings.dedent
    def run(self, ifile=None, ofile=None, odir=None, work_dir=None,
            remove=False, **kwargs):
        """
        Run the experiment

        Parameters
        ----------
        ifile: str
            The path to the input file. If None, it is assumed that it is
            stored in the ``'input'`` key in the experiment configuration
        ofile: str
            The path to the output file.  If None, it is assumed that it is
            stored in the ``'input'`` key in the experiment configuration or
            it will be stored in ``'odir/exp_id.csv'``. The output directory
            ``'odir'`` is determined by the `odir` parameter
        odir: str
            The path to the output directory. If None and not already saved
            in the configuration, it will default to
            ``'experiment_dir/outdata'``
        work_dir: str
            The path to the work directory where the binaries are copied to.
        remove: bool
            If True, the `work_dir` will be removed if it already exists

        Other Parameters
        ----------------
        ``**kwargs``
            Will be passed to the :meth:`main` method
        """
        import subprocess as spr
        import stat
        import f90nml
        from copy import deepcopy
        self.app_main(**kwargs)
        logger = self.logger
        exp_config = self.fix_paths(self.exp_config)
        project_config = self.fix_paths(self.project_config)
        experiment = self.experiment
        if not {'compile_model', 'compile'} & set(
                project_config['timestamps']):
            self.compile_model(**kwargs)
        logger.info("Running experiment %s of project %s",
                    experiment, self.projectname)
        if ifile is None:
            ifile = exp_config.get('input', self.project_config.get(
                'input',  self.global_config.get('input')))
        if ifile is None:
            raise ValueError("No input file specified!")
        if ofile is None:
            ofile = exp_config.get('outdata')
        if ofile is None:
            ofile = osp.join(
                odir or exp_config.get(
                    'outdir', osp.join(exp_config['expdir'], 'outdata')),
                str(experiment) + '.csv')
        if work_dir is None:
            work_dir = exp_config.get('workdir',
                                      osp.join(exp_config['expdir'], 'work'))
        exp_config['outdir'] = odir = osp.dirname(ofile)
        exp_config['outdata'] = ofile
        exp_config['input'] = ifile
        exp_config['indir'] = osp.dirname(ifile)
        exp_config['workdir'] = work_dir
        nml = exp_config.setdefault(
            'namelist', {'weathergen_ctl': OrderedDict(),
                         'main_ctl': OrderedDict()})
        for key in ['weathergen_ctl', 'main_ctl']:
            nml.setdefault(key, {})

        if osp.exists(work_dir) and remove:
            shutil.rmtree(work_dir)
        elif not osp.exists(work_dir):
            os.makedirs(work_dir)
        if not osp.exists(odir):
            os.makedirs(odir)

        f = project_config['bin']
        target = osp.join(work_dir, osp.basename(f))
        logger.debug('Copy executable %s to %s', f, target)
        shutil.copyfile(f, target)
        os.chmod(target, stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
        logger.debug('    Name list: %s', ordered_yaml_dump(nml))
        nml = deepcopy(nml)
        # transpose multidimensional arrays because they get transposed by
        # f90nml. Otherwise you get errors using functions like matmul
        for key, sub_nml in nml.items():
            for key2, val in sub_nml.items():
                if np.ndim(val) >= 2:
                    sub_nml[key2] = np.round(np.transpose(val), 8).tolist()
        with open(osp.join(work_dir, 'weathergen.nml'), 'w') as f:
            f90nml.write(nml, f)

        logger.debug('Running experiment...')
        logger.debug('    input: %s', ifile)
        logger.debug('    output: %s', ofile)
        t = dt.datetime.now()
        commands = 'cd %s && %s %s %s' % (work_dir, target, ifile, ofile)
        logger.debug(commands)
        spr.check_call(commands, stdout=sys.stdout, stderr=sys.stderr,
                       shell=True)
        err_msg = "Failed to run the experiment with '%s'!" % commands
        if not osp.exists(ofile):
            raise RuntimeError(
                (err_msg + "Reason: Output %s missing" % (ofile)))
        else:  # check if the file contains more than one line
            with open(ofile) as f:
                f.readline()
                if f.tell() == os.fstat(f.fileno()).st_size:
                    raise RuntimeError(
                        (err_msg + "Reason: Output %s is empty" % (ofile)))
        logger.debug('Done. Time needed: %s', dt.datetime.now() - t)

    def _modify_run(self, parser):
        parser.update_arg('ifile', short='i')
        parser.update_arg('ofile', short='o')
        parser.update_arg('odir', short='od')
        parser.update_arg('work_dir', short='wd')
        parser.update_arg('remove', short='r')

    # -------------------------------------------------------------------------
    # -------------------------- Postprocessing -------------------------------
    # ------------ Postprocessing functions for the experiment ----------------
    # -------------------------------------------------------------------------

    # ---------------------------- Evaluation ---------------------------------

    @docstrings.get_sectionsf('GWGENOrganizer.evaluate')
    @docstrings.dedent
    def evaluate(self, stations=None, other_exp=None,
                 setup_from=None, to_db=None, to_csv=None, database=None,
                 norun=False, to_return=None, complete=False, **kwargs):
        """
        Evaluate the experiment

        Parameters
        ----------
        %(GWGENOrganizer.param.parameters)s"""
        from gwgen.evaluation import Evaluator
        task_names = [task.name for task in Evaluator._registry]
        evaluator_kws = {
            key: vars(val) if isinstance(val, Namespace) else dict(val)
            for key, val in kwargs.items() if key in task_names}
        main_kws = {key: val for key, val in kwargs.items()
                    if key not in task_names}
        self.app_main(**main_kws)
        experiment = self.experiment
        exp_dict = self.fix_paths(self.config.experiments[experiment])
        eval_dir = exp_dict.setdefault(
            'evaldir', osp.join(exp_dict['expdir'], 'evaluation'))
        if not osp.exists(eval_dir):
            os.makedirs(eval_dir)
        projectname = self.projectname
        logger = self.logger
        logger.info("Evaluating experiment %s of project %s",
                    experiment, projectname)
        stations = self._get_stations(stations, other_exp, eval_dir,
                                      'eval_stations')
        global_conf = self.config.global_config
        # choose keywords for data processing
        manager = Evaluator.get_manager(config=global_conf)
        self._setup_manager(manager, stations, other_exp, setup_from, to_db,
                            to_csv, database, to_return, complete,
                            evaluator_kws)
        # update experiment namelist and configuration
        if not norun:
            manager.run(exp_dict.setdefault('evaluation', OrderedDict()))
        return manager

    def _modify_evaluate(self, parser, *args, **kwargs):
        from gwgen.evaluation import Evaluator
        self._modify_task_parser(parser, Evaluator, *args, **kwargs)

    @property
    def bias_correction_methods(self):
        return {'wind': self.wind_bias_correction,
                'tmin': self.tmin_bias_correction}

    @docstrings.dedent
    def bias_correction(
            self, keep=False, quantiles=list(range(1, 100)),
            no_evaluation=False, new_project=False, **kwargs):
        """
        Perform a bias correction for the data

        Parameters
        ----------
        keep: bool
            If not True, the experiment configuration files are not modified.
            Otherwise the `quants` section is kept for the given quantiles
        quantiles: list of float
            The quantiles to use for the bias correction. Does not have an
            effect if `no_evaluation` is set to True
        no_evaluation: bool
            If True, the existing evaluation in the configuration is used for
            the bias correction
        new_project: bool
            If True, a new project will be created even if a file in
            `project_output` exists already

        Returns
        -------
        dict
            The results of the underlying bias correction methods"""
        methods = self.bias_correction_methods

        main_kws = self.get_app_main_kwargs(kwargs, keep=True)
        bias_kws = {
            key: kwargs.pop(key) for key in set(methods).intersection(kwargs)}

        self.app_main(**main_kws)
        self.logger.debug('Calculating bias correction for experiment %s',
                          self.experiment)
        old = self.exp_config.get('evaluation', {}).get('quants')
        postproc_dir = self.exp_config.setdefault(
            'postprocdir', osp.join(self.exp_config['expdir'], 'postproc'))
        if not osp.exists(postproc_dir):
            os.makedirs(postproc_dir)
        quants_output = osp.join(postproc_dir, 'quants_bias')
        kwargs['quants'] = {'quantiles': quantiles, 'transform_wind': False,
                            'new_project': new_project,
                            'names': list(bias_kws),
                            'project_output': quants_output + '.pkl',
                            'plot_output': quants_output + '.pdf',
                            'nc_output': quants_output + '.nc'}
        self.evaluate(**kwargs)

        d = self.exp_config.setdefault('postproc', OrderedDict()).setdefault(
            'bias', OrderedDict())

        d['plot_file'] = quants_output + '.pdf'
        d['project_file'] = quants_output + '.pkl'
        d['nc_file'] = quants_output + '.nc'

        for name, kws in bias_kws.items():
            if isinstance(kws, Namespace):
                kws = vars(kws)
                for key in ['keep', 'quantiles', 'no_evaluation']:
                    kws.pop(key, None)
            methods[name](self.exp_config['evaluation']['quants'], **kws)

        if not keep:
            if old:
                self.exp_config['evaluation']['quants'] = old
            else:
                self.exp_config['evaluation'].pop('quants')

    def _modify_bias_correction(self, parser):
        self._modify_app_main(parser)
        parser.update_arg('keep', short='k')
        parser.update_arg(
            'quantiles', short='q', type=utils.str_ranges,
            metavar='f1[,f21[-f22[-f23]]]', help=docstrings.dedents("""
                The quantiles to use for calculating the percentiles.
                %(str_ranges.s_help)s."""))
        parser.pop_key('quantiles', 'nargs', None)
        parser.update_arg('new_project', short='np')
        parser.update_arg('no_evaluation', short='ne')

        sps = parser.add_subparsers(chain=True)

        # -- wind
        sp = sps.add_parser('wind')
        sp.setup_args(self.wind_bias_correction_logistic)
        sp.setup_args(self.wind_bias_correction)
        sp.update_arg('new_project', short='np')
        sp.update_arg('plot_output', short='po')
        sp.pop_arg('info')
        sp.pop_arg('close')

        # -- tmin
        sp = sps.add_parser('tmin')
        sp.setup_args(self.poly_bias_correction)
        sp.pop_arg('vname')
        sp.pop_arg('what')
        sp.pop_arg('ds')
        sp.setup_args(self.tmin_bias_correction)
        sp.update_arg('new_project', short='np')
        sp.update_arg('plot_output', short='po')
        sp.pop_arg('info')
        sp.pop_arg('close')

    @docstrings.get_sectionsf('GWGENOrganizer.wind_bias_correction_logistic')
    @docstrings.dedent
    def wind_bias_correction_logistic(
            self, info, new_project=False, plot_output=None, close=True):
        """
        Perform a bias correction for the data

        Parameters
        ----------
        info: dict
            The configuration of the quantile evaluation
        new_project: bool
            If True, a new project will be created even if a file in
            `project_output` exists already
        plot_output: str
            The name of the output file. If not specified, it defaults to
            `<exp_dir>/postproc/<vname>_bias_correction.pdf`
        close: bool
            If True, close the project at the end"""
        import pandas as pd
        from scipy import stats
        import xarray as xr
        import psyplot.project as psy

        vname = 'wind'

        self.logger.debug('Calculating bias correction for experiment %s',
                          self.experiment)
        postproc_dir = self.exp_config.setdefault(
            'postprocdir', osp.join(self.exp_config['expdir'], 'postproc'))

        df = pd.DataFrame.from_dict(info[vname], 'index')
        try:
            # drop all percentiles
            df.drop('All', inplace=True)
        except (ValueError, KeyError) as e:
            pass
        df.index.name = 'pctl'
        df.reset_index(inplace=True)
        df['unorm'] = stats.norm.ppf(
            df['pctl'].astype(float) / 100., 0, 1.0)
        ds = xr.Dataset.from_dataframe(df)

        # --- plots
        d = self.exp_config.setdefault('postproc', OrderedDict()).setdefault(
            'bias', OrderedDict()).setdefault(vname, OrderedDict())
        plot_output = plot_output or d.get('plot_output')
        if plot_output is None:
            plot_output = osp.join(
                postproc_dir, vname + '_bias_correction.pdf')

        project_output = osp.splitext(plot_output)[0] + '.pkl'
        nc_output = osp.splitext(plot_output)[0] + '.nc'

        d['plot_file'] = plot_output
        d['project_file'] = project_output
        d['nc_file'] = nc_output

        # --- slope bias correction
        if osp.exists(project_output) and not new_project:
            mp = psy.Project.load_project(project_output, datasets=[ds])
            sp2 = mp.linreg(name='slope')
        else:
            import seaborn as sns
            sns.set_style('white')
            sp1 = psy.plot.lineplot(ds, name='slope', coord='unorm',
                                    linewidth=0, marker='o', legend=False)
            sp2 = psy.plot.linreg(
                 ds, name='slope', ax=sp1[0].psy.ax,
                 coord='unorm', fit=logistic_function,
                 ylabel=('$\\mathrm{{Simulated}}\\, \\mathrm{{%s}} / '
                         '\\mathrm{{Observed}}\\, \\mathrm{{%s}}$') % (
                            vname, vname),
                 legendlabels=(
                     '$\\frac{{\\mathrm{{Simulated}}}}'
                     '{{\\mathrm{{Observed}}}} = '
                     '\\frac{{%(L)4.3f}}{{1 + \\mathrm{{e}}^{{'
                     '%(k)4.3f\\cdot(x %(x0)+4.3f)}}}}$'),
                 legend={'fontsize': 'x-large', 'loc': 'upper left'},
                 xlabel='Random number $x$ from normal distribution')
            sp2.share(sp1[0], ['color', 'xlim', 'ylim'])
        arr = sp2.plotters[0].plot_data[0]
        nml = self.exp_config['namelist']['weathergen_ctl']
        if 'L' in arr.attrs:
            nml.pop(vname + '_bias_coeffs', None)
            for letter in ['L', 'k', 'x0']:
                nml[vname + '_slope_bias_' + letter] = float(arr.attrs[letter])
        else:  # polynomial fit
            for letter in ['L', 'k', 'x0']:
                nml.pop(vname + '_slope_bias_' + letter, None)
            nml[vname + '_bias_coeffs'] = [
                    float(arr.attrs.get('c%i' % i, 0.0)) for i in range(6)]

        # --- intercept bias correction
        if osp.exists(project_output) and not new_project:
            sp2 = mp.linreg(name='intercept')
        else:
            sp1 = psy.plot.lineplot(ds, name='intercept', coord='unorm',
                                    linewidth=0, marker='o', legend=False)
            sp2 = psy.plot.linreg(
                 ds, name='intercept', ax=sp1[0].psy.ax,
                 coord='unorm', fit=exponential_function,
                 ylabel=(
                    '$\\mathrm{{Simulated}}\\, \\mathrm{{%s}} - '
                    '\\mathrm{{Observed}}\\, \\mathrm{{%s}}$ [m/s]') % (
                        vname, vname),
                 legendlabels=(
                     '$\\mathrm{{Simulated}} - \\mathrm{{Observed}} ='
                     'e^{{%(a)1.4f \\cdot x %(b)+1.4f}}$'),
                 legend={'fontsize': 'medium', 'loc': 'upper left'},
                 xlabel='Random number $x$ from normal distribution')
        arr = sp2.plotters[0].plot_data[0]
        if 'a' in arr.attrs:
            nml.pop(vname + '_intercept_bias_coeffs', None)
            for letter in ['a', 'b']:
                nml[vname + '_intercept_bias_' + letter] = float(
                    arr.attrs[letter])
        else:  # polynomial fit
            for letter in ['a', 'b']:
                nml.pop(vname + '_intercept_bias_' + letter, None)
            nml[vname + '_intercept_bias_coeffs'] = [
                float(arr.attrs.get('c%i' % i, 0.0)) for i in range(6)]
        nml[vname + '_bias_min'] = float(ds.unorm.min().values)
        nml[vname + '_bias_max'] = float(ds.unorm.max().values)
        # --- save the data
        self.logger.info('Saving plots to %s', plot_output)
        mp = psy.gcp(True)
        mp.export(plot_output)
        self.logger.info('Saving project to %s', project_output)
        mp.save_project(project_output, paths=[nc_output])

        if close:
            psy.gcp(True).close(True, True, True)

    @docstrings.get_sectionsf('GWGENOrganizer.poly_bias_correction')
    @docstrings.dedent
    def poly_bias_correction(
            self, vname, what, info, new_project=False, plot_output=None,
            deg=3, close=True, ds=None):
        """
        Perform a bias correction based on percentile and a polynomial fit

        Parameters
        ----------
        vname: str
            The variable name to use
        what: str { 'slope' | 'intercept' }
            Either slope or intercept. The parameter that should be used for
            the bias correction
        info: dict
            The configuration of the quantile evaluation
        new_project: bool
            If True, a new project will be created even if a file in
            `project_output` exists already
        plot_output: str
            The name of the output file. If not specified, it defaults to
            `<exp_dir>/postproc/<vname>_bias_correction.pdf`
        deg: int
            The degree of the fittet polynomial
        close: bool
            If True, close the project at the end
        ds: xr.Dataset
            The xarray dataset to use. Otherwise it will be created from `info`
        """
        import pandas as pd
        from scipy import stats
        import xarray as xr
        import psyplot.project as psy

        def get_symbol(i):
            if not i:
                return ''
            elif i == 1:
                return 'x'
            else:
                return 'x^' + str(i)

        self.logger.debug('Calculating %s bias correction for experiment %s',
                          vname, self.experiment)
        postproc_dir = self.exp_config.setdefault(
            'postprocdir', osp.join(self.exp_config['expdir'], 'postproc'))
        if ds is None:
            df = pd.DataFrame(info[vname]).T
            try:
                # drop all percentiles
                df.drop('All', inplace=True)
            except (ValueError, KeyError) as e:
                pass
            df.index.name = 'pctl'
            df.reset_index(inplace=True)
            df['unorm'] = stats.norm.ppf(
                df['pctl'].astype(float) / 100., 0, 1.0)
            ds = xr.Dataset.from_dataframe(df)

        # --- plots
        d = self.exp_config.setdefault('postproc', OrderedDict()).setdefault(
            'bias', OrderedDict()).setdefault(vname, OrderedDict())
        plot_output = plot_output or d.get('plot_output')
        if plot_output is None:
            plot_output = osp.join(
                postproc_dir, vname + '_bias_correction.pdf')

        project_output = osp.splitext(plot_output)[0] + '.pkl'
        nc_output = osp.splitext(plot_output)[0] + '.nc'

        d['plot_file'] = plot_output
        d['project_file'] = project_output
        d['nc_file'] = nc_output

        if what == 'slope':
            ylabel = 'Simulated/Observed'
            if vname == 'wind':
                ylabel = '$\\sqrt{{' + ylabel + '}}$'
        else:
            ylabel = 'Simulated - Observed'
        diff_symbol = ylabel
        if vname == 'tmin':
            ylabel += ' [$^\circ$C]'

        # --- slope bias correction
        if osp.exists(project_output) and not new_project:
            mp = psy.Project.load_project(project_output, datasets=[ds])
            sp2 = mp.linreg
        else:
            import seaborn as sns
            sns.set_style('white')
            sp1 = psy.plot.lineplot(ds, name=what, coord='unorm',
                                    linewidth=0, marker='o', legend=False)
            label = '$%s = %s$' % (diff_symbol, ' '.join(
                '%(c{})+4.3f{}'.format(i, get_symbol(i))
                for i in range(deg + 1)))
            sp2 = psy.plot.linreg(
                 ds, name=what, ax=sp1[0].ax,
                 coord='unorm', fit='poly' + str(int(deg)),
                 ylabel=ylabel,
                 legendlabels=label,
                 legend={'fontsize': 'large', 'loc': 'upper left'},
                 xlabel='Random number from normal distribution')
            sp2.share(sp1[0], ['color', 'xlim', 'ylim'])
        attrs = sp2.plotters[0].plot_data[0].attrs
        nml = self.exp_config['namelist']['weathergen_ctl']
        nml[vname + '_bias_coeffs'] = [
            float(attrs.get('c%i' % i, 0.0)) for i in range(6)]
        nml[vname + '_bias_min'] = float(ds.unorm.min().values)
        nml[vname + '_bias_max'] = float(ds.unorm.max().values)

        # --- save the data
        self.logger.info('Saving plots to %s', plot_output)
        mp = psy.gcp(True)
        mp.export(plot_output)
        self.logger.info('Saving project to %s', project_output)
        mp.save_project(project_output, paths=[nc_output])

        if close:
            psy.gcp(True).close(True, True, True)

    docstrings.delete_params('GWGENOrganizer.poly_bias_correction.parameters',
                             'vname', 'what')

    @docstrings.dedent
    def tmin_bias_correction(self, *args, **kwargs):
        """
        Perform a bias correction for the minimum temperature data

        Parameters
        ----------
        %(GWGENOrganizer.poly_bias_correction.parameters.no_vname|what)s"""
        return self.poly_bias_correction('tmin', 'intercept', *args, **kwargs)

    @docstrings.dedent
    def wind_bias_correction(self, *args, **kwargs):
        """
        Perform a bias correction for the wind speed

        Parameters
        ----------
        %(GWGENOrganizer.wind_bias_correction_logistic.parameters)s"""
        return self.wind_bias_correction_logistic(*args, **kwargs)

    # ----------------------- Sensitivity analysis ----------------------------

    @docstrings.dedent
    def sensitivity_analysis(self, **kwargs):
        """
        Perform a sensitivity analysis on the given parameters

        This function performs a sensitivity analysis on the current
        experiment. It creates a new project and uses the evaluation and
        parameterization of the current experiment to get information on the
        others
        """
        from gwgen.sensitivity_analysis import SensitivityAnalysis
        sa_func_map = OrderedDict([
            ('setup', 'setup'), ('compile', 'compile_model'),
            ('init', 'init'), ('run', 'run'), ('evaluate', 'evaluate'),
            ('plot', 'plot'), ('remove', 'remove')])
        sensitivity_kws = OrderedDict(
            (key, kwargs[key]) for key in sa_func_map if key in kwargs)
        main_kws = {
            key: kwargs[key] for key in set(kwargs).difference(sa_func_map)}
        self.app_main(**main_kws)
        # to make sure, we already called the choose the right experiment and
        # projectname
        experiment = self.experiment
        self.logger.debug('Running sensitivity analysis for %s', experiment)
        sa = SensitivityAnalysis(self)
        self.fix_paths(self.exp_config)
        self.fix_paths(self.project_config)
        for key, val in sensitivity_kws.items():
            if isinstance(val, Namespace):
                val = vars(val)
            getattr(sa, sa_func_map[key])(**val)

    def _modify_sensitivity_analysis(self, parser):
        from gwgen.sensitivity_analysis import (
            SensitivityAnalysis, SensitivityPlot, default_sens_config)

        def params_type(s):
            splitted = s.split('=', 1)
            key = splitted[0]
            return key, utils.str_ranges(splitted[1])

        sps = parser.add_subparsers(help='Sensitivity analysis subroutines',
                                    chain=True)

        # setup parser
        sp = sps.add_parser('setup',
                            help='Setup the sensitivity analysis model')
        sp.setup_args(SensitivityAnalysis.setup)
        self._modify_app_main(sp)
        sp.update_arg('no_move', short='nm')

        # compile parser
        sp = sps.add_parser('compile',
                            help='Compile the sensitivity analysis model')
        sp.setup_args(SensitivityAnalysis.compile_model)
        self._modify_compile_model(sp)

        # init parser
        sp = sps.add_parser(
            'init', help='Initialize the sensitivity analysis experiments')
        sp.setup_args(SensitivityAnalysis.init)
        sp.update_arg('experiment', short='id')
        sp.update_arg(
            'nml', long='namelist', type=params_type,
            help=docstrings.dedents("""
                A list from namelist parameters and their values to use.
                %(str_ranges.s_help)s.
                You can also use ``'<i>err'`` in the list which will be
                interpreted as ``'<i>'``-times the error from the
                parameterization.
                """),
            metavar='nml_key=f1[,f21[-f22[-f23]]]', nargs='+')
        sp.update_arg('run_prepare', short='prep')
        sp.update_arg('no_move', short='nm')

        # run parser
        sp = sps.add_parser(
            'run', help='Run the sensitivity analysis experiments')
        sp.setup_args(SensitivityAnalysis.run)
        sp.update_arg('remove', short='rm')
        sp.update_arg('experiments', short='ids', type=lambda s: s.split(','),
                      metavar='id1,id2,...')
        sp.pop_key('experiments', 'nargs', None)

        # evaluate parser
        sp = sps.add_parser(
            'evaluate', help='Evaluate the sensitivity analysis experiments')
        sp.setup_args(SensitivityAnalysis.evaluate)
        sp.setup_args(self.evaluate)
        sp.update_arg('experiments', short='ids', type=lambda s: s.split(','),
                      metavar='id1,id2,...')
        sp.pop_key('experiments', 'nargs', None)
        sp.update_arg('loop_exps', short='loop')
        self._modify_evaluate(sp, skip=['prepare', 'output'])

        # plot parser
        sp = sps.add_parser(
            'plot', help='Plot the results sensitivity analysis experiments')
        sp.setup_args(SensitivityAnalysis.plot)
        defaults = default_sens_config()
        sp.update_arg('names', short='n', type=lambda s: s.split(','),
                      metavar='variable,[variable[,...]]',
                      default=defaults.names)
        sp.update_arg('indicators', short='i', type=lambda s: s.split(','),
                      metavar='indicator[,indicator[,...]]',
                      default=defaults.indicators)
        sp.update_arg('meta', metavar='<yaml-file>')
        tasks = utils.unique_everseen(
            SensitivityPlot.get_manager().sort_by_requirement(
                SensitivityPlot._registry[::-1]), lambda t: t.name)
        plot_sps = sp.add_subparsers(help='Plotting tasks', chain=True)
        for task in tasks:
            plot_sp = plot_sps.add_parser(task.name, help=task.summary)
            task._modify_parser(plot_sp)

        # remove parser
        sp = sps.add_parser('remove', help="Remove the sensitivity project")
        sp.setup_args(SensitivityAnalysis.remove)
        sp.setup_args(self.remove)
        self._modify_remove(sp)
        sp.pop_arg('projectname')
        sp.pop_arg('complete')

    # -------------------------------------------------------------------------
    # ------------------------------ Miscallaneous ----------------------------
    # -------------------------------------------------------------------------

    def _get_stations(self, stations, other_exp=False, odir=None,
                      config_key=None):
        """
        Get the stations for the parameterization or evaluation

        Parameters
        ----------
        stations: str or list of str
            either a list of stations to use or a filename containing a
            1-row table with stations
        other_exp: str
            Use the configuration from another experiment
        odir: str
            The output directory in case a list of stations is provided
        config_key:
            The key in the :attr:`exp_config` configuration dictionary holding
            information on the stations
        """
        import numpy as np

        exp_dict = self.exp_config
        fname = osp.join(odir, 'stations.dat') if odir else ''
        if other_exp and stations is None:
            stations = self.fix_paths(
                self.config.experiments[other_exp]).get(config_key)
        if isinstance(stations, six.string_types):
            stations = [stations]
        if stations is None:
            try:
                fname = exp_dict[config_key]
            except KeyError:
                raise ValueError('No stations file specified!')
            else:
                stations = np.loadtxt(exp_dict[config_key],
                                      dtype='S300', usecols=[0]).astype(
                    np.str_)
        elif len(stations) == 1 and osp.exists(stations[0]):
            fname_use = stations[0]
            exists = osp.exists(fname) if fname else False
            if exists and not osp.samefile(fname, fname_use):
                os.remove(fname)
                self._link(fname_use, fname)
            elif not exists and fname:
                self._link(fname_use, fname)
            stations = np.loadtxt(
                fname_use, dtype='S300', usecols=[0]).astype(np.str_)
        elif len(stations) and fname:
            np.savetxt(fname, stations, fmt='%s')
        if config_key and (not exp_dict.get(config_key) or not osp.samefile(
                fname, exp_dict[config_key])):
            exp_dict[config_key] = fname
        return stations

    def _setup_manager(
            self, manager, stations=None, other_exp=None,
            setup_from=None, to_db=None, to_csv=None, database=None,
            to_return=None, complete=False, base_kws={}):
        """
        Setup the data in a task manager

        This method is called by :meth:`param` and :meth:`evaluate` to setup
        the data in the given `manager`

        Parameters
        ----------
        manager: gwgen.utils.TaskManager
            The manager of the tasks to set up
        stations: list of str
            a list of stations to use
        other_exp: str
            Use the configuration from another experiment instead of
        setup_from: str
            Determine where to get the data from. If `scratch`, the
            data will be calculated from the raw data. If `file`,
            the data will be loaded from a file, if `db`, the data
            will be loaded from a postgres database (Note that the
            `database` argument must be provided!).
        to_db: bool
            Save the data into a postgresql database (Note that the
            `database` argument must be provided!)
        to_csv: bool
            Save the data into a csv file
        database: str
            The name of a postgres data base to write the data to
        to_return: list of str
            The names of the tasks to return. If None, only the ones with an
            :attr:`gwgen.utils.TaskBase.has_run` are returned.
        complete: bool
            If True, setup and run all possible tasks
        base_kws: dict
            The dictionary with mapping from each task name to the
            corresponding initialization keywords
        """
        if complete:
            for task in manager.base_task._registry:
                base_kws.setdefault(task.name, {})
        experiment = self.experiment
        exp_dict = self.fix_paths(self.config.experiments[experiment])
        if database is not None:
            exp_dict['database'] = database
        # setup up the keyword arguments for the parameterization tasks
        for key, d in base_kws.items():
            if d.get('setup_from') is None:
                d['setup_from'] = setup_from
            if to_csv:
                d['to_csv'] = to_csv
            elif to_csv is None and d.get('to_csv') is None:
                # delete the argument if the subparser doesn't use it
                d.pop('to_csv', None)
            if to_db:
                # delete the argument if the subparser doesn't use it
                d['to_db'] = to_db
            elif to_db is None and d.get('to_db') is None:
                d.pop('to_db', None)
            if other_exp and not d.get('other_exp'):
                d['other_exp'] = other_exp
            exp = d.pop('other_exp', experiment) or experiment
            d['config'] = self.fix_paths(self.config.experiments[exp])
            d['project_config'] = self.config.projects[d['config']['project']]
            for key in ['stations', 'complete', 'norun', 'other_id',
                        'database']:
                d.pop(key, None)
        # choose keywords for data processing
        manager.initialize_tasks(stations, task_kws=base_kws)
        manager.setup(stations, to_return=to_return)

    def _modify_task_parser(self, parser, base_task, skip=None, only=None):
        def norun(s):
            if s is True or s == 'all':
                return True
            try:
                return bool(int(s))
            except TypeError:
                return s.split(',')
        skip = skip or []
        if only is None:
            def key_func(t):
                return t.name not in skip
        else:
            def key_func(t):
                return t.name in only and t.name not in skip
        self._modify_app_main(parser)
        parser.update_arg('setup_from', short='f', long='from',
                          dest='setup_from')
        parser.update_arg('other_exp', short='ido', long='other_id',
                          dest='other_exp')
        try:
            parser.update_arg('stations', short='s')
        except KeyError:
            pass
        parser.update_arg('database', short='db')
        parser.pop_arg('to_return', None)
        parser.update_arg(
            'norun', short='nr', const=True, nargs='?',
            type=norun, help=(
                'If set without value or "all" or a number different from 0, '
                'the data is set up and the configuration of the '
                'experiment is not affected. Otherwise it can be a comma '
                'separated list of parameterization tasks for which to only '
                'setup the data'), metavar='task1,task2,...')
        doc = docstrings.params['GWGENOrganizer.param.parameters']
        setup_from_doc, setup_from_dtype = parser.get_param_doc(
            doc, 'setup_from')
        other_exp_doc, other_exp_dtype = parser.get_param_doc(doc, 'other_exp')

        tasks = filter(key_func, utils.unique_everseen(
            base_task.get_manager().sort_by_requirement(
                base_task._registry[::-1]), lambda t: t.name))
        sps = parser.add_subparsers(title='Tasks', chain=True)
        for task in tasks:
            sp = sps.add_parser(task.name, help=task.summary,
                                formatter_class=RawTextHelpFormatter)
            task._modify_parser(sp)
            sp.add_argument(
                '-ido', '--other_id', help=other_exp_doc,
                metavar=other_exp_dtype)

    def _link(self, source, target):
        """Link two files

        Parameters
        ----------
        source: str
            The path of the source file
        target: str
            The path of the target file"""
        if self.global_config.get('copy', True) and osp.isfile(source):
            shutil.copyfile(source, target)
        elif self.global_config.get('use_relative_links', True):
            os.symlink(osp.relpath(source, osp.dirname(target)), target)
        else:
            os.symlink(osp.abspath(source), target)


def exponential_function(x, a, b):
    """
    Exponential function used by :meth:`GWGENOrganizer.wind_bias_correction`

    This function is defined as

    .. math::

        f(x) = e^{ax + b}

    Parameters
    ----------
    x: numpy.ndarray
        The x-data
    a: float
        The *a* parameter in the above equation
    b: float
        The *b* parameter in the above equation

    Returns
    -------
    np.ndarray
        The calculated :math:`f(x)`
    """
    return np.exp(a * x + b)


def logistic_function(x, L, k, x0):
    """Logistic function used in :meth:`GWGENOrganizer.wind_bias_correction`

    The function is defined as

    .. math::

        f(x) = \\frac{L}{1 + \\mathrm e^{-k(x-x_0)}}

    Parameters
    ----------
    x: numpy.ndarray
        The x-data
    L: float
        the curve's maximum value
    k: float
        The steepness of the curve
    x0: the x-value of the sigmoid's midpoint

    Returns
    -------
    np.ndarray
        The calculated :math:`f(x)`"""
    return L / (1 + np.exp(-k * (x - x0)))


def _get_parser():
    """Function returning the gwgen parser, necessary for sphinx documentation
    """
    return GWGENOrganizer.get_parser()


def main(args=None):
    """Call the :meth:`~model_organization.GWGENOrganizer.main` method of the
    :class:`GWGENOrganizer` class"""
    GWGENOrganizer.main(args)


if __name__ == '__main__':
    main()
