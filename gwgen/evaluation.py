# -*- coding: utf-8 -*-
"""Evaluation module of the gwgen module"""
from __future__ import division
import os.path as osp
import six
from collections import namedtuple
from psyplot.compat.pycompat import OrderedDict
from itertools import chain, starmap, repeat
import numpy as np
from scipy import stats
import pandas as pd
from gwgen.utils import docstrings
import gwgen.utils as utils
import logging


class Evaluator(utils.TaskBase):
    """Abstract base class for evaluation tasks

    Evaluation tasks should incorporate a run method that is called by the
    :meth:`gwgen.main.GWGENOrganizer.evaluate` method"""

    _registry = []

    @property
    def task_data_dir(self):
        """The directory where to store data"""
        return self.eval_dir

_PreparationConfigBase = namedtuple('_PreparationConfigBase',
                                    ['setup_raw', 'raw2db', 'raw2csv',
                                     'reference', 'input_path'])

_PreparationConfigBase = utils.append_doc(
    _PreparationConfigBase, docstrings.get_sections("""
Parameters
----------
setup_raw: { 'scratch' | 'file' | 'db' | None }
    The method how to setup the raw data from GHCN and EECRA

    ``'scratch'``
        To set up the task from the raw data
    ``'file'``
        Set up the task from an existing file
    ``'db'``
        Set up the task from a database
    ``None``
        If the file name of this this task exists, use this one,
        otherwise a database is provided, use this one, otherwise go
        from scratch
raw2db: bool
    If True, the raw data from GHCN and EECRA is stored in a postgres database
raw2csv: bool
    If True, the raw data from GHCN and EECRA is stored in a csv file
reference: str
    The path of the file where to store the reference data. If None and not
    already set in the configuration, it will default to
    ``'evaluation/reference.csv'``
input_path: str
    The path of the file where to store the model input. If None, and not
    already set in the configuration, it will default to
    ``'inputdir/input.csv'`` where *inputdir* is the path to the input
    directory (by default, *input* in the experiment directory)
""", '_PreparationConfigBase'))


PreparationConfig = utils.enhanced_config(_PreparationConfigBase,
                                          'PreparationConfig')


@docstrings.dedent
def default_preparation_config(
        setup_raw=None, raw2db=False, raw2csv=False, reference=None,
        input_path=None, *args, **kwargs):
    """
    The default configuration for :class:`EvaluationPreparation` instances.
    See also the :attr:`EvaluationPreparation.default_config` attribute

    Parameters
    ----------
    %(PreparationConfig.parameters)s"""
    return PreparationConfig(setup_raw, raw2db, raw2csv, reference, input_path,
                             *utils.default_config(*args, **kwargs))


class EvaluationPreparation(Evaluator):
    """Evaluation task to prepare the evaluation"""

    name = 'prepare'

    summary = 'Prepare the for experiment for evaluation'

    http_inventory = (
        'ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt')

    _datafile = ['reference.csv', 'input.csv']

    dbname = ['reference', 'input']

    has_run = False

    @property
    def default_config(self):
        ret = default_preparation_config()._replace(
            **super(EvaluationPreparation, self).default_config._asdict())
        return ret._replace(**dict(reference=self.reference_path,
                                   input_path=self.input_path))

    @property
    def datafile(self):
        """The paths to reference and input file"""
        return [self.reference_path, self.input_path]

    @property
    def ghcnd_inventory_file(self):
        return osp.join(self.data_dir, 'ghcn', 'ghcnd-inventory.txt')

    @property
    def station_list(self):
        fname = self.ghcnd_inventory_file
        station_ids, lat, lon, variables, first, last = np.loadtxt(
            fname, dtype='S11', unpack=True).astype(np.str_)
        return pd.DataFrame({'id': station_ids, 'lat': lat.astype(float),
                             'lon': lon.astype(float), 'vname': variables,
                             'firstyr': first.astype(int),
                             'lastyr': last.astype(int)},
                            index=np.arange(len(station_ids)))

    @property
    def reference_data(self):
        """The reference :class:`~pandas.DataFrame`"""
        return self.data[0]

    @reference_data.setter
    def reference_data(self, data):
        self.data[0] = data

    @property
    def input_data(self):
        """The input :class:`~pandas.DataFrame`"""
        return self.data[1]

    @input_data.setter
    def input_data(self, data):
        self.data[1] = data

    @classmethod
    def _modify_parser(cls, parser):
        parser.setup_args(default_preparation_config)
        # disable plot_output, etc.
        parser, setup_grp, run_grp = super(
            EvaluationPreparation, cls)._modify_parser(parser)
        parser.update_arg('setup_raw', short='fr', long='raw-from',
                          group=setup_grp)
        parser.update_arg('input_path', short='i', long='input',
                          group=setup_grp)
        parser.update_arg('reference', short='r', group=setup_grp)
        parser.update_arg('raw2db', group=setup_grp)
        parser.update_arg('raw2csv', group=setup_grp)
        return parser, setup_grp, run_grp

    def write2file(self, *args, **kwargs):
        """Reimplemented to sort the data according to the index"""
        for data in self.data:
            data.sort_index(inplace=True)
        return super(EvaluationPreparation, self).write2file(*args, **kwargs)

    def write2db(self, *args, **kwargs):
        """Reimplemented to sort the data according to the index"""
        for data in self.data:
            data.sort_index(inplace=True)
        return super(EvaluationPreparation, self).write2db(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = [['id', 'year', 'month', 'day'],  # reference
                               ['id', 'lon', 'lat', 'year', 'month']]  # input
        super(EvaluationPreparation, self).setup_from_db(*args, **kwargs)

    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = [['id', 'year', 'month', 'day'],  # reference
                               ['id', 'lon', 'lat', 'year', 'month']]  # input
        super(EvaluationPreparation, self).setup_from_file(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        self._manager = kwargs.pop('manager', None)
        super(EvaluationPreparation, self).__init__(*args, **kwargs)
        if self.task_config.to_csv:  # update the configuration
            self.reference_path = self.reference_path
            self.input_path = self.input_path

    def __reduce__(self):
        """Reimplemented to give provide also the manager"""
        ret = list(super(EvaluationPreparation, self).__reduce__())
        if len(ret) < 3:
            ret.append({})
        ret[2]['_manager'] = self._manager
        return tuple(ret)

    def init_from_scratch(self):
        """Initialize the setup via the parameterization classes"""
        from gwgen.parameterization import (
            Parameterizer, YearlyCompleteDailyCloud,
            YearlyCompleteDailyGHCNData, YearlyCompleteMonthlyCloud,
            YearlyCompleteMonthlyGHCNData)
        classes = [
            YearlyCompleteDailyCloud, YearlyCompleteDailyGHCNData,
            YearlyCompleteMonthlyCloud, YearlyCompleteMonthlyGHCNData]
        config = self.config.copy()
        config['paramdir'] = self.eval_dir
        kws = dict(config=config, project_config=self.project_config,
                   to_csv=self.task_config.raw2csv,
                   to_db=self.task_config.raw2db,
                   setup_from=self.task_config.setup_raw)
        base_kws = {cls.name: kws for cls in classes}
        self._manager = Parameterizer.get_manager(
            config=self.global_config.copy())
        self._manager.initialize_tasks(self.stations, task_kws=base_kws)
        if not self.global_config.get('serial'):
            import multiprocessing as mp
            for task in self._manager.tasks:
                utils._db_locks[task.dbname] = mp.Lock()
                utils._file_locks[task.datafile] = mp.Lock()
        # make sure the lat-lon information is available
        self.download_src()

    def download_src(self, force=False):
        fname = self.ghcnd_inventory_file
        if force or not osp.exists(fname):
            self.logger.info("Downloading station file from %s to %s",
                             self.http_inventory, fname)
            utils.download_file(self.http_inventory, fname)

    def setup_from_scratch(self):
        from gwgen.parameterization import (
            YearlyCompleteDailyCloud, YearlyCompleteDailyGHCNData,
            YearlyCompleteMonthlyCloud, YearlyCompleteMonthlyGHCNData)

        def get(name):
            return next(t for t in tasks if t.name == name)

        # force serial setup because we might already be in a parallel setup
        self._manager.config['serial'] = True
        self._manager.setup(self.stations, to_return='all')
        tasks = self._manager.tasks
        # save in the right order for the FORTRAN code
        order = ['tmin', 'tmax', 'mean_cloud', 'wind', 'prcp', 'wet_day']

        # daily reference
        cday = get(YearlyCompleteDailyGHCNData.name).data
        ccday = get(YearlyCompleteDailyCloud.name).data
        try:
            reference = cday.merge(
                ccday[['mean_cloud', 'wind']], left_index=True,
                right_index=True, how='left')
        except TypeError:  # indices to not match
            reference = cday.ix[1:0]  # create empty data frame
            reference['mean_cloud'] = np.array([],
                                               dtype=ccday.mean_cloud.dtype)
            reference['wind'] = np.array([],
                                         dtype=ccday.wind.dtype)
        reference['wet_day'] = (reference.prcp > 0).astype(int)
        self.reference_data = reference[order].sort_index()

        # monthly input
        cmonth = get(YearlyCompleteMonthlyGHCNData.name).data
        ccmonth = get(YearlyCompleteMonthlyCloud.name).data
        try:
            exp_input = cmonth.merge(
                ccmonth[['mean_cloud', 'wind']], left_index=True,
                right_index=True, how='left')
            # set cloud and wind to 0 where we have no reference
            exp_input.ix[exp_input.mean_cloud.isnull(),
                         ['mean_cloud', 'wind']] = 0
        except TypeError:  # indices do not match
            exp_input = cmonth.ix[1:0]  # create empty data frame
            exp_input['mean_cloud'] = np.array([],
                                               dtype=ccmonth.mean_cloud.dtype)
            exp_input['wind'] = np.array([],
                                         dtype=ccmonth.wind.dtype)
        inventory = self.station_list
        exp_input = exp_input.reset_index().merge(
            inventory[inventory.vname == 'PRCP'][['id', 'lon', 'lat']],
            on='id')
        exp_input.set_index(['id', 'lon', 'lat', 'year', 'month'],
                            inplace=True)
        self.input_data = exp_input[order].sort_index()


class OutputTask(Evaluator):
    """Task to provide all the data for input and output"""

    # the last will be ignored!
    _datafile = 'output.csv'

    dbname = 'output'

    name = 'output'

    summary = 'Load the output of the model'

    has_run = False

    @property
    def datafile(self):
        return [self.output_path]

    def write2file(self, *args, **kwargs):
        """Not implemented since the output file is generated by the model!"""
        self.logger.warn(
            "Writing to file of %s task is disabled because this is done "
            "by the model!", self.name)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month', 'day']
        super(OutputTask, self).setup_from_db(*args, **kwargs)

    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month', 'day']
        super(OutputTask, self).setup_from_file(*args, **kwargs)

    def setup_from_scratch(self):
        raise ValueError(
            "Cannot set up the %s task from scratch because it requires the "
            "output of the model!")


_KSConfig = namedtuple('_KSConfig', ['no_rounding', 'names', 'transform_wind'])


_KSConfig = utils.append_doc(_KSConfig, docstrings.get_sections("""
Parameters
----------
no_rounding: bool
    Do not round the simulation to the infered precision of the
    reference. The inferred precision is the minimum difference between
    two values with in the entire data
names: list of str
    The list of variables use for calculation. If None, all variables will be
    used
transform_wind: bool
    If True, the square root of the wind is evaluated (as this is also
    simulated in the weather generator)""", '_KSConfig'))


KSConfig = utils.enhanced_config(_KSConfig, 'KSConfig')


@docstrings.dedent
def default_ks_config(
        no_rounding=False, names=None, transform_wind=False, *args, **kwargs):
    """
    The default configuration for :class:`KSEvaluation` instances.
    See also the :attr:`KSEvaluation.default_config` attribute

    Parameters
    ----------
    %(KSConfig.parameters)s"""
    return KSConfig(no_rounding, names, transform_wind,
                    *utils.default_config(*args, **kwargs))


_QuantileConfig = namedtuple(
    '_QuantileConfig',
    ['quantiles'] + list(_KSConfig._fields))


_QuantileConfig = utils.append_doc(_QuantileConfig, docstrings.get_sections(
    docstrings.dedents("""
Parameters
----------
%(_KSConfig.parameters)s
quantiles: list of floats
    The quantiles to use for calculating the percentiles
"""), '_QuantileConfig'))


QuantileConfig = utils.enhanced_config(_QuantileConfig, 'QuantileConfig')


@docstrings.dedent
def default_quantile_config(
        quantiles=[1, 5, 10, 25, 50, 75, 90, 95, 99, 100], *args, **kwargs):
    """
    The default configuration for :class:`QuantileEvaluation` instances.
    See also the :attr:`QuantileEvaluation.default_config` attribute

    Parameters
    ----------
    %(QuantileConfig.parameters)s"""
    return QuantileConfig(quantiles, *default_ks_config(*args, **kwargs))


class QuantileEvaluation(Evaluator):
    """Evaluator to evaluate specific quantiles"""

    name = 'quants'

    summary = 'Compare the quantiles of simulation and observation'

    names = OrderedDict([
        ('prcp', {'long_name': 'Precipitation',
                  'units': 'mm'}),
        ('tmin', {'long_name': 'Min. Temperature',
                  'units': 'degC'}),
        ('tmax', {'long_name': 'Max. Temperature',
                  'units': 'degC'}),
        ('mean_cloud', {'long_name': 'Cloud fraction',
                        'units': '-'}),
        ('wind', {'long_name': 'Wind Speed',
                  'units': 'm/s'})
        ])

    @property
    def all_variables(self):
        return [[v + '_ref', v + '_sim'] for v in self.names]

    setup_requires = ['prepare', 'output']

    has_run = True

    _datafile = 'quantile_evaluation.csv'

    dbname = 'quantile_evaluation'

    #: default formatoptions for the
    #: :class:`psyplot.plotter.linreg.DensityRegPlotter` plotter
    fmt = kwargs = dict(
        legend={'loc': 'upper left'},
        cmap='w_Reds',
        title='%(pctl)sth percentile',
        xlabel='%(type)s {desc}',
        ylabel='%(type)s {desc}',
        xrange=(['minmax', 1], ['minmax', 99]),
        yrange=(['minmax', 1], ['minmax', 99]),
        legendlabels=['$R^2$ = %(rsquared)s'],
        bounds=['minmax', 11, 0, 99],
        cbar='',
        bins=10,
        ideal=[0, 1],
        id_color='r',
        sym_lims='max',
        )

    @property
    def default_config(self):
        return default_quantile_config()._replace(
            **super(QuantileEvaluation, self).default_config._asdict())

    @property
    def ds(self):
        """The dataset of the quantiles"""
        import xarray as xr
        data = self.data.reset_index()
        ds = xr.Dataset.from_dataframe(
            data.set_index('pctl', append=True).swaplevel())
        full = xr.Dataset.from_dataframe(data).drop(list(chain(
            self.data.index.names, ['pctl'])))
        idx_name = full[next(v for v in self.names) + '_sim'].dims[-1]
        full.rename({idx_name: 'full_index'}, inplace=True)
        for vref, vsim in self.all_variables:
            full.rename({vref: 'all_' + vref, vsim: 'all_' + vsim},
                        inplace=True)
        ds.merge(full, inplace=True)
        for orig, attrs, (vref, vsim) in zip(
                self.names, self.names.values(), self.all_variables):
            for prefix in ['', 'all_']:
                ds[prefix + vsim].attrs.update(attrs)
                ds[prefix + vref].attrs.update(attrs)
                ds[prefix + vsim].attrs['standard_name'] = orig
                ds[prefix + vref].attrs['standard_name'] = orig
                ds[prefix + vref].attrs['type'] = 'observed'
                ds[prefix + vsim].attrs['type'] = 'simulated'
            ds['all_' + vsim].attrs['pctl'] = 'All'
            ds['all_' + vsim].attrs['pctl'] = 'All'
        ds.pctl.attrs['long_name'] = 'Percentile'
        return ds

    def __init__(self, *args, **kwargs):
        super(QuantileEvaluation, self).__init__(*args, **kwargs)
        names = self.task_config.names
        if names is not None:
            self.names = OrderedDict(
                t for t in self.names.items() if t[0] in names)

    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year']
        return super(QuantileEvaluation, self).setup_from_file(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year']
        return super(QuantileEvaluation, self).setup_from_db(*args, **kwargs)

    def setup_from_scratch(self):
        df_ref = self.prepare.reference_data
        dfi = self.prepare.input_data.reset_index(['lon', 'lat'])
        # create simulation dataframe
        df_sim = self.output.data
        if len(df_ref) == 0 or len(df_sim) == 0:
            self.logger.debug(
                'Skipping %s because reference data contains no information!',
                self.name)
            return
        names = self.names
        # load observed precision
        if self.task_config.no_rounding:
            for name in names:
                df_sim[name].values[:] = self.round_to_ref_prec(
                    df_ref[name].values, df_sim[name].values)
        # merge reference and simulation into one single dataframe
        df = df_ref.merge(df_sim, left_index=True, right_index=True,
                          suffixes=['_ref', '_sim'])
        if {'mean_cloud', 'wind'}.intersection(names):
            df.reset_index('day', inplace=True)
            df = df.merge(dfi[['mean_cloud', 'wind']], left_index=True,
                          right_index=True)
            # mask out non-complete months for cloud validation and months with
            # 0 or 1 cloud fraction
            if 'mean_cloud' in names:
                df.ix[df['mean_cloud_ref'].isnull().values |
                      (df['mean_cloud'] == 0.0) |
                      (df['mean_cloud'] == 1.0),
                      ['mean_cloud_sim', 'mean_cloud_ref']] = np.nan
            # mask out non-complete wind for wind validation and months with
            # a mean wind speed of 0
            if 'wind' in names:
                df.ix[df['wind_ref'].isnull().values | (df['wind'] == 0.0),
                      ['wind_sim', 'wind_ref']] = np.nan
            df.drop(['mean_cloud', 'wind'], 1, inplace=True)
            df.set_index('day', append=True, inplace=True)

        # transform wind
        if self.task_config.transform_wind and 'wind' in names:
            df['wind_ref'] **= 0.5
            df['wind_sim'] **= 0.5
        # calculate the percentiles for each station and month
        g = df.sort_index().groupby(level=['id', 'year'])
        self.logger.debug('Done with basic setup')
        data = g.apply(self.calc)
        if len(data):
            data.index = data.index.droplevel(2)
        self.data = data

    @classmethod
    def _modify_parser(cls, parser):
        parser.setup_args(default_ks_config)
        parser.setup_args(default_quantile_config)
        parser, setup_grp, run_grp = super(
            QuantileEvaluation, cls)._modify_parser(parser)
        parser.update_arg(
            'quantiles', short='q', group=run_grp, type=utils.str_ranges,
            metavar='f1[,f21[-f22[-f23]]]', help=docstrings.dedents("""
                The quantiles to use for calculating the percentiles.
                %(str_ranges.s_help)s."""))
        parser.pop_key('quantiles', 'nargs', None)
        parser.update_arg('no_rounding', short='nr', group=run_grp)
        parser.update_arg('names', short='n', group=setup_grp,
                          nargs='+', metavar='variable',
                          choices=list(cls.names))
        parser.update_arg('transform_wind', short='tw', group=setup_grp)
        return parser, setup_grp, run_grp

    def create_project(self, ds):
        import psyplot.project as psy
        import seaborn as sns
        sns.set_style('white')
        for name, (vref, vsim) in zip(self.names, self.all_variables):
            self.logger.debug('Creating plots of %s', vsim)
            kwargs = dict(precision=0.1) if vref.startswith('prcp') else {}
            psy.plot.densityreg(ds, name='all_' + vsim, coord='all_' + vref,
                                fmt=self.fmt, title='All percentiles',
                                arr_names=['%s_all' % name],
                                **kwargs)
            arr_names = ['%s_%1.2f' % (name, p) for p in ds.pctl.values]
            psy.plot.densityreg(ds, name=vsim, coord=vref, fmt=self.fmt,
                                arr_names=arr_names, pctl=range(ds.pctl.size),
                                **kwargs)
        return psy.gcp(True)[:]

    def make_run_config(self, sp, info):
        for orig in self.names:
            info[orig] = d = OrderedDict()
            for plotter in sp(standard_name=orig).plotters:
                d[plotter.data.pctl if plotter.data.name.startswith('all') else
                  int(plotter.data.pctl.values)] = pctl_d = OrderedDict()
                for key in ['rsquared', 'slope', 'intercept']:
                        val = plotter.plot_data[1].attrs.get(key)
                        if val is not None:
                            pctl_d[key] = float(val)
        return info

    def calc(self, group):
        def calc_percentiles(vname):
            arr = group[vname].values
            arr = arr[~np.isnan(arr)]
            if vname.startswith('prcp'):
                arr = arr[arr > 0]
            if len(arr) == 0:
                return np.array([np.nan] * len(quantiles))
            else:
                return np.percentile(arr, quantiles)
        quantiles = self.task_config.quantiles
        df = pd.DataFrame.from_dict(dict(zip(
            chain(*self.all_variables), map(calc_percentiles,
                                            chain(*self.all_variables)))))
        df['pctl'] = quantiles
        df.set_index('pctl')
        return df

    @staticmethod
    def round_to_ref_prec(ref, sim, func=np.ceil):
        """Round one array to the precision of another

        Parameters
        ----------
        ref: np.ndarray
            The reference array to get the precision from
        sim: np.ndarray
            The simulated array to round
        func: function
            The rounding function to use

        Returns
        -------
        np.ndarray
            Rounded `sim`"""
        ref_sorted = np.unique(ref)
        if len(ref_sorted) < 2:
            return sim
        precision = (ref_sorted[1:] - ref_sorted[:-1]).min()
        return func((sim / precision) * precision)


class KSEvaluation(QuantileEvaluation):
    """Evaluation using a Kolmogorov-Smirnoff test"""

    name = 'ks'

    summary = 'Perform a kolmogorov smirnoff test'

    requires = ['prepare', 'output']

    _datafile = 'kolmogorov_evaluation.csv'

    dbname = 'kolmogorov_evaluation'

    @property
    def default_config(self):
        return default_ks_config()._replace(
            **super(QuantileEvaluation, self).default_config._asdict())

    @staticmethod
    def calc(group):
        def calc(v1, v2, name):
            if len(v1) <= 10 or len(v2) <= 10:
                return {
                    name + '_stat': [np.nan],
                    name + '_p': [np.nan],
                    name: [None],
                    'n' + name + '_sim': [np.nan],
                    'n' + name + '_ref': [np.nan]}
            statistic, p_value = stats.ks_2samp(v1, v2)
            n1 = len(v1)
            n2 = len(v2)
            n = np.sqrt((n1 + n2) / (n1 * n2))
            # if statistic > 1.36 * n, we reject the null hypothesis
            # (alpha = 0.05)
            return {
                name + '_stat': [statistic],
                name + '_p': [p_value],
                name: [statistic > 1.36 * n],
                'n' + name + '_sim': [n1],
                'n' + name + '_ref': [n2]}
        prcp_sim = group.prcp_sim.values[group.prcp_sim.values > 0]
        prcp_ref = group.prcp_ref.values[group.prcp_ref.values > 0]
        tmin_sim = group.tmin_sim.values[
            (group.tmin_sim.values < 100) & (group.tmin_sim.values > -100) &
            (~np.isnan(group.tmin_sim.values))]
        tmin_ref = group.tmin_ref.values[
            (group.tmin_ref.values < 100) & (group.tmin_ref.values > -100) &
            (~np.isnan(group.tmin_ref.values))]
        tmax_sim = group.tmax_sim.values[
            (group.tmax_sim.values < 100) & (group.tmax_sim.values > -100) &
            (~np.isnan(group.tmax_sim.values))]
        tmax_ref = group.tmax_ref.values[
            (group.tmax_ref.values < 100) & (group.tmax_ref.values > -100) &
            (~np.isnan(group.tmax_ref.values))]
        cloud_sl = group.mean_cloud_ref.notnull().values
        cloud_sim = group.mean_cloud_sim.values[cloud_sl]
        cloud_ref = group.mean_cloud_ref.values[cloud_sl]
        wind_sl = group.wind_ref.notnull().values
        wind_sim = group.wind_sim.values[wind_sl]
        wind_ref = group.wind_ref.values[wind_sl]
        return pd.DataFrame.from_dict(dict(chain(*map(six.iteritems, starmap(
            calc, [(prcp_sim, prcp_ref, 'prcp'),
                   (tmin_sim, tmin_ref, 'tmin'),
                   (tmax_sim, tmax_ref, 'tmax'),
                   (cloud_sim, cloud_ref, 'mean_cloud'),
                   (wind_sim, wind_ref, 'wind')])))))

    def significance_fractions(self, series):
        "The percentage of stations with no significant difference"
        return 100. - (len(series.ix[series.notnull() & (series)]) /
                       series.count())*100.

    def run(self, info):
        """Run the evaluation

        Parameters
        ----------
        info: dict
            The configuration dictionary"""
        logger = self.logger
        logger.info('Calculating %s evaluation', self.name)

        df = self.data
        names = list(self.names)
        for name in names:
            info[name] = float(self.significance_fractions(df[name]))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Done. Stations without significant difference:')
            for name, val in info.items():
                logger.debug('    %s: %6.3f %%' % (name, val))
        try:
            import cartopy.crs as ccrs
        except ImportError:
            self.logger.warn(
                "Cartopy is not installed, skipping plot of %s task",
                self.name)
        else:
            self.plot_map()
            info['plot_file'] = self.pdf_file

    def plot_map(self):
        from matplotlib.backends.backend_pdf import PdfPages
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        names = list(self.names)
        df = self.data[names]
        for n in names:
            df[n] = df.pop(n).astype(float)
        g = df.groupby(level='id')
        df_fract = g.agg(dict(zip(
            names, repeat(self.significance_fractions))))
        df_fract = df_fract.merge(
            g.agg(dict(zip(names, repeat('sum')))), left_index=True,
            right_index=True, suffixes=['', '_sum'])
        df_lola = EvaluationPreparation.from_task(self).station_list
        df_lola = df_lola.ix[~df_lola.duplicated('id').values]
        df_lola.set_index('id', inplace=True)
        df_plot = df_lola.merge(df_fract, how='right', left_index=True,
                                right_index=True)
        pdf = PdfPages(self.pdf_file)
        for name in names:
            fig = plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_title(self.names[name]['long_name'])
            ax.coastlines()
            ax.background_patch.set_facecolor('0.95')
            ax.set_global()
            df_plot.sort_values(name, ascending=False).plot.scatter(
                'lon', 'lat', c=name, ax=ax, colorbar=False,
                transform=ccrs.PlateCarree(), cmap='Reds_r', s=5,
                edgecolor='none')

            cbar = fig.colorbar(ax.collections[-1], orientation='horizontal')
            cbar.set_label(
                'Percentage of years not differing significantly[%]')

            pdf.savefig(fig, bbox_inches='tight')

            fig = plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_title(self.names[name]['long_name'])
            ax.coastlines()
            ax.background_patch.set_facecolor('0.95')
            ax.set_global()
            df_plot.sort_values(name, ascending=False).plot.scatter(
                'lon', 'lat', c=name + '_sum', ax=ax, colorbar=False,
                transform=ccrs.PlateCarree(), cmap='Reds', s=5,
                edgecolor='none')

            cbar = fig.colorbar(ax.collections[-1], orientation='horizontal')
            cbar.set_label('Significantly differing years')

            pdf.savefig(fig, bbox_inches='tight')
        pdf.close()
        if self.task_config.close:
            plt.close('all')

    @classmethod
    def _modify_parser(cls, parser):
        parser.setup_args(default_ks_config)
        parser, setup_grp, run_grp = super(
            QuantileEvaluation, cls)._modify_parser(parser)
        parser.update_arg('no_rounding', short='nr', group=run_grp)
        parser.update_arg('names', short='n', group=setup_grp,
                          nargs='+', metavar='variable',
                          choices=list(cls.names))
        parser.update_arg('transform_wind', short='tw', group=setup_grp)
        parser.pop_arg('nc_output', None)
        parser.pop_arg('project_output', None)
        parser.pop_arg('new_project', None)
        parser.pop_arg('project', None)
        return parser, setup_grp, run_grp


_QualityConfig = namedtuple('_QualityConfig', ['quantiles'])


_QualityConfig = utils.append_doc(_QualityConfig, docstrings.get_sections("""
Parameters
----------
quantiles: list of floats
    The quantiles to use for the quality analysis""", '_QualityConfig'))


QualityConfig = utils.enhanced_config(_QualityConfig, 'QualityConfig')


@docstrings.dedent
def default_quality_config(
        quantiles=None, *args, **kwargs):
    """
    The default configuration for :class:`SimulationQuality` instances.
    See also the :attr:`SimulationQuality.default_config` attribute

    Parameters
    ----------
    %(QualityConfig.parameters)s"""
    return QualityConfig(quantiles, *utils.default_config(*args, **kwargs))


class SimulationQuality(Evaluator):
    """Evaluator to provide one value characterizing the quality of the
    experiment

    The applied metric is the mean of

    .. math::
        m = \\left<\{\\left<\{R^2_q\}_{q\in Q}\\right>,
                     \\left<\{1 - |1 - a_q|\}_{q\in Q}\\right>,
                     \{ks\}\}\\right>

    where :math:`\\left<\\right>` denotes the mean of the enclosed set,
    :math:`q\in Q` are the quantiles from the quantile evaluation,
    :math:`R^2_q` the corresponding coefficient of determination and
    :math:`a_q` the slope of quantile :math:`q`. :math:`ks` denotes the
    fraction of stations that do not differ significantly from the observations
    according to the ks test.

    In other words, this quality estimate is the mean of the

        1. coefficients of determination
        2. the deviation from the ideal slope (:math:`a_q == 1`) and
        3. the fraction of stations that do not differ significantly

    Hence, a value of 1 mean high quality, a value of 0 low quality"""

    name = 'quality'

    summary = 'Estimate simulation quality using ks and quantile evaluation'

    has_run = True

    @classmethod
    def _modify_parser(cls, parser):
        parser.add_argument(
            '-q', '--quantiles', metavar='q1[,q1[,q2[,...]]]',
            help="The quantiles to use for the quality analysis",
            type=lambda s: list(map(float, s.split(','))))
        return parser, None, None

    @property
    def default_config(self):
        return default_quality_config()._replace(
            **super(SimulationQuality, self).default_config._asdict())

    def setup_from_scratch(self):
        """Only sets an empty dataframe"""
        self.data = pd.DataFrame([])

    def run(self, info):
        logger = self.logger
        logger.info('Calculating %s evaluation', self.name)
        quants_info = self.config['evaluation'].get('quants')
        ks_info = self.config['evaluation'].get('ks')
        missing = []
        if quants_info is None:
            missing.append('quants task')
        if ks_info is None:
            missing.append('ks task')
        if missing:
            raise ValueError("%s requires that %s has been run before!" % (
                self.name, ' and '.join(missing)))
        quantiles = self.task_config.quantiles
        if quantiles is None:
            quantiles = slice(None)
        possible_names = {'wind', 'prcp', 'tmin', 'tmax', 'mean_cloud',
                          'cloud'}
        for v, v_ks in ks_info.items():
            if v not in quants_info or v not in possible_names:
                continue
            #: Dataframe with intercept, rsquared and slope on index and
            #: quantiles as columns
            df = pd.DataFrame(quants_info[v]).loc[:, quantiles]
            try:
                del df['All']
            except KeyError:
                pass
            slope = float((1 - np.abs(1 - df.loc['slope'].values)).mean())
            rsquared = float(df.loc['rsquared'].values.mean())
            info[v] = OrderedDict([
                ('rsquared', rsquared), ('slope', slope), ('ks', v_ks / 100.),
                ('quality', float(np.mean([rsquared, slope, v_ks / 100.])))])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Simulation quality:')
            for name, val in info.items():
                logger.debug('    %s: %6.3f' % (name, val['quality']))
