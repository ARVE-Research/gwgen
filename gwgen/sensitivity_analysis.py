# -*- coding: utf-8 -*-
"""Module for a sensitivity analysis for one experiment"""
import os
import six
from functools import partial
from argparse import Namespace
import yaml
import os.path as osp
from collections import namedtuple
from itertools import product, chain, repeat, starmap
import logging
import numpy as np
from gwgen.main import OrderedDict, GWGENOrganizer
import gwgen.utils as utils
from gwgen.utils import isstring, docstrings
import matplotlib as mpl


class SensitivityAnalysis(object):
    """Class that performs and manages a sensitivity analysis for the given
    organizer"""

    @property
    def logger(self):
        return logging.getLogger('%s.%s' % (__name__, self.__class__.__name__))

    @property
    def experiment(self):
        """The base experiment of this sensitivity analysis"""
        return self.organizer.experiment

    @property
    def experiments(self):
        projectname = self.projectname
        return self.organizer.config.experiments.project_map[projectname]

    @property
    def exp_config(self):
        return self.organizer.exp_config

    @property
    def config(self):
        """The configuration of the sensitivity analysis"""
        return self.exp_config.setdefault('sensitivity_analysis',
                                          OrderedDict())

    @property
    def projectname(self):
        """The projectname of the sensitivity analysis"""
        return self.config['project']

    @projectname.setter
    def projectname(self, value):
        self.config['project'] = value

    @property
    def project_config(self):
        """The project configuration of the sensitivity analysis"""
        return self.organizer.config.projects.setdefault(self.projectname,
                                                         OrderedDict())

    @property
    def sa_organizer(self):
        """The organizer for this sensitivity analysis"""
        try:
            return self._sa_organizer
        except AttributeError:
            self._sa_organizer = GWGENOrganizer(self.organizer.config)
            try:
                projectname = self.projectname
            except KeyError:
                raise ValueError(
                    "No setup of the sensitivity analysis has been done yet! "
                    "Please run the setup method!")
            self._sa_organizer.projectname = projectname
        return self._sa_organizer

    def __init__(self, organizer):
        """
        Parameters
        ----------
        organizer: gwgen.main.GWGENOrganizer
            The organizer to perform thre sensitivity analysis for
        """
        self.organizer = organizer

    @docstrings.get_sectionsf('SensitivityAnalysis._parallelize_command')
    @docstrings.dedent
    def _parallelize_command(self, kws, experiments=None,
                             loop_exps=False):
        """
        Run a GWGENOrganizer command in parallel for all the experiments in
        this analysis

        Parameters
        ----------
        kws: dict
            Keys must be the name of a command of the :attr:`organizer` of this
            analysis, values must be dictionaries for the corresponding command
        experiments: list of str
            The experiments to use. If None, all experiments are used
        loop_exps: bool
            If True the experiments of the sensitivity analysis are called
            one after each other. Otherwise they are called in parallel, each
            on a single core (unless the the serial option in the global
            configuration is set)
        """
        import multiprocessing as mp
        experiments = experiments or self.experiments
        if len(experiments) == 1 and osp.exists(experiments[0]):
            with open(experiments[0]) as f:
                experiments = [l.rstrip() for l in f.readlines()]
        config = self.organizer.global_config
        commands = self.organizer.commands + list(
            self.organizer.parser_commands.values())
        if not loop_exps and not config.get('serial'):
            all_kws = (
                {key: dict(chain([('experiment', exp)], kws[key].items()))
                 for key in kws.keys() if key in commands}
                for exp in experiments)
            nprocs = config.get('nprocs', 'all')
            if nprocs == 'all':
                nprocs = mp.cpu_count()
            nprocs = min([nprocs, len(experiments)])
            config['serial'] = True
            self.logger.debug('Starting %i processes', nprocs)
            pool = mp.Pool(nprocs)
            res = pool.map_async(self, all_kws)
            for (organizer, ns), experiment in zip(res.get(), experiments):
                changed = organizer.config.experiments[experiment]
                self.organizer.config.experiments[experiment] = changed
            config['serial'] = False
            pool.close()
            pool.join()
            pool.terminate()
        else:
            for experiment in experiments or self.experiments:
                for key, val in kws.items():
                    if key in commands:
                        val['experiment'] = experiment
                kws['experiment'] = experiment
                self.sa_organizer.start(**kws)

    def __call__(self, kws):
        """Call the :meth:`~model_organization.ModelOrganizer.start` of the
        :attr:`organizer` attribute

        Parameters
        ----------
        kws: dict
            Any keywords that are passed to the
            :meth:`~model_organization.ModelOrganizer.start` method

        Returns
        -------
        model_organization.ModelOrganizer
            The :attr:`sa_organizer`
        argparse.Namespace
            The return of the :meth:`~model_organization.ModelOrganizer.start`
            method
        """
        return self.sa_organizer, self.sa_organizer.start(**kws)

    def setup(self, root_dir=None, projectname=None, link=True,
              no_move=False, *args, **kwargs):
        """
        Set up the experiments for the project

        Parameters
        ----------
        root_dir: str
            The path to the root directory where the experiments, etc. will
            be stored. If not given, a new directory will be created at the
            same location as the original project
        projectname: str
            The name of the project that shall be initialized at `root_dir`. A
            new directory will be created namely
            ``root_dir + '/' + projectname``. If not given, defaults to
            ``'sensitivity_<id>'`` where ``'<id>'`` is the experiment id
        link: bool
            If set, the source files are linked to the original ones instead
            of copied
        no_move: bool
            If True, the project in the configuration files is not moved to the
            position right before the configuration of the current project
        """
        config = self.config
        if projectname is None:
            projectname = config.setdefault(
                'project', 'sensitivity_' + str(self.organizer.experiment))
        self.projectname = projectname
        if root_dir is None:
            root_dir = self.project_config.get(
                'root', osp.dirname(self.organizer.project_config['root']))
        self.project_config['root'] = root_dir
        organizer = self.sa_organizer
        organizer.start(setup=dict(
            root_dir=root_dir, projectname=projectname, link=link,
            src_project=self.organizer.projectname))
        if not six.PY2 and not no_move:
            utils.ordered_move(organizer.config.projects, projectname,
                               self.organizer.projectname)

    def compile_model(self):
        """Compile the sensitivity analysis model"""
        self.sa_organizer.start(compile_model=dict(
            projectname=self.projectname))

    def init(self, nml=None, experiment=None, run_prepare=False,
             use_param=False, no_move=False):
        """
        Initialize the experiments for the sensitivity analysis

        Parameters
        ----------
        nml: dict
            A mapping from namelist parameters to the values to use. Ranges
            might be lists of numbers or ``'<i>err'`` to use ``'<i>'``-times
            the error from the parameterization. You might also provide up to
            three values in case on of them is a string with ``'err'`` in it,
            where the first value corresponds to the minimum, the second to
            the maximum and the third to the number of steps.
        experiment: str
            The base experiment name to use. Should somehow contain a number
            which will be increased for each experiment. If None, it will
            default to ``'<id>_sens0'``
        run_prepare: bool
            If True or no ``'input'`` key exists for the experiment, the
            ``'prepare'`` evaluation task is run to get the reference data.
            Otherwise, if False and the ``'input'`` key does not exist, the
            specified file is used
        use_param: bool
            If True/set and the parameterization is required, use the
            parameterization of the original experiment
        no_move: bool
            If True, the experiment in the configurations are not moved to the
            position right before the configuration of the current experiment
        """
        def get_err_range(error, nml, key, values):
            """Compute the ranges for the given namelist `key` based upon the
            mean"""
            converted = [0] * len(values)
            for i, val in enumerate(values):
                if isstring(val) and 'err' in val:
                    m = utils.float_patt.search(val)
                    multiplier = float(m.group()) if m else 1.0
                    converted[i] = nml[key] + multiplier * errors[key]
                else:
                    converted[i] = float(val)
            return list(map(float, np.linspace(*converted, endpoint=True)))

        def setup_paramdir(experiment):
            """Link files from the base experiment param dir to the new one"""
            paramdir = organizer.exp_config.setdefault(
                    'paramdir', osp.join(self.organizer.exp_config['expdir'],
                                         'parameterization'))
            root_dir = self.exp_config.get('paramdir')
            if root_dir and osp.exists(root_dir):
                for f in os.listdir(root_dir):
                    organizer._link(osp.join(root_dir, f),
                                    osp.join(paramdir, f))
            if self.exp_config.get('database'):
                organizer.exp_config['database'] = self.exp_config[
                    'database']

        def update_nml():
            for nml_key, settings in self.exp_config.get(
                    'namelist', {}).items():
                for key, val in settings.items():
                    organizer.exp_config['namelist'][nml_key].setdefault(
                        key, val)

        def transposed_dict(iterable):
            return map(OrderedDict, starmap(zip, zip(
                       repeat(errs.keys()), product(*iterable))))

        logger = self.logger

        if nml is None:
            nml = self.config.get('namelist', self.project_config.get(
                'sensitivity_namelist'))
        if nml is None:
            raise ValueError(
                "No parameter ranges specified! Please use the params "
                "parameter!")
        nml = OrderedDict(nml)
        errs = OrderedDict()
        ranges = OrderedDict()
        for key, val in nml.items():
            if any(isstring(s) and 'err' in s for s in val):
                errs[key] = val
            else:
                ranges[key] = list(map(float, val))
        nml = OrderedDict(chain(ranges.items(), errs.items()))
        self.config['namelist'] = nml.copy()
        self.project_config['sensitivity_namelist'] = nml.copy()
        try:
            eval_stations = self.exp_config['eval_stations']
        except KeyError:
            raise ValueError("No evaluation stations specified for the "
                             "experiment %s!" % (self.experiment))
        if errs:
            from gwgen.parameterization import Parameterizer
            param_stations = self.exp_config.get('param_stations')
            if param_stations is None:
                raise ValueError("No parameterization stations specified for "
                                 "the experiment %s!" % (self.experiment))
            tasks = list(map(Parameterizer.get_task_for_nml_key, errs))
            other_tasks = list(map(Parameterizer.get_task_for_nml_key, ranges))
            err_bins = OrderedDict()
            flags = OrderedDict()
            flags_no_mean = OrderedDict()
        else:
            tasks = []
        organizer = self.sa_organizer
        if experiment is None:
            experiment = self.experiment + '_sens0'

        #: base string for the experiment description. The values are inserted
        #: later
        base_description = ('Sensitivity analysis of %s with %s') % (
            self.experiment, ', '.join('%s={}' % key for key in chain(
                ranges, errs)))
        # choose if we have to evaluate the data
        run_prepare = run_prepare or {'input', 'reference'}.difference(
            self.exp_config)
        if not run_prepare:
            input_file = self.exp_config['input']
            reference = self.exp_config['reference']

        for i, d in enumerate(map(lambda t: OrderedDict(zip(*t)), zip(
                repeat(ranges.keys()), product(*ranges.values())))):
            experiment = utils.get_next_name(experiment)
            organizer.init(projectname=self.projectname, experiment=experiment)
            if not six.PY2 and not no_move:
                utils.ordered_move(organizer.config.experiments, experiment,
                                   self.experiment)
            organizer.exp_config['namelist'] = {'weathergen_ctl': d.copy()}
            if run_prepare:
                organizer.start(evaluate=dict(
                    stations=eval_stations, prepare={'to_csv': True},
                    experiment=experiment))
                input_file = organizer.exp_config['input']
                reference = organizer.exp_config['reference']
                run_prepare = False
            organizer.exp_config['input'] = input_file
            organizer.exp_config['reference'] = reference
            organizer.exp_config['eval_stations'] = eval_stations
            organizer.exp_config['base_exp'] = last = organizer.experiment
            if use_param:
                setup_paramdir(experiment)
            update_nml()
            if not tasks:
                organizer.exp_config['description'] = base_description.format(
                    *d.values())
            else:  # run the parameterization
                organizer.exp_config['param_stations'] = param_stations
                param_config = {t.name: {} for t in tasks}
                for t, (nml_key, val) in zip(other_tasks, d.items()):
                    config_key = t.get_config_key(nml_key)
                    if config_key:
                        param_config[t.name] = {config_key: val}
                organizer.fix_paths(organizer.exp_config)
                organizer.fix_paths(organizer.project_config)

                logger.info('Parameterizing experiment %i', i)
                manager = organizer.start(param=dict(
                    experiment=experiment, **param_config)).param
                param_tasks = {t.name: t for t in manager.tasks}
                nml = organizer.exp_config['namelist']['weathergen_ctl']

                # insert description
                organizer.exp_config['description'] = base_description.format(
                    *chain(d.values(), map(nml.get, errs)))

                # insert meta information that makes it easier to categorize
                # the experiments later
                organizer.exp_config['flags'] = OrderedDict([
                    (key, 'mean') for key in errs])

                errors = {key: param_tasks[t.name].get_error(key)
                          for key, t in zip(errs, tasks)}
                func = partial(get_err_range, errors, nml)
                # Convert the error ranges into lists
                err_ranges = list(starmap(func, errs.items()))
                logger.debug('Error ranges:')
                for key, vals in zip(errs, err_ranges):
                    if not len(vals) % 2:  # insert mean for configuration
                        vals.insert(len(vals) // 2, nml[key])
                    if i == 0:
                        err_bins[key] = np.zeros(
                            (np.product(list(map(len, ranges.values()))),
                             len(vals)))
                        flags[key] = list(map(
                            lambda f: '%4.2ferr' % f, get_err_range(
                                {key: 1}, {key: 0}, key, errs[key])))
                        flags_no_mean[key] = flags[key][:]
                        # insert (or rename) mean for configuration
                        if not len(flags[key]) % 2:
                            flags[key].insert(len(flags[key]) // 2, 'mean')
                        else:
                            flags[key][(len(flags[key]) - 1) // 2] = 'mean'
                            flags_no_mean[key].pop(len(flags[key]) // 2)
                    err_bins[key][i] = vals
                    logger.debug('    %s: %s', key, vals)
                    vals.pop((len(vals) - 1) // 2)  # delete the mean again

                # split them up and use them as namelist input
                # note: d2 is the extended namelist for each experiment
                for d2, fl in zip(transposed_dict(err_ranges), transposed_dict(
                        flags_no_mean.values())):
                    experiment = utils.get_next_name(experiment)
                    combined = OrderedDict(chain(d.items(), d2.items()))
                    organizer.start(init=dict(
                        projectname=self.projectname, experiment=experiment,
                        description=base_description.format(*combined)))
                    if not six.PY2 and not no_move:
                        utils.ordered_move(
                            organizer.config.experiments, experiment,
                            self.experiment)
                    organizer.exp_config['namelist'] = {
                        'weathergen_ctl': combined}
                    update_nml()
                    organizer.exp_config['input'] = input_file
                    organizer.exp_config['reference'] = reference
                    organizer.exp_config['eval_stations'] = eval_stations
                    organizer.exp_config['base_exp'] = last
                    organizer.exp_config['flags'] = fl
        if errs:
            self.config['unstructured'] = uconf = OrderedDict()
            for err, arr in err_bins.items():
                minmeanmax = np.zeros((arr.shape[1], 3))
                minmeanmax[:, 0] = arr.min(axis=0)
                minmeanmax[:, 1] = arr.mean(axis=0)
                minmeanmax[:, 2] = arr.max(axis=0)
                uconf[err] = OrderedDict(zip(flags[key], minmeanmax.tolist()))
                logger.debug('%s', err)
                logger.debug('   minima: %s', minmeanmax[:, 0])
                logger.debug('     mean: %s', minmeanmax[:, 1])
                logger.debug('   maxima: %s', minmeanmax[:, 2])

    docstrings.keep_params('GWGENOrganizer.run.parameters', 'remove')
    docstrings.keep_params(
        'SensitivityAnalysis._parallelize_command.parameters', 'experiments')

    @docstrings.dedent
    def run(self, experiments=None, remove=False):
        """
        Run the analysis

        Parameters
        ----------
        %(SensitivityAnalysis._parallelize_command.parameters.experiments)s
        %(GWGENOrganizer.run.parameters.remove)s"""
        self._parallelize_command(dict(run=dict(remove=remove)),
                                  experiments=experiments)

    docstrings.keep_params(
        'SensitivityAnalysis._parallelize_command.parameters', 'loop_exps')

    @docstrings.dedent
    def evaluate(self, experiments=None, loop_exps=False, **kwargs):
        """
        Evaluate the experiments of the sensitivity analysis

        Parameters
        ----------
        %(SensitivityAnalysis._parallelize_command.parameters.experiments)s
        %(SensitivityAnalysis._parallelize_command.parameters.loop_exps)s
        %(GWGENOrganizer.evaluate.parameters)s
        """
        for key, val in kwargs.items():
            if isinstance(val, Namespace):
                kwargs[key] = val = vars(val)
            if isinstance(val, dict):
                val.pop('experiments', None)
                val.pop('loop_exps', None)
        self._parallelize_command(dict(evaluate=kwargs),
                                  experiments=experiments,
                                  loop_exps=loop_exps)

    @docstrings.dedent
    def plot(self, indicators=['rsquared', 'slope', 'ks', 'quality'],
             names=['prcp', 'tmin', 'tmax', 'mean_cloud', 'wind'],
             meta=None, **kwargs):
        """
        Plot the result of the sensitivity analysis

        2 different plots are made for each variable and quality indicator:

            1. 1d plots with the indicator on the y-axis and the namelist
               parameter on the x-axis
            2. 2d plots with one indicator on the x-axis and another on the
               y-axis for each possible combination

        You can disable one of the plot types via the `only_1d` and `only_2d`
        parameter.

        Parameters
        ----------
        indicators: str or list of str
            The name of the indicators from the `quality` evaluation task
        names: str or list of str
            The name(s) of the variable(s) to plot
        meta: dict or path to yaml configuration file
            Alternative meta information for the data set
        ``**kwargs``
            Any task of the :class:`SensitivityPlot` framework
        """
        meta = meta or self.config.get('additional_meta')
        if isinstance(meta, six.string_types):
            with open(meta) as f:
                meta = yaml.load(f)
        if meta is not None:
            self.config['additional_meta'] = meta
        for task, d in list(kwargs.items()):
            d = kwargs[task] = vars(d) if isinstance(d, Namespace) else dict(d)
            d['names'] = names
            d['indicators'] = indicators
            d['meta'] = meta
            d['config'] = self.exp_config
            d['project_config'] = self.organizer.project_config
            d['sa'] = self
        manager = SensitivityPlot.get_manager(
            config=self.organizer.global_config)
        manager.initialize_tasks(np.array(self.experiments), kwargs)
        manager.setup(np.array(self.experiments))
        manager.run(self.config)
        return manager

    def remove(self, *args, **kwargs):
        """
        Remove the project and files of the sensitivity analysis"""
        self.organizer.start(
            remove=dict(projectname=self.projectname, complete=True,
                        *args, **kwargs))


_SensitivityPlotConfig = namedtuple(
    '_SensitivityPlotConfig', ('sa', 'indicators', 'names', 'meta'))

_SensitivityPlotConfig = utils.append_doc(
    _SensitivityPlotConfig, docstrings.get_sections("""
Parameters
----------
sa: SensitivityAnalysis
    The analyser to use
indicators: str or list of str
    The name of the indicators from the `quality` evaluation task
names: str or list of str
    The name of the variables to plot
meta: dict
    Alternative meta information for the data set
""", '_SensitivityPlotConfig'))

SensitivityPlotConfig = utils.enhanced_config(_SensitivityPlotConfig,
                                              'SensitivityPlotConfig')


@docstrings.dedent
def default_sens_config(
        sa=None, indicators=['rsquared', 'slope', 'ks', 'quality'],
        names=['prcp', 'tmin', 'tmax', 'mean_cloud', 'wind'],
        meta={}, *args, **kwargs):
    """
    The default configuration for :class:`SensitivityPlot` instances.
    See also the :attr:`SensitivityPlot.default_config` attribute

    Parameters
    ----------
    %(SensitivityPlotConfig.parameters)s"""
    return SensitivityPlotConfig(
            sa, indicators, names, meta,
            *utils.default_config(*args, **kwargs))


class SensitivityPlot(utils.TaskBase):
    """A base class for the plots of the :class:`SensitivityAnalysis`"""

    _registry = []

    meta = {
        'rsquared': dict(long_name='Mean R$^2$ of all quantiles'),
        'slope': dict(long_name='Slope indicator'),
        'quality': dict(long_name='Quality indicator'),
        'ks': dict(
            long_name='Fraction of stations without significant difference'),
        'thresh': dict(long_name='Crossover point of Gamma-GP distribution',
                       units='mm'),
        'gp_shape': dict(long_name='Generalized Pareto Shape parameter',
                         units='-')
        }

    variables_meta = {
        'prcp': 'Precipitation',
        'tmin': 'Min. Temperature',
        'tmax': 'Max. Temperature',
        'mean_cloud': 'Cloud fraction'}

    _datafile = 'sensitivity_analysis.csv'

    dbname = 'sensitivity_analysis'

    @property
    def task_data_dir(self):
        """The directory where to store data"""
        return self.sa_dir

    @property
    def namelist(self):
        """The namelist of the sensitivity analysis holding the settings"""
        return self.task_config.sa.config['namelist']

    @property
    def err_nml_keys(self):
        return [key for key, l in self.namelist.items() if any(
            isstring(s) and 'err' in s for s in l)]

    @property
    def ranges_nml_keys(self):
        errs = self.err_nml_keys
        return [key for key in self.namelist if key not in errs]

    @property
    def organizer(self):
        return self.task_config.sa.organizer

    @property
    def base_experiments(self):
        all_exps = self.organizer.config.experiments
        return [exp for exp in self.stations
                if all_exps[exp].get('base_exp') == exp]

    @property
    def default_config(self):
        return default_sens_config()._replace(
            **super(SensitivityPlot, self).default_config._asdict())

    @classmethod
    def _modify_parser(cls, parser):
        parser.setup_args(default_sens_config)
        parser, setup_grp, run_grp = super(
            SensitivityPlot, cls)._modify_parser(parser)
        parser.pop_arg('names')
        parser.pop_arg('indicators')
        parser.pop_arg('sa')
        parser.pop_arg('meta')
        return parser, setup_grp, run_grp

    @property
    def sql_dtypes(self):
        import sqlalchemy
        ret = super(SensitivityPlot, self).sql_dtypes
        for indicator in ['rsquared', 'slope', 'ks', 'quality'] + list(
                self.namelist.keys()):
            ret[indicator] = sqlalchemy.REAL
        ret['vname'] = sqlalchemy.TEXT
        return ret

    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'vname'] + list(self.namelist.keys())
        return super(SensitivityPlot, self).setup_from_file(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'vname'] + list(self.namelist.keys())
        return super(SensitivityPlot, self).setup_from_db(*args, **kwargs)

    def add_meta(self, ds):
        meta = self.meta.copy()
        meta.update(self.task_config.meta)
        for key, attrs in meta.items():
            if key in ds:
                ds[key].attrs.update(attrs)
            if key + '_flag' in ds:
                ds[key + '_flag'].attrs.update(attrs)

    @property
    def ds(self):
        """Gives the base dataset for the plot"""
        import xarray as xr
        ds = xr.Dataset.from_dataframe(self.data.reset_index('id'))
        self.add_meta(ds)
        return ds

    def setup_from_scratch(self):
        import pandas as pd
        experiments = self.stations
        all_exps = self.organizer.config.experiments
        dfs = []
        for variable in self.task_config.names:
            data = [{} for _ in experiments]
            for i, exp in enumerate(experiments):
                try:
                    data[i] = all_exps[exp]['evaluation']['quality'][variable]
                except KeyError:
                    pass
            df = pd.DataFrame(
                data, index=pd.Index(experiments, name='id'))
            df2 = pd.DataFrame(
                [all_exps[exp]['namelist']['weathergen_ctl']
                 for exp in experiments],
                index=pd.Index(experiments, name='id'))
            merged = df.merge(df2, left_index=True, right_index=True)
            if self.err_nml_keys:
                flags_df = pd.DataFrame(
                    [all_exps[exp]['flags'] for exp in experiments],
                    index=pd.Index(experiments, name='id'))
                merged = merged.merge(
                    flags_df, left_index=True, right_index=True,
                    suffixes=('', '_flag'))
            merged['vname'] = variable
            errs_keys = [key + '_flag' for key in self.err_nml_keys]
            dfs.append(merged.set_index(
                ['vname'] + self.ranges_nml_keys + errs_keys,
                append=True))
        self.data = pd.concat(dfs)


class SensitivityPlot1D(SensitivityPlot):
    """1d plots with the indicator on the y-axis and the namelist parameter on
    the x-axis"""

    name = 'plot1d'

    summary = ('1d plots with the indicator on the y-axis and the namelist '
               'parameter on the x-axis')

    has_run = True

    fmt = dict(
        xlabel='{desc}',
        ylabel='%(long_name)s',
        title='%(vname_long)s',
        plot=' ',
        marker='o',
        color='coolwarm')

    def create_project(self, ds):
        import psyplot.project as psy

        def get_key(nml_key):
            return nml_key if nml_key not in errs else nml_key + '_flag'
        vmeta = self.variables_meta
        variables = ds.vname.values
        indicators = self.task_config.indicators
        nml = self.namelist
        errs = self.err_nml_keys
        vmeta = OrderedDict([(key, vmeta[key]) for key in ds.vname.values])
        for nml_key in nml:
            for i, variable in enumerate(variables):
                for ind in indicators:
                    other_keys = {key: range(ds[key].size)
                                  for key in map(get_key,
                                                 set(nml) - {nml_key})}
                    label = ', '.join(
                        map('{0}=%({0})s'.format, other_keys))
                    other_keys['vname'] = i
                    fmt = self.fmt.copy()
                    psy.plot.lineplot(
                        ds, name=ind, dims=other_keys, fmt=fmt,
                        coord=nml_key, legendlabels=label,
                        attrs={'vname_long': vmeta.get(variable)})
        return psy.gcp(True)[:]


_SensitivityPlot2DConfig = namedtuple(
    '_SensitivityPlot2DConfig',
    ['logx', 'logy'] + list(_SensitivityPlotConfig._fields))

_SensitivityPlot2DConfig = utils.append_doc(
    _SensitivityPlot2DConfig, docstrings.get_sections(docstrings.dedents("""
Parameters
----------
logx: bool
    If True, the x-axis will be logarithmic
logy: bool
    If True, the y-axis will be logarithmic
%(_SensitivityPlotConfig.parameters)s
"""), '_SensitivityPlot2DConfig'))

SensitivityPlot2DConfig = utils.enhanced_config(_SensitivityPlot2DConfig,
                                                'SensitivityPlot2DConfig')


@docstrings.dedent
def default_plot2d_config(
        logx=False, logy=False, *args, **kwargs):
    """
    The default configuration for :class:`SensitivityPlot` instances.
    See also the :attr:`SensitivityPlot.default_config` attribute

    Parameters
    ----------
    %(SensitivityPlot2DConfig.parameters)s"""
    return SensitivityPlot2DConfig(
            logx, logy, *default_sens_config(*args, **kwargs))


class SensitivityPlot2D(SensitivityPlot):
    """2d plots with one namelist parameter on the y-axis and one on the
    x-axis and a color coding the the indicator"""

    name = 'plot2d'

    summary = ('2d plots with one namelist parameter on the y-axis and one on '
               'the x-axis and a color coding the the indicator')

    has_run = True

    @property
    def default_config(self):
        return default_plot2d_config()._replace(
            **super(SensitivityPlot2D, self).default_config._asdict())

    @classmethod
    def _modify_parser(cls, parser):
        parser.setup_args(default_plot2d_config)
        parser, setup_grp, run_grp = super(
            SensitivityPlot2D, cls)._modify_parser(parser)
        parser.update_arg('logx', group=run_grp)
        parser.update_arg('logy', group=run_grp)
        return parser, setup_grp, run_grp

    def run(self, info):
        import psyplot.data as psyd
        import matplotlib.pyplot as plt
        import psy_simple.plotters as psyps
        import matplotlib.colors as mcol
        import seaborn as sns
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.patches as patches
        sns.set_style('white')
        full_df = self.data.reset_index()
        plot_output = self.task_config.plot_output
        pdf = PdfPages(plot_output)
        all_exps = self.organizer.config.experiments
        for ind in self.task_config.indicators:
            for variable in self.task_config.names:
                df = full_df[full_df.vname == variable]
                vals = {
                    key: group.set_index('gp_shape').sort_index()[ind]
                    for key, group in df.groupby('thresh')}
                y = psyd._infer_interval_breaks(sorted(vals.keys()))
                fig = plt.figure(edgecolor='none', facecolor='none')
                if mpl.__version__ < '2.0':
                    ax = plt.axes(axisbg='0.9')
                else:
                    ax = plt.axes(facecolor='0.9')
                all_vals = np.concatenate(list(vals.values()))
                vmin, vmax = psyps.DataTicksCalculator._round_min_max(
                    np.nanmin(all_vals), np.nanmax(all_vals))
                norm = mcol.BoundaryNorm(
                    np.linspace(vmin, vmax, 11, endpoint=True), 10)
                cmap = plt.get_cmap('Greens', 10)
                # set missing values to (dark) red
                cmap.set_bad((0.7686274509803922, 0.3058823529411765,
                              0.3215686274509804))
                plots = []
                x_vals = {}
                for i, key in enumerate(sorted(df.thresh.unique())):
                    data = np.ma.array(vals[key], mask=np.isnan(vals[key]))
                    x_vals[key] = x = psyd._infer_interval_breaks(
                        vals[key].index.values)
                    plots.append(ax.pcolormesh(
                        x, y[i:i+2], data[np.newaxis, :], norm=norm,
                        cmap=cmap))
                ax.set_xlabel('Generalized Pareto shape parameter')
                ax.set_ylabel('Gamma-GP crossover point [mm]')
                ax.set_title(self.variables_meta[variable])
                cbar = plt.colorbar(plots[-1], orientation='horizontal')
                cbar.set_label(self.meta[ind]['long_name'])
                if self.task_config.logx:
                    ax.set_xscale('log')
                if self.task_config.logy:
                    ax.set_yscale('log')
                # draw box around colorbar
                plt.rcParams['axes.edgecolor'] = 'k'
                if mpl.__version__ < '2.0':
                    ax = plt.axes([0.08, 0.03, 0.85, 0.22], axisbg='none')
                else:
                    ax = plt.axes([0.08, 0.03, 0.85, 0.22], fc='none')
                ax.grid(False)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False,
                                               edgecolor='k', linewidth=2))
                if not self.err_nml_keys:
                    pdf.savefig(fig)
                    continue
                mean_boxes = []
                cmap = plt.get_cmap('coolwarm', 2)
                means = {}
                medians = {}  # XXX currently not used
                for exp in self.base_experiments:
                    nml = all_exps[exp]['namelist']['weathergen_ctl']
                    param_config = all_exps[exp]['parameterization']['prcp']
                    means[nml['thresh']] = param_config['gpshape_mean']
                    medians[nml['thresh']] = param_config['gpshape_median']
                for i, key in enumerate(sorted(df.thresh.unique())):
                    mean_index = df.gp_shape[
                        df.thresh == key].values.searchsorted(means[key])
                    mean_boxes.append(patches.Rectangle(
                        [x_vals[key][mean_index], y[i]],
                        x_vals[key][mean_index + 1] - x_vals[key][mean_index],
                        y[i + 1] - y[i], edgecolor=cmap(0), facecolor='none',
                        lw=2.0))
                    ax.add_patch(mean_boxes[-1])
                plt.legend([mean_boxes[0]],
                           ['mean'], title='GP shape parameter')
                pdf.savefig(fig)
        pdf.close()
        info['plot_file'] = plot_output
