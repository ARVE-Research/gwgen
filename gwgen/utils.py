from __future__ import division
import inspect
import os
import os.path as osp
import shutil
import re
import six
import copy
import logging
import abc
from itertools import chain, groupby
from collections import namedtuple
import pandas as pd
import numpy as np
import docrep
from tempfile import mkdtemp
from psyplot.compat.pycompat import OrderedDict, filterfalse
from psyplot.config.rcsetup import safe_list


docstrings = docrep.DocstringProcessor()


float_patt = re.compile(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')


logger = logging.getLogger(__name__)


if six.PY2:
    FileExistsError = OSError


def download_file(url, target=None):
    """Download a file from the internet

    Parameters
    ----------
    url: str
        The url of the file
    target: str or None
        The path where the downloaded file shall be saved. If None, it will be
        saved to a temporary directory

    Returns
    -------
    file_name: str
        the downloaded filename"""
    logger.info('Downloading %s to %s', url, target)
    if target is not None and not osp.exists(osp.dirname(target)):
        os.makedirs(osp.dirname(target))
    if six.PY3:
        from urllib import request
        return request.urlretrieve(url, target)[0]
    else:
        import urllib
        ret = urllib.urlretrieve(url, target)[0]
        # workaround. clean up urls. otherwise you get problems if downloading
        # more than one file from the same source
        # (http://bugs.python.org/issue27973)
        urllib.urlcleanup()
        return ret


def dir_contains(dirname, path, exists=True):
    """Check if a file of directory is contained in another.

    Parameters
    ----------
    dirname: str
        The base directory that should contain `path`
    path: str
        The name of a directory or file that should be in `dirname`
    exists: bool
        If True, the `path` and `dirname` must exist

    Notes
    -----
    `path` and `dirname` must be either both absolute or both relative
    paths"""
    if exists:
        dirname = osp.abspath(dirname)
        path = osp.abspath(path)
        if six.PY3:
            return osp.samefile(osp.commonpath([dirname, path]), dirname)
        else:
            return osp.exists(path) and osp.samefile(
                osp.commonprefix([dirname, path]), dirname)
    return dirname in osp.commonprefix([dirname, path])


def isstring(s):
    return isinstance(s, six.string_types)

docstrings.params['str_ranges.s_help'] = """
    A comma (``','``) separated string. A single value in this string
    represents one number, ranges can also be used via a separation by
    comma (``'-'``). Hence, ``'2009,2012-2015'`` will be
    converted to ``[2009,2012, 2013, 2014]`` and ``2009,2012-2015-2`` to
    ``[2009, 2012, 2015]``"""


minus_patt = re.compile(r'(?<!^)(?<!-)-')


@docstrings.dedent
def str_ranges(s):
    """
    Convert a string of comma separated values to an iterable

    Parameters
    ----------
    s: str%(str_ranges.s_help)s

    Returns
    -------
    list
        The values in s converted to a list"""
    def get_numbers(s):
        splitted = minus_patt.split(s)
        try:
            nums = list(map(float, minus_patt.split(s)))
        except ValueError:
            return splitted
        if len(nums) == 1:
            return nums
        else:
            import numpy as np
            return np.arange(*nums)
    return list(chain.from_iterable(map(get_numbers, s.split(','))))


def unique_everseen(iterable, key=None):
    """List unique elements, preserving order. Remember all elements ever seen.

    Function taken from https://docs.python.org/2/library/itertools.html"""
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def get_next_name(old, fmt='%i'):
    """Return the next name that numerically follows `old`"""
    nums = re.findall('\d+', old)
    if not nums:
        raise ValueError("Could not get the next name because the old name "
                         "has no numbers in it")
    num0 = nums[-1]
    num1 = str(int(num0) + 1)
    return old[::-1].replace(num0[::-1], num1[::-1], 1)[::-1]


docstrings.params['get_value_note'] = """
    If the key goes some
    levels deeper, keys may be separated by a ``'.'`` (e.g.
    ``'namelists.weathergen'``). Hence, to insert a ``','``, it must
    be escaped by a preceeding ``'\'``."""


@docstrings.dedent
def go_through_dict(key, d, setdefault=None):
    """
    Split up the `key` by . and get the value from the base dictionary `d`

    Parameters
    ----------
    key: str
        The key in the `config` configuration. %(get_value_note)s
    d: dict
        The configuration dictionary containing the key
    setdefault: callable
        If not None and an item is not existent in `d`, it is created by
        calling the given function

    Returns
    -------
    str
        The last level of the key
    dict
        The dictionary in `d` that contains the last level of the key
    """
    patt = re.compile(r'(?<!\\)\.')
    sub_d = d
    splitted = patt.split(key)
    n = len(splitted)
    for i, k in enumerate(splitted):
        if i < n - 1:
            if setdefault is not None:
                sub_d = sub_d.setdefault(k, setdefault())
            else:
                sub_d = sub_d[k]
        else:
            return k, sub_d


def safe_csv_append(df, path, *args, **kwargs):
    """Convenience method to dump a data frame to csv without removing the old

    This function dumps the given `df` to the file specified by `path`. If
    `path` already exists, we read the header of the file and sort `df`
    according to this header

    Parameters
    ----------
    df: pandas.DataFrame
        The data frame to store
    path: str
        The path where to store the data
    ``**kwargs``
        Any other keyword for the :meth:`pandas.DataFrame.to_csv` method"""
    exists = osp.exists(path)
    if exists:
        idx_names = df.index.names
        order = [col for col in pd.read_csv(path, nrows=1).columns
                 if col not in idx_names]
    else:
        order = list(df.columns)
    df[order].to_csv(path, mode='a', *args, header=not exists, **kwargs)


def ordered_move(d, to_move, pos):
    """Move a key in an ordered dictionary to another position

    Parameters
    ----------
    d: collections.OrderedDict
        The dictionary containing the keys
    to_move: str
        The key to move
    pos: str
        The name of the key that should be followed by `to_move`
    """
    keys = list(d)
    i2move = keys.index(to_move)
    orig_pos = keys.index(pos)
    if i2move > orig_pos:
        d.move_to_end(to_move)
        for key in keys[orig_pos:i2move] + keys[i2move+1:]:
            d.move_to_end(key)
    else:
        for key in keys[i2move + 1:orig_pos]:
            d.move_to_end(key)
        d.move_to_end(to_move)
        for key in keys[orig_pos:]:
            d.move_to_end(key)


_engines = {}


@docstrings.get_sectionsf('get_postgres_engine')
@docstrings.dedent
def get_postgres_engine(database, user=None, host='127.0.0.1', port=None,
                        create=False, test=False):
    """
    Get the engine to access the given `database`

    This method creates an engine using sqlalchemy's create_engine function
    to access the given `database` via postgresql. If the database is not
    existent, it will be created

    Parameters
    ----------
    database: str
        The name of a psql database. If provided, the processed data will
        be stored
    user: str
        The username to use when logging into the database
    host: str
        the host which runs the database server
    port: int
        The port to use to log into the the database
    create: bool
        If True, it is tried to create the database if not existent as postgres
        user
    test: bool
        If True, test the connection before returning the engine

    Returns
    -------
    sqlalchemy.engine.base.Engine
        Tha engine to access the database

    Notes
    -----
    The engine is for single usage!"""
    import sqlalchemy
    logger = logging.getLogger(__name__)
    base_str = 'postgresql://'
    if user:
        base_str += user + '@'
    base_str += host
    if port:
        base_str += ':' + port
    engine_str = base_str + '/' + database  # to create the database
    logger.info("Creating engine with %s", engine_str)
    engine = sqlalchemy.create_engine(engine_str,
                                      poolclass=sqlalchemy.pool.NullPool)
    if test:
        try:
            logger.debug("Try to connect...")
            conn = engine.connect()
        except sqlalchemy.exc.OperationalError:
            # data base does not exist, so create it
            logger.debug("Failed...", exc_info=True)
            if create:
                logger.debug("Creating database by logging into postgres")
                pengine = sqlalchemy.create_engine(base_str + '/postgres')
                conn = pengine.connect()
                conn.execute('commit')
                conn.execute('CREATE DATABASE ' + database)
                conn.close()
            else:
                return None, engine_str
        else:
            conn.close()
            engine = sqlalchemy.create_engine(
                engine_str, poolclass=sqlalchemy.pool.NullPool)
    logger.debug('Done.')
    return engine, engine_str


def file_len(fname):
    """Get the number of lines in `fname`"""
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_module_path(mod):
    """Convenience method to get the directory of a given python module"""
    return osp.dirname(inspect.getabsfile(mod))


def get_toplevel_module(mod):
    return mod.__name__.split('.')[0]


def _requirement_property(requirement):

    def get_x(self):
        return self._requirements[requirement]

    return property(
        get_x, doc=requirement + " parameterization instance")


def append_doc(namedtuple_cls, doc):
    if six.PY3:
        namedtuple_cls.__doc__ += '\n' + doc
        return namedtuple_cls
    else:
        class DocNamedTuple(namedtuple_cls):
            __doc__ = namedtuple_cls.__doc__ + '\n' + doc
            __slots__ = ()
        DocNamedTuple.__name__ = namedtuple_cls.__name__
        return DocNamedTuple


_SetupConfig = namedtuple(
    '_SetupConfig', ['setup_from', 'to_csv', 'to_db', 'remove',
                     'skip_filtering'])

_SetupConfig = append_doc(_SetupConfig, docstrings.get_sections("""
Configuration for the setup of tasks via their :meth:`~TaskBase.setup`

Parameters
----------
setup_from: { 'scratch' | 'file' | 'db' | None }
    The method how to setup the instance either from

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
to_csv: bool
    If True, the data at setup will be written to a csv file
to_db: bool
    If True, the data at setup will be written to into a database
remove: bool
    If True and the old data file already exists, remove before writing to it
skip_filtering: bool
    If True, skip the filtering for the correct stations in the datafile
""", '_SetupConfig'))


_RunConfig = namedtuple(
    '_RunConfig',
    ['plot_output', 'nc_output', 'project_output', 'new_project', 'project',
     'close'])

_RunConfig = append_doc(_RunConfig, docstrings.get_sections("""
Configuration for the run of tasks via their :meth:`~TaskBase.setup`

Parameters
----------
plot_output: str
    An alternative path to use for the PDF file of the plot
nc_output: str
    An alternative path (or multiples depending on the task) to use for the
    netCDF file of the plot data
project_output: str
    An alternative path to use for the psyplot project file of the plot
new_project: bool
    If True, a new project will be created even if a file in `project_output`
    exists already
project: str
    The path to a psyplot project file to use for this parameterization
close: bool
    Close the project at the end
""", '_RunConfig'))

TaskConfig = namedtuple(
    'TaskConfig', _SetupConfig._fields + _RunConfig._fields)

TaskConfig = append_doc(TaskConfig, docstrings.get_sections(
    docstrings.dedents("""
Configuration of tasks for their :meth:`~TaskBase.setup` and
:meth:`~TaskBase.run` methods.

Parameters
----------
%(_SetupConfig.parameters)s
%(_RunConfig.parameters)s"""), 'TaskConfig'))


@docstrings.dedent
def default_config(
        setup_from=None, to_csv=False, to_db=False, remove=False,
        skip_filtering=False,
        plot_output=None, nc_output=None, project_output=None,
        new_project=False, project=None, close=True):
    """
    The default configuration for TaskBase instances. See also the
    :attr:`TaskBase.default_config` attribute

    Parameters
    ----------
    %(TaskConfig.parameters)s"""
    return TaskConfig(setup_from, to_csv, to_db, remove, skip_filtering,
                      plot_output, nc_output, project_output, new_project,
                      project, close)


class TaskMeta(abc.ABCMeta):
    """Meta class for the :class:`TaskBase`"""

    def __new__(cls, name, bases, namespace):
        new_cls = super(TaskMeta, cls).__new__(
            cls, name, bases, namespace)
        if new_cls.name:
            new_cls._registry.append(new_cls)
        for requirement in new_cls.setup_requires:
            setattr(new_cls, requirement, _requirement_property(requirement))
        return new_cls


def _param_worker(kwargs):
    from gwgen.parameterization import Parameterizer
    return Parameterizer.process_data(
        db_locks=_db_locks, file_locks=_file_locks, **kwargs)


def init_interprocess_locks(db_locks, file_locks, lock_dir):
    from fasteners import InterProcessLock
    logger.debug('Initializing %i locks', len(db_locks) + len(file_locks))
    for db_lock in db_locks:
        fname = osp.join(lock_dir, 'db_' + db_lock + '.lock')
        _db_locks[db_lock] = InterProcessLock(fname)
    for file_lock in file_locks:
        fname = osp.join(lock_dir,
                         'file_' + osp.basename(file_lock) + '.lock')
        _file_locks[file_lock] = InterProcessLock(fname)


def init_locks(db_locks, file_locks):
    _db_locks.update(db_locks)
    _file_locks.update(file_locks)

_db_locks = {}
_file_locks = {}


def enhanced_config(config_cls, name):
    ret = namedtuple(name, config_cls._fields + TaskConfig._fields)

    return append_doc(ret, docstrings.get_sections(docstrings.dedents("""
        Configuration of the :class:`EvaluationPreparation` class

        Parameters
        ----------
        %({}.parameters)s
        %(TaskConfig.parameters)s
        """.format(config_cls.__name__)), name))


@six.add_metaclass(TaskMeta)
class TaskBase(object):
    """Abstract base class for parameterization and evaluation tasks

    Abstract base class that introduces the methods for the parameterization
    and evaluation framework. The name of the task is specified in the
    :attr:`name` attribute. You can implement the connection to other
    tasks (within the same framework) in the :attr:`setup_requires` attribute.
    The corresponding instances to the identifiers in the
    :attr:`setup_requires` attribute can later be accessed through the given
    attribute.

    Examples
    --------
    Let's define a parameterizer that does nothing but setup_requires another
    parameterization task named *cloud* as connection::

        >>> class CloudParameterizer(Parameterizer):
        ...     name = 'cloud'
        ...     def setup_from_scratch(self):
        ...         pass
        ...
        >>> class DummyParameterizer(Parameterizer):
        ...     setup_requires = ['cloud']
        ...     name = 'dummy'
        ...     def setup_from_scratch(self):
        ...         pass
        ...
        >>> cloud = CloudParameterizer()
        >>> dummy = DummyParameterizer(cloud=cloud)
        >>> dummy.cloud is cloud
        True"""

    #: The registered parameterization classes (are set up automatically by
    #: :class:`TaskMeta`). If you want to start a new framework, set this
    #: variable at class definition of your framework base
    _registry = []

    #: list of str. identifiers of required classes for this task
    setup_requires = []

    #: required tasks for this instance. See :meth:`set_requirements`
    _requirements = None

    #: str. name of the task
    name = None

    #: pandas.DataFrame. The dataframe holding the daily data
    data = None

    #: str. The basename of the csv file where the data is stored by the
    #: :meth:`TaskBase.write2file` method and read by the
    #: :meth:`TaskBase.setup_from_file`
    _datafile = ""

    #: The database name to use
    dbname = ''

    #: str. summary of what this task does
    summary = ''

    #: dict. Formatoptions to use when making plots with this task
    fmt = {}

    #: bool. Boolean that is True if there is a run method for this task
    has_run = False

    #: bool. Boolean that is True if the task can be setup in parallel
    setup_parallel = True

    #: :class:`threading.Thread` objects that are started during the setup.
    #: It will be waited for them to finish before continuing with another
    #: process
    threads = []

    @property
    def default_config(self):
        """The default configuration of this task inserted with the
        :attr:`pdf_file`, :attr:`nc_file` and :attr:`project_file` attributes
        """
        return default_config()._replace(
            plot_output=self.pdf_file, nc_output=self.nc_file,
            project_output=self.project_file)

    @property
    def data_dir(self):
        """str. Path to the directory where the source data of the project
        is located"""
        return self.project_config['data']

    @property
    def param_dir(self):
        """str. Path to the directory were the processed parameterization
        data is stored"""
        ret = self.config.setdefault('paramdir', osp.join(
            self.config['expdir'], 'parameterization'))
        if not osp.exists(ret):
            try:
                os.makedirs(ret)
            except FileExistsError:
                pass
        return ret

    @property
    def cloud_dir(self):
        """str. Path to the directory were the processed parameterization
        data is stored"""
        ret = self.config.setdefault('clouddir', osp.join(
            self.config['expdir'], 'cloud_parameterization'))
        if not osp.exists(ret):
            try:
                os.makedirs(ret)
            except FileExistsError:
                pass
        return ret

    @property
    def eval_dir(self):
        """str. Path to the directory were the processed evaluation data is
        stored"""
        ret = self.config.setdefault('evaldir', osp.join(
            self.config['expdir'], 'evaluation'))
        if not osp.exists(ret):
            try:
                os.makedirs(ret)
            except FileExistsError:
                pass
        return ret

    @property
    def sa_dir(self):
        """str. Path to the directory were the processed sensitivity analysis
        data is stored"""
        ret = self.config.setdefault('sadir', osp.join(
            self.config['expdir'], 'sensitivity_analysis'))
        if not osp.exists(ret):
            try:
                os.makedirs(ret)
            except FileExistsError:
                pass
        return ret

    @property
    def input_dir(self):
        """str. Path to the directory were the input data is stored"""
        ret = self.config.setdefault('indir', osp.join(
            self.config['expdir'], 'input'))
        if not osp.exists(ret):
            try:
                os.makedirs(ret)
            except FileExistsError:
                pass
        return ret

    @property
    def reference_path(self):
        """The path to the reference file in the configuration"""
        return self.config.get(
            'reference', osp.join(self.eval_dir, 'reference.csv'))

    @reference_path.setter
    def reference_path(self, value):
        self.config['reference'] = value

    @property
    def df_ref(self):
        """The reference data frame"""
        df = pd.read_csv(self.reference_path,
                         index_col=['id', 'year', 'month', 'day'])
        stations = list(self.stations)
        if len(stations) == 1:
            stations = slice(stations[0], stations[0])
        return df.loc(axis=0)[stations]

    @property
    def input_path(self):
        """The path to the project input file in the configuration"""
        return self.config.get(
            'input', osp.join(self.input_dir, 'input.csv'))

    @input_path.setter
    def input_path(self, value):
        self.config['input'] = value

    @property
    def output_dir(self):
        """str. Path to the directory were the input data is stored"""
        ret = self.config.setdefault('outdir', osp.join(
            self.config['expdir'], 'output'))
        if not osp.exists(ret):
            try:
                os.makedirs(ret)
            except FileExistsError:
                pass
        return ret

    @property
    def output_path(self):
        """The path to the project output file in the configuration"""
        return self.config['outdata']

    @abc.abstractproperty
    def task_data_dir(self):
        """The directory where to store data"""
        pass

    @property
    def datafile(self):
        """str. The path to the csv file where the data is stored by the
        :meth:`Parameterizer.write2file` method and read by the
        :meth:`Parameterizer.setup_from_file`"""
        if isinstance(self._datafile, six.string_types):
            return osp.join(self.task_data_dir, self._datafile)
        else:
            return list(map(lambda f: osp.join(self.task_data_dir, f),
                            self._datafile))

    @property
    def nc_file(self):
        """NetCDF file for the project"""
        return osp.join(self.task_data_dir, self.name + '.nc')

    @property
    def project_file(self):
        """Pickle file for the project"""
        return osp.join(self.task_data_dir, self.name + '.pkl')

    @property
    def pdf_file(self):
        """pdf file with figures the project"""
        return osp.join(self.task_data_dir, self.name + '.pdf')

    @property
    def engine(self):
        """The sqlalchemy engine to access the database"""
        global_config = self.global_config
        database = self.config.get(
            'database', self.project_config.get(
                'database', self.global_config.get('database')))
        if not database or global_config.get('no_postgres'):
            return None
        # first we check whether everything works with the database
        # We add 'or None' explicitly because otherwise the user would not be
        # able to reset the settings
        user = global_config.get('user') or None
        port = global_config.get('port') or None
        host = global_config.get('host') or '127.0.0.1'
        return get_postgres_engine(database, user, host, port)[0]

    @property
    def sql_dtypes(self):
        """The data types to write the data into a postgres database"""
        import sqlalchemy

        def get_names(df):
            if df is not None:
                return chain(df.columns, df.index.names)
            return []

        dtype = {
            'station_id': sqlalchemy.INTEGER, 'tmin': sqlalchemy.REAL,
            'id': sqlalchemy.TEXT, 'prcp': sqlalchemy.REAL,
            'tmax': sqlalchemy.REAL, 'mean_cloud': sqlalchemy.REAL,
            'wet_day': sqlalchemy.SMALLINT, 'ndaymon': sqlalchemy.SMALLINT,
            'year': sqlalchemy.SMALLINT, 'month': sqlalchemy.SMALLINT,
            'day': sqlalchemy.SMALLINT, 'hour': sqlalchemy.SMALLINT,
            'wind': sqlalchemy.REAL}
        if not isinstance(self._datafile, six.string_types):
            names = set(chain.from_iterable(map(get_names, self.data)))
            used_types = names.intersection(dtype)
            return {n: [dtype[n]] * len(self.data) for n in used_types}
        else:
            names = list(chain(self.data.columns, self.data.index.names))
            return {key: val for key, val in dtype.items() if key in names}

    _logger = None

    @property
    def logger(self):
        """The logger of this task"""
        if self._logger is None:
            self.logger = None
        return self._logger

    @logger.setter
    def logger(self, value):
        if isinstance(value, six.string_types):
            value = logging.getLogger(value)
        elif value is None and not self.global_config.get('serial'):
            import multiprocessing as mp
            value = logging.getLogger('.'.join(
                [__name__, self.__class__.__name__, self.name or '',
                 mp.current_process().name]))
        elif value is None:
            value = logging.getLogger('.'.join(
                [__name__, self.__class__.__name__, self.name or '']))
        self._logger = value

    @property
    def setup_from(self):
        ret = self.task_config.setup_from
        if ret is None:
            ret = self._get_setup()
            self.setup_from = ret
        return ret

    @setup_from.setter
    def setup_from(self, value):
        self.task_config = self.task_config._replace(setup_from=value)

    @docstrings.get_sectionsf('TaskBase')
    @docstrings.dedent
    def __init__(self, stations, config, project_config, global_config,
                 data=None, requirements=None, *args, **kwargs):
        """
        Parameters
        ----------
        stations: list
            The list of stations to process
        config: dict
            The configuration of the experiment
        project_config: dict
            The configuration of the underlying project
        global_config: dict
            The global configuration
        data: pandas.DataFrame
            The data to use. If None, use the :meth:`setup` method
        requirements: list of :class:`TaskBase` instances
            The required instances. If None, you must call the
            :meth:`set_requirements` method later

        Other Parameters
        ----------------
        ``*args, **kwargs``
            The configuration of the task. See the :class:`TaskConfig` for
            arguments. Note that if you provide ``*args``, you have to provide
            all possible arguments
        """
        self.global_config = global_config
        self.config = config
        self.project_config = project_config
        if args:
            self.task_config = self.default_config._make(args)
        else:
            for key, val in kwargs.items():
                if val is None:
                    kwargs[key] = getattr(self.default_config, key, None)
            self.task_config = self.default_config._replace(**kwargs)
        self.stations = stations
        # overwrite the class attribute of the formatoptions
        self.fmt = self.fmt.copy()
        if data is not None or isinstance(self._datafile, six.string_types):
            self.data = data
        else:
            self.data = [[] for i in range(len(self._datafile))]

        if requirements is not None:
            self._requirements = requirements

        if self.task_config.remove:
            for datafile in safe_list(self.datafile):
                if osp.exists(datafile):
                    self.logger.debug('Removing %s', datafile)
                    os.remove(datafile)

    def __reduce__(self):
        if 'remove' in self.task_config._fields:
            config = self.task_config._replace(remove=False)
        else:
            config = self.task_config
        return self.__class__, tuple(chain(
            (self.stations, self.config, self.project_config, self.global_config,
             self.data, self._requirements), config))

    docstrings.delete_params(
        'TaskBase.parameters', 'config', 'project_config', 'global_config')

    @classmethod
    @docstrings.dedent
    def from_organizer(cls, organizer, stations, *args, **kwargs):
        """
        Create a new instance from a :class:`model_organization.ModelOrganizer`

        Parameters
        ----------
        organizer: model_organization.ModelOrganizer
            The organizer to use the configuration from
        %(TaskBase.parameters.no_config|project_config|global_config)s

        Other Parameters
        ----------------
        %(TaskBase.other_parameters)s

        Returns
        -------
        TaskBase
            An instance of the calling class
        """
        return cls(stations, organizer.exp_config, organizer.project_config,
                   organizer.global_config, *args, **kwargs)

    docstrings.delete_params(
        'TaskBase.parameters', 'stations', 'config', 'project_config',
        'global_config')

    @classmethod
    @docstrings.get_summaryf('TaskBase.from_task')
    @docstrings.get_sectionsf('TaskBase.from_task')
    @docstrings.dedent
    def from_task(cls, task, *args, **kwargs):
        """
        Create a new instance from another task

        Parameters
        ----------
        task: TaskBase
            The organizer to use the configuration from. Note that it can also
            be of a different type than this class
        %(TaskBase.parameters.no_stations|config|project_config|global_config)s

        Other Parameters
        ----------------
        %(TaskBase.other_parameters)s

        See Also
        --------
        setup_from_instances: To combine multiple instances of the class

        Notes
        -----
        Besides the `skip_filtering` parameter, the :attr:`task_config` is not
        inherited from `task`
        """
        if getattr(task.task_config, 'skip_filtering', None):
            kwargs.setdefault('skip_filtering',
                              task.task_config.skip_filtering)
        return cls(task.stations, task.config, task.project_config,
                   task.global_config, *args, **kwargs)

    def set_requirements(self, requirements):
        """Set the requirements for this task

        Parameters
        ----------
        requirements: list of :class:`TaskBase` instances
            The tasks as specified in the :attr:`setup_requires` attribute"""
        d = {t.name: t for t in requirements}
        if self._requirements is not None:
            for key, val in self._requirements.items():
                d.setdefault(key, val)
        missing = set(self.setup_requires).difference(d)
        if self.setup_from == 'scratch' and missing:
            raise ValueError(
                "Missing requirements %s for %s task!" % (
                    ', '.join(missing), self.name))
        self._requirements = d

    def _get_setup(self):
        if self._datafile and all(map(osp.exists, safe_list(self.datafile))):
            return 'file'
        if self.dbname:
            engine = self.engine
            if engine is not None and all(map(
                    engine.has_table, safe_list(self.dbname))):
                return 'db'
        return 'scratch'

    @docstrings.get_sectionsf('TaskBase._setup_or_init')
    @docstrings.dedent
    def _setup_or_init(self, method=None):
        """
        Method to initialize or setup the data of a task

        This method is called by :meth:`setup` and :meth:`init_task` and
        calls :meth:`setup_from_file`, :meth:`setup_from_db` and
        :meth:`setup_from_scratch` (or the corresponding `init` method)
        depending  on :attr:`setup_from`

        Parameters
        ----------
        method: { 'setup' | 'init' }
            The methods to call. If method is ``'setup'``, the (depending on
            :attr:`setup_from`), e.g. the :meth:`setup_from_scratch` is called,
            otherwise (e.g.) the :meth:`init_from_scratch` method is called
        """
        if self.setup_requires and self._requirements is None:
            raise ValueError('set_requirements method has not been called!')
        setup_from = self.setup_from
        self.logger.debug('%s from %s', method, setup_from)
        ret = getattr(self, method + '_from_' + setup_from)()
        self.logger.debug('Done.')
        return ret

    @docstrings.dedent
    def init_task(self):
        """
        Method that is called on the I/O-Processor to initialize the setup"""
        return self._setup_or_init('init')

    @docstrings.dedent
    def setup(self):
        """
        Set up the database for this task
        """
        from threading import Thread
        ret = self._setup_or_init('setup')
        if self.task_config.to_csv:
            thread = Thread(target=self.write2file)
            thread.start()
            self.threads.append(thread)
        if self.task_config.to_db:
            thread = Thread(target=self.write2db)
            thread.start()
            self.threads.append(thread)
        return ret

    def _split_kwargs(self, kws):
        """Convenience method to return the keywords for each data file

        Parameters
        ----------
        kws: dict
            A mapping whose values correspond to the shape of the
            :attr:`_datafile` attribute. I.e. if there are two data files,
            then all the values in `kws` must be lists of length 2. If the
            :attr:`_datafile` attribute is a string, then we don't expect any
            list

        Returns
        -------
        list of dict
            The splitted `kws`"""
        if isinstance(self._datafile, six.string_types):
            return [kws]
        return [{key: val[i] for key, val in kws.items()}
                for i in range(len(self._datafile))]

    def _get_data(self, i):
        """Get the data at position `i`

        Parameters
        ----------
        i: int
            The integer position where to set the data. If the
            :attr:`_datafile` attribute is a string, `i` will be ignored."""
        if isinstance(self._datafile, six.string_types):
            return self.data
        return self.data[i]

    def _set_data(self, data, i):
        """Set the  data at position `i`

        Parameters
        ----------
        data: pandas.DataFrame
            The dataframe to set
        i: int
            The integer position where to set the data. If the
            :attr:`_datafile` attribute is a string, `i` will be ignored."""
        if isinstance(self._datafile, six.string_types):
            self.data = data
        else:
            self.data[i] = data

    def init_from_file(self):
        """Initialize the task from already stored files"""
        pass

    def init_from_db(self):
        """Initialize the task from datatables already created"""
        pass

    def init_from_scratch(self):
        """Initialize the task from the configuration settings"""
        pass

    def setup_from_file(self, **kwargs):
        """Set up the task from already stored files"""
        kwargs = self._split_kwargs(kwargs)
        chunksize = self.global_config.get('chunksize', 10 ** 5)
        for i, datafile in enumerate(safe_list(self.datafile)):
            if not self.task_config.skip_filtering:
                data = []
                for all_data in pd.read_csv(datafile, chunksize=chunksize,
                                            **kwargs[i]):
                    if 'id' in all_data.columns:
                        all_data.set_index('id', inplace=True)
                    stations = list(self.stations)
                    if len(all_data.index.names) == 1:
                        data.append(all_data.loc(axis=0)[stations])
                    else:
                        names = all_data.index.names
                        axis = names.index('id')
                        key = [slice(None) for _ in range(axis)] + [
                            stations] + [
                                slice(None) for _ in range(
                                    axis, len(names) - 1)]
                        data.append(all_data.sort_index().loc(axis=0)[
                            tuple(key)])
                self._set_data(pd.concat(data), i)
            else:
                self._set_data(pd.read_csv(datafile, **kwargs[i]), i)

    def setup_from_db(self, **kwargs):
        """Set up the task from datatables already created"""
        kwargs = self._split_kwargs(kwargs)
        for i, dbname in enumerate(safe_list(self.dbname)):
            if self.task_config.skip_filtering:
                self._set_data(
                    pd.read_sql_query("SELECT * FROM %s" % (dbname, ),
                                      self.engine, **kwargs[i]),
                    i)
            else:
                self._set_data(pd.read_sql_query(
                    "SELECT * FROM %s WHERE id IN (%s)" % (
                        dbname, ', '.join(map("'{0}'".format, self.stations))),
                    self.engine, **kwargs[i]), i)

    @classmethod
    @docstrings.get_sectionsf('TaskBase.setup_from_instances')
    @docstrings.dedent
    def setup_from_instances(cls, base, instances, copy=False):
        """
        Combine multiple task instances into one instance

        Parameters
        ----------
        base: TaskBase
            The base task to use the configuration from
        instances: list of :class:`TaskBase`
            The tasks containing the data
        copy: bool
            If True, a copy of `base` is returned, otherwise `base` is
            modified inplace"""
        if copy:
            import copy
            obj = copy.copy(base)
        else:
            obj = base
        obj.logger.debug('Setting up from %i instances', len(instances))
        if isinstance(cls._datafile, six.string_types):
            data = pd.concat([ini.data for ini in instances])
        else:
            data = [pd.concat([ini.data[i] for ini in instances])
                    for i in range(len(cls._datafile))]
        obj.stations = np.concatenate(tuple(ini.stations for ini in instances))
        obj.data = data
        return obj

    @abc.abstractmethod
    def setup_from_scratch(self):
        """Setup the data from the configuration settings"""
        pass

    @docstrings.get_sectionsf('Parameterizer.run',
                              sections=['Parameters', 'Returns'])
    @docstrings.dedent
    def run(self, info, *args, **kwargs):
        """
        Run the task

        This method uses the data that has been setup through the :meth:`setup`
        method to process some configuration

        Parameters
        ----------
        dict
            The dictionary with the configuration settings for the namelist
        dict
            The dictionary holding additional meta information"""
        import xarray as xr
        self.logger.info('Calculating %s task', self.name)

        # ---- file names
        nc_output = self.task_config.nc_output
        plot_output = self.task_config.plot_output
        project_output = self.task_config.project_output
        info['nc_file'] = nc_output
        info['plot_file'] = plot_output
        info['project_file'] = project_output

        # ---- open dataset
        ds_orig = ds_list = self.ds
        if isinstance(ds_list, xr.Dataset):
            ds_list = [ds_list]

        # ---- create project
        inproject = self.task_config.project or project_output
        if not self.task_config.new_project and osp.exists(inproject):
            import psyplot.project as psy
            self.logger.debug('    Loading existing project %s', inproject)
            sp = psy.Project.load_project(inproject, datasets=ds_list)
        else:
            self.logger.debug('    Creating project...')
            sp = self.create_project(ds_orig)

        # ---- save data and project
        pdf = sp.export(plot_output, tight=True, close_pdf=False)
        if project_output:
            self.logger.debug('    Saving project to %s', project_output)
            if nc_output:
                for f in safe_list(nc_output):
                    if osp.exists(f):
                        os.remove(f)
                save_kws = dict(use_rel_paths=True, paths=safe_list(nc_output))
            else:  # save the entire dataset into the pickle file
                save_kws = dict(ds_description={'ds'})
            sp.save_project(project_output, **save_kws)
        # ---- make plots not covered by psyplot
        self.plot_additionals(pdf)

        # ---- configure the experiment
        self.make_run_config(sp, info, *args, **kwargs)

        # ---- export the figures
        self.logger.debug('    Saving plots to %s', plot_output)
        pdf.close()

        # ---- close the project
        if kwargs.get('close', True) or self.task_config.close:
            sp.close(True, True, True)
        self.logger.debug('Done.')

    @docstrings.get_sectionsf('TaskBase.create_project')
    @docstrings.dedent
    def create_project(self, ds):
        """
        To be reimplemented for each task with :attr:`has_run`

        Parameters
        ----------
        ds: xarray.Dataset
            The dataset to plot"""
        import psyplot.project as psy
        return psy.gcp()

    @docstrings.get_sectionsf('TaskBase.make_run_config')
    @docstrings.dedent
    def make_run_config(self, sp, info):
        """
        Method to be reimplemented for each task with :attr:`has_run`
        to manipulate the configuration

        Parameters
        ----------
        sp: psyplot.project.Project
            The project of the data
        info: dict
            The dictionary for saving additional information of the task"""
        return

    @docstrings.get_sectionsf('TaskBase.plot_additionals')
    @docstrings.dedent
    def plot_additionals(self, pdf):
        """
        Method to be reimplemented to make additional plots (if necessary)

        Parameters
        ----------
        pdf: matplotlib.backends.backend_pdf.PdfPages
            The PdfPages instance which can be used to save the figure
        """
        return

    @classmethod
    def _modify_parser(cls, parser):
        """
        Classmethod to modify the arguments of a command line parser

        Parameters
        ----------
        parser: gwgen.main.FuncArgParser
            The :class:`argparse.ArgumentParser` instance that holds the
            arguments from the :meth:`run` method

        Returns
        -------
        gwgen.main.FuncArgParser
            The above `parser`
        setup_grp
            The argument group for the setup method
        run_grp
            The argument group for the run method"""
        parser.setup_args(default_config)

        setup_grp = parser.add_argument_group(
            'Setup arguments', 'Arguments affecting the setup of the data')
        for arg in _SetupConfig._fields:
            parser.update_arg(arg, group=setup_grp)
        parser.update_arg('setup_from', short='f', long='from',
                          choices=['scratch', 'file', 'db'])
        if not cls.dbname:
            parser.pop_arg('to_db')
        if not cls._datafile:
            parser.pop_arg('to_csv')
            parser.pop_arg('remove')
        else:
            parser.update_arg('remove', short='rm')
        parser.update_arg('skip_filtering', short='sf')

        if not cls.has_run:
            run_grp = None
            for arg in _RunConfig._fields:
                parser.pop_arg(arg)
        else:
            run_grp = parser.add_argument_group(
                'Run arguments', 'Arguments for the experiment configuration')
            for arg in _RunConfig._fields:
                parser.update_arg(arg, group=run_grp)
            parser.update_arg('plot_output', short='o')
            parser.update_arg('nc_output', short='onc')
            parser.update_arg('project_output', short='op')
            parser.update_arg('new_project', short='np')
            # XXX Bug fix until docstrings.keep_params works right XXX
            parser.pop_key('project', 'action', None)
            # XXX
            parser.update_arg('project', short='p')
            parser.pop_arg('close')
        return parser, setup_grp, run_grp

    def get_run_kws(self, kwargs):
        return {key: val for key, val in kwargs.items()
                if key in inspect.getargspec(self.run)[0]}

    def write2db(self, **kwargs):
        """Write the data from this task to the database given by the
        :attr:`engine` attribute"""
        for i, (dbname, kws, dtype) in enumerate(zip(
                safe_list(self.dbname), self._split_kwargs(kwargs),
                self._split_kwargs(self.sql_dtypes))):
            data = self._get_data(i)
            if data is None or not len(data):
                continue
            if 'id' in data.columns:
                data = data.set_index('id')
            df_names = set(chain(data.columns, data.index.names))
            missing = df_names.difference(dtype)
            if missing:
                self.logger.warn('No data type was specified for %s', missing)
                dtype = None
            else:
                if len(df_names) != len(dtype):
                    dtype = {key: dtype[key] for key in df_names}
                kwargs.setdefault('dtype', dtype)
            lock = _db_locks.get(dbname)
            if lock:
                self.logger.debug('Acquiring lock...')
                lock.acquire()
            self.logger.info('Writing %s lines to data table %s', len(data),
                             dbname)
            kws.setdefault('if_exists', 'append')
            try:
                data.to_sql(dbname, self.engine, **kws)
            except:
                raise
            finally:
                if lock:
                    lock.release()
            self.logger.info('Done')

    def write2file(self, **kwargs):
        """Write the database to the :attr:`datafile` file"""
        for i, (datafile, kws) in enumerate(zip(safe_list(self.datafile),
                                                self._split_kwargs(kwargs))):
            data = self._get_data(i)
            if data is None or not len(data):
                continue
            lock = _file_locks.get(datafile)
            if lock:
                self.logger.debug('Acquiring lock...')
                lock.acquire()
            exists = osp.exists(datafile)
            self.logger.debug('Writing data to %sexisting file %s',
                              'not ' if not exists else '', datafile)
            try:
                safe_csv_append(data, datafile, **kws)
            except:
                raise
            finally:
                if lock:
                    self.logger.debug('Release lock')
                    lock.release()
            self.logger.debug('Done')

    @classmethod
    def get_manager(cls, *args, **kwargs):
        """
        Return a manager of this class that can be used to setup and organize
        tasks"""
        return TaskManager(cls, *args, **kwargs)


class TaskManager(object):
    """A manager to run the tasks within a task framework"""

    #: A subclass of the :class:`TaskBase` class whose
    #: :attr:`TaskBase._registry` attribute shall be used
    base_task = None

    docstrings.keep_params('TaskBase.parameters', 'stations', 'logger')

    _logger = None

    @property
    def logger(self):
        """The logger of this task"""
        if self._logger is None:
            self.logger = None
        return self._logger

    @logger.setter
    def logger(self, value):
        if isinstance(value, six.string_types):
            value = logging.getLogger(value)
        elif value is None and not self.config.get('serial'):
            import multiprocessing as mp
            value = logging.getLogger('.'.join(
                [__name__, self.__class__.__name__,
                 mp.current_process().name]))
        elif value is None:
            value = logging.getLogger('.'.join(
                [__name__, self.__class__.__name__]))
        self._logger = value

    @docstrings.dedent
    def __init__(self, base_task=TaskBase, tasks=None, config={}):
        """
        Parameters
        ----------
        base_task: TaskBase
            A subclass of the :class:`TaskBase` class whose tasks shall be
            used within this manager.
        tasks: list of :class:`TaskBase` instances
            The initialized tasks to use. If None, you need to call the
            :meth:`initialize_tasks` method
        config: dict
            The configuration of this manager containing information about the
            multiprocessing
        """
        self.base_task = base_task
        self.config = config
        self.tasks = tasks

    def __reduce__(self):
        return self.__class__, (self.base_task, self.tasks, self.config)

    @docstrings.get_sectionsf('TaskManager._get_tasks')
    @docstrings.dedent
    def _get_tasks(self, stations, task_kws={}):
        """
        Initaliaze the tasks

        This classmethod uses the :class:`TaskBase` framework to
        initialize the parameterization tasks

        Parameters
        ----------
        %(TaskBase.parameters.stations|logger)s
        task_kws: dict
            Keywords can be valid identifiers of the :class:`TaskBase`
            instances, dictionaries may be mappings for their
            :meth:`~TaskBase.setup` method

        Returns
        -------
        list
            A list of :class:`TaskBase` instances"""
        def init_task(task):
            kws = task_kws[task.name]
            config = kws.pop('config')
            project_config = kws.pop('project_config')
            return task(stations, config, project_config, self.config, **kws)
        tasks = {task.name: init_task(task) for task in map(
            self.get_task_cls, task_kws)}
        task_kws = task_kws.copy()
        # insert the requirements
        checked_requirements = False
        while not checked_requirements:
            checked_requirements = True
            for key, task in tasks.copy().items():
                if task.setup_from == 'scratch':
                    for req_task in self.get_requirements(key, False):
                        if req_task.name not in tasks:
                            checked_requirements = False
                            tasks[req_task.name] = req_task.from_task(task)
        # sort the tasks for their requirements
        sorted_tasks = list(self.sort_by_requirement(tasks.values()))
        for i, instance in enumerate(sorted_tasks):
            requirements = [ini for ini in sorted_tasks[:i]
                            if ini.name in instance.setup_requires]
            instance.set_requirements(requirements)
        return sorted_tasks

    @docstrings.dedent
    def initialize_tasks(self, stations, task_kws={}):
        """
        Initialize the setup of the tasks

        This classmethod uses the :class:`TaskBase` framework to
        initialize the setup on the I/O-processor

        Parameters
        ----------
        %(TaskManager._get_tasks.parameters)s"""
        task_kws = {key: val.copy() for key, val in task_kws.items()}
        stations = np.asarray(stations)
        if not stations.ndim:
            stations = stations.reshape(1)
        # sort the tasks for their requirements
        self.tasks = tasks = self._get_tasks(stations, task_kws)
        for instance in tasks:
            instance.init_task()

    def get_task(self, identifier):
        """Return the task corresponding in this manager of `identifier`

        Parameters
        ----------
        identifier: str
            The :attr:`name` attribute of the :class:`TaskBase` subclass

        Returns
        -------
        TaskBase
            The requested task"""
        try:
            return next(
                task for task in self.tasks if task.name == identifier)
        except StopIteration:
            raise KeyError(
                'Manager has no task {}. Possibilities: {}'.format(
                    identifier, ', '.join(t.name for t in self.tasks)))

    def get_task_cls(self, identifier):
        """Return the task class corresponding to the given `identifier`

        Parameters
        ----------
        identifier: str
            The :attr:`name` attribute of the :class:`TaskBase` subclass

        Returns
        -------
        TaskBase
            The class of the requested task"""
        try:
            return next(
                task_cls for task_cls in self.base_task._registry[::-1]
                if task_cls.name == identifier)
        except StopIteration:
            raise KeyError('Unkown task {}'.format(identifier))

    def get_requirements(self, identifier, all_requirements=True):
        """Return the required task classes for this task

        Parameters
        ----------
        identifier: str
            The :attr:`name` attribute of the :class:`Parameterizer` subclass
        all_requirements: bool
            If True, all requirements are searched recursively. Otherwise only
            the direct requirements are returned

        Returns
        -------
        list of :class:`Parameterizer`
            A list of Parameterizer subclasses that are required for the task
            of the given `identifier`"""
        def get_requirements(task_cls):
            for identifier in task_cls.setup_requires:
                req_cls = self.get_task_cls(identifier)
                ret.append(req_cls)
                if all_requirements:
                    get_requirements(req_cls)
        ret = []
        get_requirements(self.get_task_cls(identifier))
        return ret

    @staticmethod
    def sort_by_requirement(objects):
        """Sort the given tasks by their logical order

        Parameters
        ----------
        objects: list of :class:`TaskBase` subclasses or instances
            The objects to sort

        Returns
        -------
        list of :class:`TaskBase` subclasses or instances
            The same as `objects` but sorted"""
        def get_requirements(current):
            for name in list(remaining):
                if name in current.setup_requires and name in remaining:
                    get_requirements(remaining.pop(name))
            ret.append(current)
        remaining = {task.name: task for task in objects}
        ret = []
        while remaining:
            get_requirements(remaining.pop(next(iter(remaining))))
        return ret

    @docstrings.get_sectionsf('TaskManager.setup', sections=['Parameters',
                                                             'Returns'])
    def setup(self, stations, to_return=None):
        """
        Setup the data for the tasks in parallel or serial

        Parameters
        ----------
        stations: list of str
            The stations to process
        to_return: list of str
            The names of the tasks to return. If None, all tasks that have a
            run method will be returned
        """
        config = self.config
        stations = np.asarray(stations)
        if not stations.ndim:
            stations = stations.reshape(1)
        if to_return is None:
            to_return = [task.name for task in self.tasks if task.has_run]
        elif to_return == 'all':
            to_return = [task.name for task in self.tasks]
        else:
            to_return = safe_list(to_return)
        if not config.get('serial'):
            self.tasks = self._setup_parallel(stations, to_return)
        else:
            # serial processing
            self.tasks = self._setup(
                stations=stations, to_return=to_return)

    def _setup_parallel(self, stations, to_return=None):
        """
        Setup the data for the tasks in parallel

        Parameters
        ----------
        %(TaskManager.setup.parameters)s

        Returns
        -------
        list
            A list of :class:`Parameterizer` instances specified in `to_return`
            that hold the data
        """
        config = self.config
        logger = self.logger
        scheduler = config.get('scheduler')
        if scheduler is not None:
            from distributed import Client
            args = (scheduler, ) if scheduler else ()
            client = Client(*args)
            lock_dir = mkdtemp(prefix='tmp_gwgen_locks_')
            logger.debug('Temporary lock directory: %s', lock_dir)
        else:
            import multiprocessing as mp
        all_tasks = self.tasks
        grouped = [(key, list(tasks)) for key, tasks in groupby(
                       all_tasks, lambda task: task.setup_parallel)]
        ret_tasks = []
        orig_stations = stations
        for i, (key, tasks) in enumerate(grouped):
            self.tasks = tasks
            if key:
                logger.info('Processing %s tasks in parallel', len(tasks))
                # initialize pool
                nprocs = config.get('nprocs', 'all')
                if nprocs == 'all':
                    if scheduler is not None:
                        nprocs = len(client.ncores().values())
                    else:
                        nprocs = mp.cpu_count()
                # split up the stations for the workers
                max_stations = min(int(np.ceil(len(orig_stations) / nprocs)),
                                   config.get('max_stations', 500))
                if len(orig_stations) > max_stations:
                    stations = np.split(orig_stations, np.arange(
                        max_stations, len(stations), max_stations, dtype=int))
                else:
                    stations = [orig_stations]
                # The real number of stations list. It might happen that we
                # have more processors than stations which then results in
                # empty arrays in `stations`
                nstations_lists = next((i for i, l in enumerate(stations)
                                        if len(l) == 0), len(stations))
                # make sure we don't send a list of empty stations to a process
                stations = stations[:nstations_lists]
                nprocs = min(nprocs, nstations_lists)
                if scheduler is None:
                    # create locks
                    for task in self.tasks:
                        for fname in safe_list(task.datafile):
                            _file_locks[fname] = mp.Lock()
                        for dbname in safe_list(task.dbname):
                            _db_locks[dbname] = mp.Lock()
                    # start the pool
                    logger.debug(
                        'Starting %s processes for %s station lists',
                        nprocs, len(stations))
                    pool = mp.Pool(nprocs, initializer=init_locks,
                                   initargs=(_db_locks, _file_locks))
                else:
                    file_locks = list(chain(*(
                        safe_list(task.datafile) for task in self.tasks)))
                    db_locks = list(chain(*(
                        safe_list(task.datafile) for task in self.tasks)))
                if i != len(grouped):
                    unsafe = list(chain(*grouped[i+1::2]))
                    _to_return = to_return + list(chain(*(
                        t.setup_requires for t in chain(*unsafe[1::2]))))
                else:
                    _to_return = to_return
                args = [[s, _to_return, True] for s in stations]
                # start the computation
                if scheduler is not None:
                    try:
                        kws = {'workers': set(client.cluster.workers[:nprocs])}
                    except AttributeError:
                        kws = {}
                    for proc_args in args:
                        proc_args.extend([file_locks, db_locks, lock_dir])
                    res = client.map(self, args, pure=False, **kws)
                    tasks = client.gather(res)
                else:
                    res = pool.map_async(self, args)
                    tasks = res.get()
                    pool.close()
                    pool.join()
                    pool.terminate()
                tasks = [
                    task.setup_from_instances(
                        next(t for t in all_tasks if t.name == task.name),
                        [proc_tasks[i] for proc_tasks in tasks])
                    for i, task in enumerate(tasks[0])]
                ret_tasks.extend(tasks)
            else:
                logger.info('Processing %s tasks in serial', len(tasks))
                ret_tasks.extend(self._setup(
                        stations=orig_stations, to_return=to_return))
        _db_locks.clear()
        _file_locks.clear()
        if scheduler is not None:
            shutil.rmtree(lock_dir)
            client.shutdown()
        return [task for task in ret_tasks if task.name in to_return]

    def __call__(self, t):
        return self._setup(*t)

    docstrings.keep_params('TaskBase.parameters', 'stations')

    @docstrings.dedent
    def _setup(self, stations, to_return=None, copy_tasks=False,
               file_locks=None, db_locks=None, lock_dir=None):
        """
        Process the given stations

        This classmethod uses the :class:`TaskBase` framework to run the
        task of the gwgen model

        Parameters
        ----------
        %(TaskManager.setup.parameters)s
        copy_tasks: bool
            If True, we will create a copy of the tasks in this manager before
            setting up the data. This avoids conflicts during parallel
            processing

        Returns
        -------
        %(TaskManager.setup.returns)s"""
        # sort the tasks for their requirements
        if file_locks is not None:
            init_interprocess_locks(db_locks=db_locks, file_locks=file_locks,
                                    lock_dir=lock_dir)
        if to_return is None:
            to_return = [ini.name for ini in self.tasks if ini.has_run]
        for instance in self.tasks:
            instance.stations = stations
            # tasks might filter their stations (e.g. the cloud
            # parameterization) therefore we check again and skip the instance
            # if necessary
            if len(instance.stations):
                instance.setup()
            else:
                self.logger.debug(
                    'Skipping %s task because it contains no stations!',
                    instance.name)
        for task in self.tasks:
            for thread in task.threads:
                thread.join()
        ret = list(filter(lambda ini: ini.name in to_return, self.tasks))
        if copy_tasks:
            # copy the instance in order to avoid complications with
            # parallel processing
            for i, task in enumerate(ret[:]):
                ret[i] = copy.copy(task)
            for task in filter(lambda ini: ini.name not in to_return,
                               self.tasks):
                del task.data
        return ret

    def run(self, full_info, *args):
        for task in self.tasks:
            if task.has_run:
                full_info[task.name] = info = OrderedDict()
                task.run(info, *args)
