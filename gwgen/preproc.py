# -*- coding: utf-8 -*-
"""Additional routines for preprocessing"""
import tempfile
import os.path as osp
from collections import namedtuple
import numpy as np
import pandas as pd
from psyplot.compat.pycompat import OrderedDict
import gwgen.utils as utils
from gwgen.utils import docstrings


class CloudPreproc(utils.TaskBase):

    @property
    def task_data_dir(self):
        return osp.join(self.data_dir, 'eecra')

    _registry = []


class CloudInventory(CloudPreproc):
    """A task for computing the EECRA inventory for each station"""

    name = 'eecra_inventory'

    summary = 'Compute the inventory of the EECRA stations'

    http_xstall = 'http://cdiac.ornl.gov/ftp/ndp026c/XSTALL'

    _datafile = 'eecra_inventory.csv'

    dbname = 'eecra_inventory'

    has_run = True

    @property
    def setup_parallel(self):
        return self.setup_from == 'scratch'

    @property
    def default_config(self):
        return default_cloud_inventory_config()._replace(
            **super(CloudInventory, self).default_config._asdict())

    @property
    def xstall_df(self):
        """The dataframe corresponding to the XSTALL stations"""
        use_xstall = self.task_config.xstall
        if utils.isstring(use_xstall):
            fname = self.task_config.no_xstall
        else:
            fname = tempfile.NamedTemporaryFile().name
            utils.download_file(self.http_xstall, fname)
        arr = np.loadtxt(fname, usecols=[1, 2, 3])
        df = pd.DataFrame(arr, columns=['station_id', 'lat', 'lon'])
        df['station_id'] = df.station_id.astype(int)
        df.set_index('station_id', inplace=True)
        return df

    @classmethod
    def _modify_parser(cls, parser):
        parser.setup_args(default_cloud_inventory_config)
        cls.has_run = False
        parser, setup_grp, run_grp = super(CloudInventory, cls)._modify_parser(
            parser)
        parser.update_arg('xstall', group=setup_grp)
        cls.has_run = True
        return parser, setup_grp, run_grp

    def __init__(self, *args, **kwargs):
        super(CloudInventory, self).__init__(*args, **kwargs)
        self.__setup = False

    def setup(self, *args, **kwargs):
        self.__setup = True
        super(CloudInventory, self).setup(*args, **kwargs)

    def init_from_scratch(self):
        from gwgen.parameterization import HourlyCloud
        task = HourlyCloud.from_task(self)
        task.download_src(task.raw_dir)  # make sure the source files exist

    def setup_from_file(self, **kwargs):
        """Set up the task from already stored files (and avoid locating the
        stations of this task)"""
        kwargs = self._split_kwargs(kwargs)
        for i, datafile in enumerate(utils.safe_list(self.datafile)):
            self._set_data(pd.read_csv(datafile, **kwargs[i]), i)

    def setup_from_db(self, **kwargs):
        """Set up the task from datatables already created (and avoid locating
        the stations of this task)"""
        kwargs = self._split_kwargs(kwargs)
        for i, dbname in enumerate(utils.safe_list(self.dbname)):
            self._set_data(pd.read_sql_query(
                "SELECT * FROM %s" % (self.dbname),
                self.engine, **kwargs[i]), i)

    def setup_from_scratch(self):
        from gwgen.parse_eecra import parse_file

        def compute(fname):
            g = parse_file(fname).groupby('station_id')[
                ['lat', 'lon', 'year']]
            df = g.mean()
            df['counts'] = g.size()
            std = g.std()
            df['lon_std'] = std.lon
            df['lat_std'] = std.lat
            return df

        self.data = pd.concat(list(map(compute, self.stations)))

    def write2db(self, *args, **kwargs):
        if self.__setup:
            return
        kwargs.setdefault('if_exists', 'replace')
        super(CloudInventory, self).write2db(*args, **kwargs)

    def write2file(self, *args, **kwargs):
        if self.__setup:
            return
        super(CloudInventory, self).write2file(*args, **kwargs)

    def run(self, info):

        self.__setup = False

        if self.setup_from == 'scratch':
            df = self.data
            # we may use a parallel setup which requires a weighted average
            g = df.groupby(level='station_id')
            total_counts = g.counts.transform("sum")
            df['lat'] = df.counts / total_counts * df.lat
            df['lon'] = df.counts / total_counts * df.lon
            df['lat_std'] = (df.counts / total_counts) * df.lat_std ** 2
            df['lon_std'] = (df.counts / total_counts) * df.lon_std ** 2
            eecra = g.agg(OrderedDict([
                    ('lat', 'sum'), ('lon', 'sum'), ('lat_std', 'sum'),
                    ('lon_std', 'sum'),
                    ('year', ('min', 'max')), ('counts', 'sum')]))
            eecra.columns = ['lat', 'lon', 'lat_std', 'lon_std',
                             'firstyear', 'lastyear', 'counts']
            eecra[['lat_std', 'lon_std']] **= 0.5

            use_xstall = self.task_config.xstall

            if use_xstall:
                to_replace = self.xstall_df
                # keep only matching stations
                to_replace = to_replace.join(eecra[[]], how='inner')
                eecra.loc[to_replace.index, ['lat', 'lon']] = to_replace
            self.data = eecra

        if self.task_config.to_csv:
            self.write2file()
        if self.task_config.to_db:
            self.write2db()


class CloudGHCNMap(CloudPreproc):
    """A task for computing the EECRA inventory for each station"""

    name = 'eecra_ghcn_map'

    setup_requires = ['eecra_inventory']

    summary = 'Compute the inventory of the EECRA stations'

    _datafile = 'eecra_ghcn_map.csv'

    dbname = 'eecra_ghcn_map'

    has_run = True

    @property
    def default_config(self):
        return default_cloud_ghcn_map_config()._replace(
            **super(CloudGHCNMap, self).default_config._asdict())

    @classmethod
    def _modify_parser(cls, parser):
        parser.setup_args(default_cloud_ghcn_map_config)
        cls.has_run = False
        parser, setup_grp, run_grp = super(CloudGHCNMap, cls)._modify_parser(
            parser)
        parser.update_arg('max_distance', group=setup_grp, short='md')
        parser.pop_arg('to_db')
        parser.pop_arg('setup_from')
        cls.has_run = True
        return parser, setup_grp, run_grp

    def __init__(self, *args, **kwargs):
        super(CloudGHCNMap, self).__init__(*args, **kwargs)
        self.task_config = self.task_config._replace(
            setup_from='scratch', to_db=False)
        self.__setup = False

    def setup(self, *args, **kwargs):
        self.__setup = True
        super(CloudGHCNMap, self).setup(*args, **kwargs)

    def init_from_scratch(self):
        from gwgen.parameterization import HourlyCloud
        task = HourlyCloud.from_task(self)
        task.download_src(task.raw_dir)  # make sure the source files exist

    def setup_from_scratch(self):
        """Does nothing but initializing an empty data frame. The real work is
        done in the :meth:`run` method"""
        self.data = pd.DataFrame([], columns=['station_id', 'dist'],
                                 index=pd.Index([], name='id'))

    def write2db(self, *args, **kwargs):
        raise NotImplementedError(
            'The data is always written to the database!')

    def write2file(self, *args, **kwargs):
        if self.__setup:
            return
        super(CloudGHCNMap, self).write2file(*args, **kwargs)

    def run(self, info):
        def create_geog(table):
            conn.execute(
                'ALTER TABLE %s ADD COLUMN geog geography(POINT,4326) ;' % (
                    table))
            conn.execute("""
                UPDATE %s
                    SET geog = ST_SetSRID(ST_MakePoint(lon,lat),4326);""" % (
                    table))
            conn.execute(
                'CREATE INDEX {0}_geog ON {0} USING GIST (geog);'.format(table)
                )
        from gwgen.evaluation import EvaluationPreparation
        self.__setup = False

        inv = self.eecra_inventory

        if not self.engine.has_table(inv.dbname):
            inv.write2db()
        conn = self.engine.connect()
        if 'geog' not in pd.read_sql_query('SELECT * FROM %s LIMIT 0' % (
                inv.name), conn).columns:
            create_geog(inv.dbname)

        t = EvaluationPreparation.from_task(self)
        # download inventory
        t.download_src()
        ghcn = t.station_list
        ghcn = ghcn.ix[ghcn.vname == 'PRCP'].set_index('id')
        ghcn.to_sql('ghcn_inventory', self.engine, if_exists='replace')
        create_geog('ghcn_inventory')

        conn.execute("DROP TABLE IF EXISTS eecra_ghcn_map;")

        conn.execute("""
            CREATE TABLE eecra_ghcn_map AS (
                SELECT DISTINCT ON (id) id, station_id, dist FROM (
                    SELECT DISTINCT ON (a.station_id)
                        b.id, a.station_id, ST_Distance(a.geog, b.geog) AS dist
                    FROM eecra_inventory a
                        INNER JOIN ghcn_inventory b ON ST_DWithin(
                            a.geog, b.geog, 1000)
                    ORDER BY a.station_id, ST_Distance(a.geog, b.geog)) foo
                ORDER BY id, dist);""")

        self.data = pd.read_sql('eecra_ghcn_map', self.engine, index_col='id')
        conn.close()

        if self.task_config.to_csv:
            self.write2file()


CloudInventoryConfig = namedtuple(
    'CloudInventoryConfig', ['xstall'] + list(utils.TaskConfig._fields))


# to_db is set to True by default because it is required
docstrings.delete_params('TaskConfig.parameters', 'to_db', 'setup_from')


CloudInventoryConfig = utils.append_doc(
    CloudInventoryConfig, docstrings.get_sections(docstrings.dedents("""
    Parameters
    ----------
    xstall: bool or str
        If True (default), download the XSTALL file from %s.
        This file contains some estimates of station longitude and latitude.
        If ``False`` or empty string, the file is not used, otherwise, if set
        with a string, it is interpreted as the path to the local file
    %%(TaskConfig.parameters.no_to_db|setup_from)s
    """ % CloudInventory.http_xstall), 'CloudInventoryConfig'))


@docstrings.dedent
def default_cloud_inventory_config(xstall=True, *args, **kwargs):
    """
    Default config for :class:`CloudInventory`

    Parameters
    ----------
    %(CloudInventoryConfig.parameters)s"""
    return CloudInventoryConfig(
        xstall, *utils.default_config(*args, **kwargs))


CloudGHCNMapConfig = namedtuple(
    'CloudGHCNMapConfig', ['max_distance'] + list(utils.TaskConfig._fields))


# to_db is set to True by default because it is required
docstrings.delete_params('TaskConfig.parameters', 'to_db', 'setup_from')


CloudGHCNMapConfig = utils.append_doc(
    CloudGHCNMapConfig, docstrings.get_sections(docstrings.dedents("""
    Parameters
    ----------
    max_distance: float
        The maximum distance in meters for which we consider two stations as
        equal (Default: 1000m)
    %(TaskConfig.parameters.no_to_db|setup_from)s
    """), 'CloudGHCNMapConfig'))


@docstrings.dedent
def default_cloud_ghcn_map_config(max_distance=1000., *args, **kwargs):
    """
    Default config for :class:`CloudGHCNMap`

    Parameters
    ----------
    %(CloudGHCNMapConfig.parameters)s"""
    return CloudGHCNMapConfig(
        max_distance, *utils.default_config(*args, **kwargs))
