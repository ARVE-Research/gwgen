# -*- coding: utf-8 -*-
"""Test module for the parameterization. Note that the entire task framework is
very sensible to internal errors , so if anything goes wrong, there should be
an error within the setup"""
import unittest
import os
import os.path as osp
import shutil
import numpy as np
import pandas as pd
# we use assert_frame_equal instead of pd.equal, because it is less strict
from pandas.util.testing import assert_frame_equal
import _base_testing as bt
import gwgen.parameterization as param
import gwgen.utils as utils
from psyplot.config.rcsetup import safe_list


#: Message to provide a bit more information on the two data frames
df_diff_msg = (
    'Task data and reference differ!\n{0}\n'
    'Task\n%s\n{0}\n'
    'Reference\n%s{0}').format('-' * 80)


def df_equals(df, df_ref, *args, **kwargs):
    """Simple wrapper around assert_frame_equal to use unittests assertion

    Parameters
    ----------
    df: pd.DataFrame
        The simulation data frame
    df_ref: pd.DataFrame
        The reference data frame

    Returns
    -------
    None or Exception
        Either None if everything went fine, otherwise the raised Exception"""
    try:
        assert_frame_equal(df, df_ref, *args, **kwargs)
    except Exception as e:
        return e


class _ParameterizerTestMixin(object):
    """A mixing for parameterizer tests"""

    param_cls = None

    @property
    def name(self):
        return self.param_cls.name

    def test_param(self, check_index_duplicates=True,
                   check_data_duplicates=False, to_csv=True, **kwargs):
        def get_data(task):
            if not isinstance(task.data, pd.DataFrame):
                return task.data
            return [task.data]

        def no_duplicates(df):
            return df.ix[~df.index.duplicated(keep=False)]

        if self.param_cls is None:
            return
        to_db = self.use_db
        name = self.name
        self._test_init()
        kwargs.setdefault(name, {})
        kwargs.setdefault('to_return', [name])
        manager = self.organizer.param(
            stations=self.stations_file, to_csv=to_csv, to_db=to_db,
            **kwargs)
        task = manager.get_task(name)
        ref_data = get_data(task)

        # check if run worked
        if task.has_run:
            self.assertEqual(
                set(task.namelist_keys),
                set(self.organizer.exp_config['namelist']['weathergen_ctl']))

        setup_from_file = False
        for fname in filter(None, safe_list(task.datafile)):
            setup_from_file = True
            self.assertTrue(osp.exists(fname),
                            msg='Datafile %s of %s task does not exist!' % (
                                fname, name))
        if check_index_duplicates:
            for df in get_data(task):
                idx = df.index
                self.assertFalse(
                    idx.has_duplicates,
                    msg='%s task index data has duplicates!\n%s' % (
                        name, idx.values[idx.duplicated(keep=False)]))
        if check_data_duplicates:
            self.assertFalse(task.data.duplicated().any(),
                             msg='%s task data has duplicates!\n%s' % (
                                name, task.data[
                                    task.data.duplicated(keep=False)]))
        engine = task.engine
        setup_from_db = False
        if engine is not None:
            sql_dtypes = task._split_kwargs(task.sql_dtypes)
            for i, table in enumerate(safe_list(task.dbname)):
                if table is not None:
                    setup_from_db = True
                    self.assertTrue(
                        engine.has_table(table),
                        msg='Database has no table %s of %s task' % (table,
                                                                     name))
                    data = task._get_data(i)
                    data_cols = set(data.columns) | set(data.index.names)
                    self.assertEqual(set(sql_dtypes[i]) & data_cols,
                                     data_cols,
                                     msg='Missing sql dtype for %s' % name)

        # check setup from file
        if setup_from_file:
            manager = self.organizer.param(
                stations=self.stations_file, **kwargs)
            new_task = manager.get_task(name)
            self.assertEqual(new_task.setup_from, 'file',
                             msg='setup_from of %s task should be "file"!' % (
                                name))
            new_data = get_data(new_task)
            self.assertEqual(len(new_data), len(ref_data),
                             msg=('Number of dataframes for %s task are not '
                                  'equal after setup from file!') % name)
            for df, df_ref in zip(new_data, ref_data):
                df.sort_index(inplace=True)
                df_ref.sort_index(inplace=True)
                df = no_duplicates(df)
                df_ref = no_duplicates(df_ref)
                mask = (df != df_ref).values.any(axis=1)
                self.assertIsNone(df_equals(df, df_ref, check_dtype=False),
                                  msg=df_diff_msg % (df.ix[mask],
                                                     df_ref.ix[mask]))
        # check setup from db
        if setup_from_db:
            for fname in filter(None, safe_list(task.datafile)):
                os.remove(fname)
            manager = self.organizer.param(
                stations=self.stations_file, **kwargs)
            new_task = manager.get_task(name)
            self.assertEqual(new_task.setup_from, 'db',
                             msg='setup_from of %s task should be "db"!' % (
                                 name))
            new_data = get_data(new_task)
            self.assertEqual(len(new_data), len(ref_data),
                             msg=('Number of dataframes for %s task are not '
                                  'equal after setup from db!') % name)
            for df, df_ref in zip(new_data, ref_data):
                df.sort_index(inplace=True)
                df_ref.sort_index(inplace=True)
                df = no_duplicates(df)
                df_ref = no_duplicates(df_ref)
                mask = (df != df_ref).values.any(axis=1)
                self.assertIsNone(df_equals(df, df_ref, check_dtype=False),
                                  msg=df_diff_msg % (df.ix[mask],
                                                     df_ref.ix[mask]))
        return manager

# the usage of distributed is not supported at the moment
#    def test_param_distributed(self, *args, **kwargs):
#        """Test the parameterization with the distributed package"""
#        if not self.organizer.global_config.get('serial'):
#            from distributed import LocalCluster
#            import multiprocessing as mp
#            c = LocalCluster(n_workers=mp.cpu_count(), diagnostics_port=None)
#            self.organizer.global_config['scheduler'] = c.scheduler_address
#            ret = self.test_param(*args, **kwargs)
#            c.close()
#            return ret


class Test_DailyGHCNData(bt.BaseTest, _ParameterizerTestMixin):
    """Test case for the :class:`gwgen.parameterization.DailyGHCNData` task"""

    param_cls = param.DailyGHCNData

    stations_type = 'short'

    @unittest.skipIf(not bt.online, 'Only works with internet connection')
    def test_full_source_exists(self):
        """Test whether the url of the tar ball can be downloaded"""
        src = param.DailyGHCNData.http_source
        self._test_url(src)
        self.assert_(True)

    @unittest.skipIf(not bt.online, 'Only works with internet connection')
    def test_single_source_exists(self):
        """Test whether the url of the tar ball can be downloaded"""
        src = param.DailyGHCNData.http_single
        self._test_url(src.format(self.stations[0]) + '.dly')
        self.assert_(True)


class Test_CompleteDailyGHCNData(bt.BaseTest, _ParameterizerTestMixin):
    """Test case for the :class:`gwgen.parameterization.CompleteDailyGHCNData`
    task"""

    param_cls = param.CompleteDailyGHCNData

    stations_type = 'short'

    def test_param(self, **kwargs):
        manager = super(Test_CompleteDailyGHCNData, self).test_param(**kwargs)
        data = manager.get_task(self.name).data
        monthly = data.groupby(level=['id', 'year', 'month']).agg(len)
        monthly['day'] = 1
        s = pd.to_datetime(monthly.reset_index()[['year', 'month', 'day']])
        s.index = monthly.index
        monthly['ndays'] = ((s + pd.datetools.thisMonthEnd) - s).dt.days + 1
        difference = monthly['prcp'] != monthly['ndays']
        self.assertTrue(
            (monthly['prcp'] == monthly['ndays']).all(),
            msg='Prcp and number of days in month differ! %s' % (
                monthly[['prcp', 'ndays']][difference]))


class Test_YearlyCompleteDailyGHCNData(bt.BaseTest, _ParameterizerTestMixin):
    """Test case for the :class:`gwgen.parameterization.CompleteDailyGHCNData`
    task"""

    param_cls = param.YearlyCompleteDailyGHCNData

    stations_type = 'short'

    def test_param(self, **kwargs):
        import calendar
        manager = super(Test_YearlyCompleteDailyGHCNData, self).test_param(
            **kwargs)
        data = manager.get_task(self.name).data
        yearly = data.groupby(level=['id', 'year']).agg(len)
        years = yearly.index.get_level_values('year')
        ndays = 365 + np.vectorize(calendar.isleap)(years).astype(int)
        yearly['ndays'] = ndays
        difference = yearly['prcp'].values != ndays
        self.assertTrue(
            (yearly['prcp'].values == ndays).all(),
            msg='Prcp and number of days in month differ! %s' % (
                yearly[['prcp', 'ndays']].iloc[difference]))


class Test_MonthlyGHCNData(bt.BaseTest, _ParameterizerTestMixin):
    """Test case for the :class:`gwgen.parameterization.MonthlyGHCNData` task
    """

    param_cls = param.MonthlyGHCNData

    stations_type = 'short'


class Test_CompleteMonthlyGHCNData(bt.BaseTest, _ParameterizerTestMixin):
    """Test case for the
    :class:`gwgen.parameterization.CompleteMonthlyGHCNData` task"""

    param_cls = param.CompleteMonthlyGHCNData

    stations_type = 'short'


class Test_PrcpDistParams(bt.BaseTest, _ParameterizerTestMixin):
    """Test case for the :class:`gwgen.parameterization.PrcpDistParams` task"""

    param_cls = param.PrcpDistParams


class Test_TemperatureParameterizer(bt.BaseTest, _ParameterizerTestMixin):
    """Test case for the
    :class:`gwgen.parameterization.TemperatureParameterizer` task"""

    param_cls = param.TemperatureParameterizer

    def test_param(self, **kwargs):
        return super(Test_TemperatureParameterizer, self).test_param(
            temp=dict(cutoff=0,
                      tmin_range1=(-40, -5), tmin_range2=(5, 20),
                      tmax_range1=(-40, -5), tmax_range2=(5, 20)), **kwargs)


class _CloudTestMixin(object):
    """A base class defining a test for using eecra stations directly"""

    def test_param_eecra(self, **kwargs):
        self.stations_file = self.eecra_stations_file
        kwargs.setdefault(self.param_cls.name, {}).setdefault(
            'args_type', 'eecra')
        self.test_param(**kwargs)


class Test_HourlyCloud(bt.BaseTest, _ParameterizerTestMixin, _CloudTestMixin):
    """Test case for the :class:`gwgen.parameterization.HourlyCloud` task"""

    param_cls = param.HourlyCloud

    stations_type = 'short'

    @unittest.skipIf(not bt.online, 'Only works with internet connection')
    def test_source_url(self):
        """Test whether the urls of the EECRA database work"""
        for key, url in self.param_cls.urls.items():
            self._test_url(url)

    def test_extraction(self):
        self._test_init()
        orig_data = self.organizer.project_config['data']
        self.organizer.project_config['data'] = self.test_dir
        eecra_dir = osp.join(self.test_dir, 'eecra')
        os.makedirs(eecra_dir)
        shutil.copytree(osp.join(orig_data, 'eecra', 'raw'),
                        osp.join(eecra_dir, 'raw'))
        task = param.HourlyCloud.from_organizer(self.organizer, self.stations)
        task.years = [1996, 1997]  # the years to use
        task.months = [1, 2]   # JAN, FEB
        task.init_from_scratch()
        for station in task.eecra_stations:
            fname = osp.join(eecra_dir, 'stations', str(station) + '.csv')
            self.assertTrue(osp.exists(fname),
                            msg='Missing file %s!')
            self.assertGreater(utils.file_len(fname), 1,
                               msg='%s seems to be empty!' % fname)

    def test_param(self, **kwargs):
        """Unfortunately the raw EECRA data contains duplicates, so we don't
        check for them"""
        kwargs['check_index_duplicates'] = False
        super(Test_HourlyCloud, self).test_param(**kwargs)


class Test_DailyCloud(bt.BaseTest, _ParameterizerTestMixin, _CloudTestMixin):
    """Test case for the :class:`gwgen.parameterization.DailyCloud` task"""

    param_cls = param.DailyCloud

    stations_type = 'short'


class Test_MonthlyCloud(bt.BaseTest, _ParameterizerTestMixin, _CloudTestMixin):
    """Test case for the :class:`gwgen.parameterization.MonthlyCloud` task"""

    param_cls = param.MonthlyCloud

    stations_type = 'short'


class Test_CompleteMonthlyCloud(bt.BaseTest, _ParameterizerTestMixin,
                                _CloudTestMixin):
    """Test case for the :class:`gwgen.parameterization.CompleteMonthlyCloud`
    task"""

    param_cls = param.CompleteMonthlyCloud

    stations_type = 'short'


class Test_YearlyCompleteMonthlyCloud(bt.BaseTest, _ParameterizerTestMixin,
                                      _CloudTestMixin):
    """Test case for the
    :class:`gwgen.parameterization.YearlyCompleteMonthlyCloud` task"""

    param_cls = param.YearlyCompleteMonthlyCloud

    stations_type = 'short'


class Test_CompleteDailyCloud(bt.BaseTest, _ParameterizerTestMixin,
                              _CloudTestMixin):
    """Test case for the :class:`gwgen.parameterization.CompleteDailyCloud`
    task"""

    param_cls = param.CompleteDailyCloud

    stations_type = 'short'

    def test_param(self, **kwargs):
        manager = super(Test_CompleteDailyCloud, self).test_param(**kwargs)
        # test whether all months are complete
        data = manager.get_task(self.name).data
        monthly = data.groupby(level=['id', 'year', 'month']).agg(len)
        monthly['day'] = 1
        s = pd.to_datetime(monthly.reset_index()[['year', 'month', 'day']])
        s.index = monthly.index
        monthly['ndays'] = ((s + pd.datetools.thisMonthEnd) - s).dt.days + 1
        difference = monthly['mean_cloud'] != monthly['ndays']
        self.assertTrue(
            (monthly['mean_cloud'] == monthly['ndays']).all(),
            msg='mean_cloud and number of days in month differ! %s' % (
                monthly[['mean_cloud', 'ndays']][difference]))


class Test_YearlyCompleteDailyCloud(bt.BaseTest, _ParameterizerTestMixin,
                                    _CloudTestMixin):
    """Test case for the
    :class:`gwgen.parameterization.YearlyCompleteDailyCloud` task"""

    param_cls = param.YearlyCompleteDailyCloud

    stations_type = 'short'

    def test_param(self, **kwargs):
        import calendar
        manager = super(Test_YearlyCompleteDailyCloud, self).test_param(
            **kwargs)
        data = manager.get_task(self.name).data
        yearly = data.groupby(level=['id', 'year']).agg(len)
        years = yearly.index.get_level_values('year')
        ndays = 365 + np.vectorize(calendar.isleap)(years).astype(int)
        yearly['ndays'] = ndays
        difference = yearly['mean_cloud'].values != ndays
        self.assertTrue(
            (yearly['mean_cloud'].values == ndays).all(),
            msg='mean_cloud and number of days in month differ! %s' % (
                yearly[['mean_cloud', 'ndays']].iloc[difference]))


class Test_CloudParameterizer(bt.BaseTest, _ParameterizerTestMixin,
                              _CloudTestMixin):
    """Test case for the :class:`gwgen.parameterization.CloudParameterizer`
    task"""

    param_cls = param.CloudParameterizer


class Test_CompleteMonthlyWind(bt.BaseTest, _ParameterizerTestMixin,
                               _CloudTestMixin):
    """Test case for the :class:`gwgen.parameterization.CompleteMonthlyWind`
    task"""

    param_cls = param.CompleteMonthlyWind

    stations_type = 'short'


class Test_YearlyCompleteMonthlyWind(bt.BaseTest, _ParameterizerTestMixin,
                                     _CloudTestMixin):
    """Test case for the
    :class:`gwgen.parameterization.YearlyCompleteMonthlyWind` task"""

    param_cls = param.YearlyCompleteMonthlyWind

    stations_type = 'short'


class Test_WindParameterizer(bt.BaseTest, _ParameterizerTestMixin,
                             _CloudTestMixin):
    """Test case for the :class:`gwgen.parameterization.WindParameterizer`
    task"""

    param_cls = param.WindParameterizer


class Test_CrossCorrelation(bt.BaseTest, _ParameterizerTestMixin):
    """Test case for the
    :class:`gwgen.parameterization.CrossCorrelation` task"""

    param_cls = param.CrossCorrelation


if __name__ == '__main__':
    unittest.main()
