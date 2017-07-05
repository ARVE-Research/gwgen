import os
import os.path as osp
import six
import shutil
import unittest
import glob
import gwgen
import numpy as np
import tempfile
from gwgen.main import GWGENOrganizer
import gwgen.utils as utils
from model_organization.config import setup_logging

test_root = osp.abspath(osp.dirname(__file__))

_test_stations = osp.join(test_root, 'test_stations.dat')

_short_test_stations = osp.join(test_root, 'test_stations_short.dat')

_long_test_stations = osp.join(test_root, 'test_stations_long.dat')

_eecra_test_stations = osp.join(test_root, 'eecra_test_stations.dat')

_short_eecra_test_stations = osp.join(test_root,
                                      'eecra_test_stations_short.dat')


setup_logging(osp.join(test_root, 'logging.yaml'))

on_travis = os.getenv('TRAVIS')


db_config = dict(
    database='travis_ci_test')


class BaseTest(unittest.TestCase):
    """Test the :class:`gwgen.main.ModuleOrganizer` class"""

    test_dir = None

    remove_at_cleanup = True

    use_db = True

    stations_type = 'normal'

    _station_files = {
        'normal': _test_stations,
        'short': _short_test_stations,
        'long': _long_test_stations,
        }

    _eecra_station_files = {
        'normal': _eecra_test_stations,
        'short': _short_eecra_test_stations,
        }

    @classmethod
    def setUpClass(cls):
        # try to connect to a postgres database
        if cls.use_db:
            try:
                utils.get_postgres_engine(create=True, test=True, **db_config)
                cls.use_db = True
            except:
                cls.use_db = False

    def setUp(self):
        from psyplot import rcParams
        self.test_dir = tempfile.mkdtemp(prefix='tmp_gwgentest')
        os.environ['GWGENCONFIGDIR'] = self.config_dir = osp.join(
            self.test_dir, 'config')
        if not osp.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not osp.exists(self.config_dir):
            os.makedirs(self.config_dir)
        self.test_db = osp.basename(self.test_dir)
        src_station_file = self._station_files[self.stations_type]
        self.stations_file = osp.join(self.test_dir,
                                      osp.basename(src_station_file))
        shutil.copyfile(src_station_file, self.stations_file)

        src_eecra_file = self._eecra_station_files[self.stations_type]
        self.eecra_stations_file = osp.join(self.test_dir,
                                            osp.basename(src_eecra_file))
        shutil.copyfile(src_eecra_file, self.eecra_stations_file)
        self.organizer = GWGENOrganizer()
        global_conf = self.organizer.config.global_config
        global_conf['data'] = osp.join(test_root, 'test_data')
        global_conf['use_relative_links'] = False
        if on_travis:
            global_conf['nprocs'] = 2
        if self.use_db:
            self._clear_db()
            global_conf.update(db_config)
        rcParams['plotter.linreg.nboot'] = 1

    def tearDown(self):
        from psyplot import rcParams
        if self.remove_at_cleanup:
            if osp.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
            if osp.exists(self.config_dir):
                shutil.rmtree(self.config_dir)
            if self.use_db:
                self._clear_db()
        rcParams.update_from_defaultParams(plotters=False)

        del self.organizer
        del self.test_dir
        del self.config_dir

    def _clear_db(self):
        engine = utils.get_postgres_engine(db_config['database'])[0]
        conn = engine.connect()
        for table in engine.table_names():
            conn.execute("DROP TABLE %s;" % table)
        conn.close()

    @property
    def stations(self):
        """A numpy array of the stations in :attr:`test_stations`"""
        return np.loadtxt(
            self.stations_file, dtype='S11', usecols=[0]).astype(
                np.str_)

    def _test_setup(self):
        """Test the setup of a project. We make this method private such that
        it is not called everytime"""
        self.organizer.setup(self.test_dir, 'test_project0', link=False)
        mpath = osp.join(self.test_dir, 'test_project0')
        self.assertTrue(osp.isdir(mpath))
        original_files = sorted(map(osp.basename, glob.glob(osp.join(
            osp.dirname(gwgen.__file__), 'src', '*.f90'))))
        copied_files = sorted(map(osp.basename, glob.glob(osp.join(
            mpath, 'src', '*.f90'))))
        self.assertEqual(original_files, copied_files)
        self.assertIn('test_project0', self.organizer.config.projects)

        # createa new project and let it automatically assign the name
        self.organizer.setup(self.test_dir)
        mpath = osp.join(self.test_dir, 'test_project1')
        self.assertTrue(osp.isdir(mpath))
        original_files = sorted(map(osp.basename, glob.glob(osp.join(
            osp.dirname(gwgen.__file__), 'src', '*.f90'))))
        copied_files = sorted(map(osp.basename, glob.glob(osp.join(
            mpath, 'src', '*.f90'))))
        self.assertEqual(original_files, copied_files)
        self.assertIn('test_project1', self.organizer.config.projects)

    def _test_init(self):
        """Test the intialization of a new experiment. We make this method
        private such that it is not called everytime"""
        self.organizer.setup(self.test_dir)
        projectname = self.organizer.projectname
        self.organizer.init(experiment='testexp0')
        expdir = osp.join(
            self.test_dir, projectname, 'experiments', 'testexp0')
        self.assertTrue(osp.exists(expdir),
                        msg='Experiment directory %s does not exist!' % expdir)
        self.assertIn('testexp0', self.organizer.config.experiments)

        # test without argument
        self.organizer.setup(self.test_dir)
        projectname = self.organizer.projectname
        self.organizer.init(experiment=None)
        expdir = osp.join(
            self.test_dir, projectname, 'experiments', 'testexp1')
        self.assertTrue(osp.exists(expdir),
                        msg='Experiment directory %s does not exist!' % expdir)
        self.assertIn('testexp1', self.organizer.config.experiments)

    @staticmethod
    def _test_url(url, *args, **kwargs):
        if six.PY3:
            from urllib import request
            request.urlopen(url, *args, **kwargs)
        else:
            import urllib
            urllib.urlopen(url, *args, **kwargs)

    def assertAlmostArrayEqual(self, actual, desired, rtol=1e-07, atol=0,
                               msg=None, **kwargs):
        """Asserts that the two given arrays are almost the same

        This method uses the :func:`numpy.testing.assert_allclose` function
        to compare the two given arrays.

        Parameters
        ----------
        actual : array_like
            Array obtained.
        desired : array_like
            Array desired.
        rtol : float, optional
            Relative tolerance.
        atol : float, optional
            Absolute tolerance.
        equal_nan : bool, optional.
            If True, NaNs will compare equal.
        err_msg : str, optional
            The error message to be printed in case of failure.
        verbose : bool, optional
            If True, the conflicting values are appended to the error message.
        """
        try:
            np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol,
                                       err_msg=msg or '', **kwargs)
        except AssertionError as e:
            if six.PY2:
                self.fail(e.message)
            else:
                self.fail(str(e))


# check if we are online by trying to connect to google
try:
    BaseTest._test_url('https://www.google.de')
    online = True
except:
    online = False
