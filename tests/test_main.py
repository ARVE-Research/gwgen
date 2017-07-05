"""Test module for the :mod:`gwgen.main` module"""
import os.path as osp
import unittest
from gwgen.utils import file_len
import _base_testing as bt


class OrganizerTest(bt.BaseTest):
    """Test the :class:`gwgen.main.ModuleOrganizer` class"""

    def test_compile_model(self):
        """Test the compilation of a model"""
        self.organizer.setup(self.test_dir)
        projectname = self.organizer.projectname
        self.organizer.compile_model()
        binpath = osp.join(self.test_dir, projectname, 'bin', 'weathergen')
        self.assertTrue(osp.exists(binpath),
                        msg='binary %s does not exist!' % binpath)

    def test_run(self):
        self._test_init()
        organizer = self.organizer
        exp = organizer.experiment
        self.organizer.run(ifile=osp.join(
            bt.test_root, 'test_data', 'input.csv'))
        fname = osp.join(self.test_dir, organizer.projectname, 'experiments',
                         exp, 'outdata', exp + '.csv')
        self.assertTrue(osp.exists(fname), msg='Output file %s not found!' % (
            fname))
        nlines = file_len(fname)
        self.assertGreater(nlines, 2, msg='No output generated!')

    def test_wind_bias_correction(self):
        """Test gwgen bias wind"""
        self._test_init()
        self.organizer.exp_config['reference'] = osp.join(
            bt.test_root, 'test_data', 'long_reference.csv')
        fname, ext = osp.splitext(self.stations_file)
        self.stations_file = osp.join(bt.test_root, 'test_stations_long.dat')
        self.organizer.exp_config['eval_stations'] = self.stations_file
        ifile = osp.join(bt.test_root, 'test_data', 'long_input.csv')
        self.organizer.exp_config['namelist'] = {
            'weathergen_ctl': {'wind_bias_coeffs': [1.0] + [0.0] * 5,
                               'wind_intercept_bias_a': -9999.,
                               'wind_intercept_bias_b': -9999.},
            'main_ctl': {}}
        self.organizer.parse_args(['run',  '-i', ifile])
        self.organizer.parse_args('bias -q 1-100-5,99 wind'.split())
        self.organizer.fix_paths(self.organizer.exp_config)
        ofile = self.organizer.exp_config['postproc']['bias']['wind'][
            'plot_file']
        self.assertTrue(osp.exists(ofile), msg=ofile + ' is missing')


if __name__ == '__main__':
    unittest.main()
