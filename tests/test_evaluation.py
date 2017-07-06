# -*- coding: utf-8 -*-
"""Test module for the parameterization. Note that the entire task framework is
very sensible to internal errors , so if anything goes wrong, there should be
an error within the setup"""
import unittest
import os.path as osp
import pandas as pd
# we use assert_frame_equal instead of pd.equal, because it is less strict
from pandas.util.testing import assert_frame_equal
import _base_testing as bt
import test_parameterization as tp


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


class EvaluationPreparationTest(bt.BaseTest):
    """Test case for the :class:`gwgen.evaluation.EvaluationPreparation` class
    """

    def test_setup(self):
        """Test the setup from scratch"""
        self._test_init()
        organizer = self.organizer
        organizer.evaluate(self.stations_file, prepare={}, to_csv=True)
        ifile1 = osp.join(self.test_dir, organizer.projectname, 'experiments',
                          organizer.experiment, 'input', 'input.csv')
        self.assertTrue(osp.exists(ifile1), msg=ifile1 + ' is missing!')
        ref1 = osp.join(self.test_dir, organizer.projectname, 'experiments',
                        organizer.experiment, 'evaluation', 'reference.csv')
        self.assertTrue(osp.exists(ref1), msg=ref1 + ' is missing!')
        index_cols = ['id', 'year', 'month', 'day']

        # test for equality
        df1 = pd.read_csv(ifile1, index_col=index_cols[:-1])
        ifile2 = osp.join(bt.test_root, 'test_data', 'input.csv')
        df2 = pd.read_csv(ifile2, index_col=index_cols[:-1])
        merged = df2[[]].merge(df1, left_index=True, right_index=True,
                               how='left')
        result = tp.df_equals(merged.sort_index(), df2.sort_index())
        self.assertIsNone(result, msg=result)

        df1 = pd.read_csv(ref1, index_col=index_cols)
        ref2 = osp.join(bt.test_root, 'test_data', 'reference.csv')
        df2 = pd.read_csv(ref2, index_col=index_cols)
        merged = df2[[]].merge(df1, left_index=True, right_index=True,
                               how='left')
        result = tp.df_equals(merged.sort_index(), df2.sort_index())
        self.assertIsNone(result, msg=result)


if __name__ == '__main__':
    unittest.main()
