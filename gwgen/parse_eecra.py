# -*- codng: utf-8 -*-
import re
import os.path as osp
import numpy as np
import pandas as pd
from collections import OrderedDict
from gwgen.utils import file_len
from gwgen._parseeecra import parseeecra

names = [
    'year', 'month', 'day', 'hour',
    'IB',
    'lat',
    'lon',
    'station_id',
    'LO',
    'ww',
    'N',
    'Nh',
    'h',
    'CL',
    'CM',
    'CH',
    'AM',
    'AH',
    'UM',
    'UH',
    'IC',
    'SA',
    'RI',
    'SLP',
    'WS',
    'WD',
    'AT',
    'DD',
    'EL',
    'IW',
    'IP']


def parse_file(ifile, year=None):
    """Parse a raw data file from EECRA and as a pandas DataFrame

    Parameters
    ----------
    ifile: str
        The raw (uncompressed) data file
    year: int
        The first year in the data file

    Returns
    -------
    pandas.DataFrame
        `ifile` parsed into a dataframe"""
    if year is None:
        m = re.match(r'\w{3}(\d{2})L', osp.basename(ifile))
        if not m:
            raise TypeError(
                "Could not infer year of file %s! Use the 'year' "
                "parameter!" % (ifile, ))
        year = int(m.group(1))
        year += 1900 if year > 60 else 2000
    df = pd.DataFrame.from_dict(OrderedDict(
        zip(names, parseeecra.parse_file(ifile, year, file_len(ifile)))))
    return df


def extract_data(ids, src_dir, target_dir, years=range(1971, 2010),
                 imonths=range(1, 13)):
    """Extract the data for the given EECRA stations

    This function extracts the data for the given `ids` from the EECRA data
    base stored in  `src_dir` into one file for each *id* in `ids`. The
    resulting filename will be like *id.csv*.

    Parameters
    ----------
    ids: np.ndarray of dtype int
        The numpy integer array with the station ids to extract
    src_dir: str
        The path to the source directory containing the raw (uncompressed)
        EECRA database
    target_dir: str
        The path to the output directory
    years: np.ndarray of dtype int
        The numpy integer array with the years to extract (by default, all
        years between 1971 and 2010)
    imonths: np.ndarray of dtype int
        The numpy integer array with the months to extract (by default, all
        from january to december)

    Returns
    -------
    numpy.ndarray
        The paths of the filenames corresponding to ids"""
    ids = np.asarray(ids).astype(int)
    years = np.asarray(years).astype(int)
    imonths = np.asarray(imonths).astype(int)
    for arr in [ids, years, imonths]:
        if arr.ndim == 0:
            arr.reshape((1,))
    parseeecra.extract_data(
        ids, osp.join(src_dir, ''), osp.join(target_dir, ''), years, imonths)
    return np.array([osp.join(src_dir, str(station_id) + '.csv')
                     for station_id in ids])
