# -*- coding: utf-8 -*-
from gwgen._parseghcnrow import parseghcnrow
import six
import pandas as pd
import numpy as np
from itertools import chain
import datetime as dt
import re

if six.PY2:
    from itertools import imap as map, izip as zip

daymon_patt = re.compile(r'(?:\w|-){11}(\d{6})(?:TMAX|TMIN|PRCP)')


def read_ghcn_file(ifile):
    """Read in a GHCN station data file and convert it to a dataframe

    Parameters
    ----------
    ifile: str
        The path to a ghcn datafile

    Returns
    -------
    pandas.DataFrame
        The `ifile` converted to a dataframe"""
    # get number of days in the file
    with open(ifile) as f:
        ndays = np.sum(list(map(ndaymon, np.unique(list(map(
            lambda m: m.group(1), filter(
                None, map(daymon_patt.match, f.readlines()))))))))
    stationid, dates, variables, flags, j = parseghcnrow.parse_station(
        ifile, ndays or 100)
    dates = dates[:j]
    variables = variables[:j].astype(np.float64)
    flags = flags[:j].astype(np.str_)
    flags = np.core.defchararray.replace(flags, ' ', '')
    variables[np.isclose(variables, -9999.) |
              np.isclose(variables, -999.9)] = np.nan
    vlst = ['tmin', 'tmax', 'prcp']
    df = pd.DataFrame.from_dict(dict(chain(
        [('id', np.repeat(np.array([stationid]).astype(np.str_), j))],
        zip(('year', 'month', 'day'), dates.T),
        zip(vlst, variables.T),
        chain(*[zip((var + '_m', var + '_q', var + '_s'), arr)
                for var, arr in zip(vlst, np.rollaxis(flags, 2, 1).T)]))))
    return df


def ndaymon(yearmon):
    """Calculate the number of days for one month in a year

    Parameters
    ----------
    yearmon: str
        The first 4 numbers stand for the year, the others for the month
        (in datetime writing: ``'%Y%m'``)"""
    year = int(yearmon[:4])
    month = int(yearmon[4:])
    d = dt.date(year, month, 1)
    d2 = d.replace(
        year=year + 1 if month == 12 else year,
        month=1 if month == 12 else month + 1)
    return (d2 - d).days
