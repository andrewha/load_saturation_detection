"""Load Saturation Detection Package.
Assumption check functions.
"""
from collections import namedtuple

import numpy as np
import pandas as pd

from ._validation import validate_array_ndim, validate_positive, validate_in_range


def is_constant(ts: list | np.ndarray | pd.Series) -> tuple[bool]:
    """Check if time series data is constant.

    Parameters
    ----------
    ts : list | np.ndarray | pd.Series
        Time series data
        
    Returns
    -------
    tuple[bool]
        Namedtuple with field
        'status': True (constant) | False (non-constant)
    """
    ts = ts.copy()
    if isinstance(ts, list):
        ts = np.array(ts)
    ts = validate_array_ndim(ts, 1)

    status = None
    result = namedtuple('result', ['status'])

    if isinstance(ts, np.ndarray):
        status = np.all(np.diff(ts) == 0.)
    if isinstance(ts, pd.Series):
        status = (ts.diff().dropna() == 0.).all()

    return result(status,)


def is_monotonic_increasing(ts: list | np.ndarray | pd.Series,
                            p_0: float = 0.8) -> tuple[bool, float]:
    """Check if time series data is monotonically increasing.
    Compute the proportion of non-negative finite differences within all finite differences.

    Parameters
    ----------
    ts : list | np.ndarray | pd.Series
        Time series data
    p_0 : float | int
        Critical proportion value, by default 0.8

    Returns
    -------
    tuple[bool, float]
        Namedtuple with fields
        'status': True (monotonic) | False (non-monotonic)
        'p': proportion of non-negative finite differences
    """
    p_0 = validate_positive(p_0, 'p_0')
    ts = ts.copy()
    if isinstance(ts, list):
        ts = np.array(ts)
    ts = validate_array_ndim(ts, 1)

    status = None
    p = None
    result = namedtuple('result', ['status', 'p'])

    if isinstance(ts, np.ndarray):
        p = np.mean(np.diff(ts) >= 0)
    if isinstance(ts, pd.Series):
        p = (ts.diff() >= 0).mean(skipna=True)
    if p is not None:
        status = p > p_0

    return result(status, p)


def has_transition(ts: list | np.ndarray | pd.Series,
                       val: float,
                       q_lower: float = 0.35,
                       q_upper: float = 0.65) -> tuple[str, str, float, float]:
    """Check regime transition observability in data: 
    elastic-only, saturated-only, transition-observable (elastic followed by saturated).

    Parameters
    ----------
    ts : list | np.ndarray | pd.Series
        Time series data
    val : float
        Value that should be covered by [data_lower, data_upper]
    q_lower : float
        Lower quantile value, e.g., 0.1, by default 0.35
    q_upper : float
        Upper quantile value, e.g., 0.9, by default 0.65

    Returns
    -------
    tuple[str, str, float, float]
        Namedtuple with fields
        'status': True (monotonic) | False (non-monotonic)
        'regime': Observed regime of data
        'data_lower': Data value for lower quantile
        'data_upper': Data value for upper quantile
        
    """
    q_lower = validate_in_range(q_lower, 0.0, 1.0, 'q_lower')
    q_upper = validate_in_range(q_upper, 0.0, 1.0, 'q_upper')

    ts = ts.copy()
    if isinstance(ts, list):
        ts = np.array(ts)
    ts = validate_array_ndim(ts, 1)

    status, regime, data_lower, data_upper = None, None, None, None
    result = namedtuple('result', ['status', 'regime', 'data_lower', 'data_upper'])

    if isinstance(ts, np.ndarray):
        data_lower, data_upper = np.nanquantile(ts, [q_lower, q_upper])
    if isinstance(ts, pd.Series):
        data_lower, data_upper = ts.quantile([q_lower, q_upper]).values

    if data_lower is not None and data_upper is not None:
        if val > data_upper:
            status = False
            regime = 'elastic-only'
        elif val < data_lower:
            status = False
            regime = 'saturated-only'
        else:
            status = True
            regime = 'transition-observable'

    return result(status, regime, data_lower, data_upper)
