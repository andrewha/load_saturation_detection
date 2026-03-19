"""Load Saturation Detection Package.
Core functions.
"""
from collections import namedtuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import acf as ACF
from statsmodels.regression.rolling import RollingOLS

from ._validation import (validate_array_ndim,
                          validate_arrays_shape_equal,
                          validate_in_range,
                          validate_positive,
                          validate_type)


def trend_lowess(ts: list | np.ndarray | pd.Series,
                 frac: float = 2.0 / 3.0) -> np.ndarray|pd.Series:
    """Estimate trend of time series data via LOWESS.

    Parameters
    ----------
    ts : list | np.ndarray | pd.Series
        Time series data.
    frac : float, optional
        LOWESS smoothing span, by default 2.0 / 3.0

    Returns
    -------
    pd.Series
        Estimated trend values.
    """
    ts = ts.copy()
    if isinstance(ts, list):
        ts = np.array(ts)
    ts = validate_array_ndim(ts, 1)
    frac = validate_in_range(frac, 0.0, 1.0, 'frac')

    trend = None
    t = np.arange(ts.shape[0])
    trend = lowess(ts, t, frac=frac, return_sorted=False)
    if isinstance(ts, pd.Series):
        trend = pd.Series(trend)
        trend.index = ts.index

    return trend


def window_acf(ts: list | np.ndarray | pd.Series,
                        nlags: int | None = None,
                        alpha: float = 0.05) -> tuple[np.ndarray|pd.Series,
                                                    np.ndarray|pd.Series,
                                                    int]:
    """Estimate local moment window size via ACF.

    Parameters
    ----------
    ts : list | np.ndarray | pd.Series
        Detrended time series data
    nlags : int | None, by default None
        Number of lags to return ACF for
    alpha : float, optional
        Lag significance level, by default 0.05

    Returns
    -------
    tuple[np.ndarray|pd.Series, np.ndarray|pd.Series, int]
        ACF, ACF peaks, window size
    """

    ts = ts.copy()
    if isinstance(ts, list):
        ts = np.array(ts)
    ts = validate_array_ndim(ts, 1)
    if nlags is not None:
        nlags = validate_type(nlags, int, 'nlags')
        nlags = validate_in_range(nlags, 1, ts.shape[0] - 1, 'nlags')
    alpha = validate_in_range(alpha, 0.0, 1.0, 'alpha')

    acf, acf_peaks, window = None, None, None
    result = namedtuple('result', ['acf', 'acf_peaks', 'window'])
    acf = ACF(ts, nlags=nlags, alpha=alpha)
    acf_peaks = np.diff(acf[0])
    acf_peaks = np.insert(acf_peaks, 0, np.nan) # Restore preceding nan
    acf_peaks = acf_peaks > 0
    acf_peaks = np.where(acf_peaks, 1.0, np.nan)
    # Longest significant lag
    window = int(np.flatnonzero(acf[0] > (acf[0] - acf[1][:, 0]))[-1])
    window = max(2, window) # window = 2 if no significant autocorrelation

    return result(acf, acf_peaks, window)


def moments_right(ts: list | np.ndarray | pd.Series,
            window: int = 2) -> tuple[np.ndarray|pd.Series,
                                        np.ndarray|pd.Series,
                                        float]:
    """Compute rolling means (first local moments) and 
    rolling variances (centered second local moments) of time series data
    using right (causal) sliding windows.

    Parameters
    ----------
    ts : list | np.ndarray | pd.Series
        Time series
    window : int, optional
        Local moment window size, by default 2

    Returns
    -------
    tuple[np.ndarray|pd.Series, np.ndarray|pd.Series, float]
        Namedtuple with fields
        'means': rolling means
        'variances': rolling variances
        'mean_max': mean value that maximizes variance
    """
    ts = ts.copy()
    if isinstance(ts, list):
        ts = np.array(ts)
    ts = validate_array_ndim(ts, 1)
    window = validate_type(window, int, 'window')
    window = validate_positive(window, 'window')

    means, variances, mean_max = None, None, None
    result = namedtuple('result', ['means', 'vars', 'mean_max'])

    if isinstance(ts, np.ndarray):
        windows = sliding_window_view(ts, window_shape=window)
        means = np.nanmean(windows, axis=1)
        means = np.insert(means, 0, np.full(window - 1, np.nan)) # Restore preceding nan
        variances = np.nanvar(windows, axis=1, ddof=1)
        variances = np.insert(variances, 0, np.full(window - 1, np.nan)) # Restore preceding nan
        mean_max = means[np.nanargmax(variances)]
    if isinstance(ts, pd.Series):
        means = ts.rolling(window, center=False).mean()
        variances = ts.rolling(window, center=False).var(ddof=1)
        mean_max = means.loc[variances.idxmax()]

    return result(means, variances, mean_max)


def response_lr(y: list | np.ndarray | pd.Series,
                     X: list | np.ndarray | pd.Series, # pylint: disable=C0103
                     bandwidth: int | float | None = None) -> np.ndarray|pd.Series|None:
    """Estimate local response function via local linear regression.

    Parameters
    ----------
    y : list | np.ndarray | pd.Series
        Dependent variable data
    X : list | np.ndarray | pd.Series
        Independent variable data
    bandwidth : int | float | None, optional
        Local linear regression bandwidth, by default None
        If integer, it is the bandwidth h
        If float, it is the bandwidth parameter k, and h is estimated via ROT
        If None, it is set to bandwidth parameter k = X.shape[0] / 3

    Returns
    -------
    np.ndarray|pd.Series
        Vector of estimated response
    """
    X = X.copy()
    if isinstance(X, list):
        X = np.array(X)
    X = validate_array_ndim(X, 1, 'X')
    y = y.copy()
    if isinstance(y, list):
        y = np.array(y)
    y = validate_array_ndim(y, 1, 'y')
    X, y = validate_arrays_shape_equal(X, y, 'X', 'y')
    if bandwidth is None:
        bandwidth = X.shape[0] / 3
    bandwidth = validate_type(bandwidth, (int, float), 'bandwidth')
    bandwidth = validate_positive(bandwidth, 'bandwidth')
    if isinstance(bandwidth, float): # k, else h
        bandwidth = int(np.floor(bandwidth * X.shape[0] ** (-1/5)))
    bandwidth = validate_in_range(bandwidth, 2, X.shape[0], 'bandwidth')

    beta_lr = None
    X = sm.add_constant(X)
    rols = RollingOLS(endog=y, exog=X, window=bandwidth)
    rols_res = rols.fit(params_only=True)
    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        beta_lr = rols_res.params[:, 1]
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        beta_lr = rols_res.params[0]
    return beta_lr


def plateau_earliest(ts_ind: list | np.ndarray | pd.Series, length: int = 1) -> int:
    """Find the earliest start index of the subsequence of required length (streak).

    Parameters
    ----------
    ts_ind : list | np.ndarray | pd.Series
        Indicator time series data (True / False)
    length : int, optional
        Required subsequence length, by default 1

    Returns
    -------
    int
        Index of the first element in the streak
        Return -1 if not found
    """
    length = validate_type(length, int, 'length')
    n = len(ts_ind)
    length = validate_in_range(length, 1, n, 'length')
    ts_ind = ts_ind.copy()
    ts_ind = np.array(ts_ind)
    ts_ind = validate_array_ndim(ts_ind, 1)

    current_streak = 0
    for i in range(n):
        if ts_ind[i]: # True
            current_streak += 1
            if current_streak == length:
                return i - length + 1 # Return start index if streak found
        else:
            current_streak = 0 # Streak broken
    return -1 # No streak found
