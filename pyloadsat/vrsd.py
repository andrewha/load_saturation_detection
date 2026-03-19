"""Load Saturation Detection Package.
Variance Response-Based Saturation Detector (VRSD) implementation.
"""
from collections import namedtuple
from typing import Any, Self

import numpy as np
import pandas as pd

from ._validation import validate_array_ndim
from .core import trend_lowess, window_acf, moments_right, response_lr, plateau_earliest
from .checks import is_monotonic_increasing, has_transition

class Detector:
    """Variance Response-Based Saturation Detector class.
    """

    def __init__(self,
                name: str | None = None,
                frac: float = 2.0 / 3.0,
                p_0: float = 0.8,
                window: int | None = None,
                nlags: int | None = None,
                alpha: float = 0.05,
                q_lower: float = 0.35,
                q_upper: float = 0.65,
                bandwidth: int | float | None = None) -> None:
        """Init detector.

        Parameters
        ----------
        name : str | None, optional
            Detector's name, by default None
        frac : float, optional
            LOWESS smoothing span, by default 2.0 / 3.0
        p_0 : float, optional
            Critical proportion value, by default 0.8
        window : int | None, optional
            Local moment window size, by default None
        nlags : int | None
            Number of lags to return ACF for, by default None
        alpha : float
            Lag significance level, by default 0.05
        q_lower : float, optional
            Lower quantile value, e.g., 0.1, by default 0.35
        q_upper : float, optional
            Upper quantile value, e.g., 0.9, by default 0.65
        bandwidth : int | float | None , optional
            Local linear regression bandwidth, by default None
            If integer, it is the bandwidth h
            If float, it is the bandwidth parameter k, and h is estimated via ROT
            If None, it is set to bandwidth parameter k = X.shape[0] / 3
        """
        # Hyperparameters
        self.name = name
        self.frac = frac
        self.p_0 = p_0
        self.window = window
        self.nlags = nlags
        self.alpha = alpha
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.bandwidth = bandwidth
        # Learned parameters
        self._trend = None
        self._means, self._variances, self._mean_max = None, None, None
        self._response = None
        # Detected parameters
        self._t_crit, self._load_crit = None, None
        self._length = None

        self.fitted = False
        self.detected = False

    def fit(self, ts: list | np.ndarray | pd.Series) -> Self:
        """Fit detector: estimate trend, local moments, and response function

        Parameters
        ----------
        ts : list | np.ndarray | pd.Series
            Time series data
        
        Returns
        -------
        Self
            Detector instance for cascading calls
        """
        ts = ts.copy()
        if isinstance(ts, list):
            ts = np.array(ts)
        ts = validate_array_ndim(ts, 1)

        trend = trend_lowess(ts, frac=self.frac)
        self._trend = trend

        monotonic_status, _ = is_monotonic_increasing(trend, p_0=self.p_0)
        if not monotonic_status:
            raise RuntimeError('Trend is not monotonically increasing.')

        if self.window is None:
            _, _, window = window_acf(ts - trend, nlags=self.nlags, alpha=self.alpha)
            self.window = window

        self._means, self._variances, self._mean_max = moments_right(trend, window=self.window)
        transition_status, transition_regime, _, _ = has_transition(self.means, self.mean_max,
                                                                    self.q_lower, self.q_upper)
        if not transition_status:
            raise RuntimeError(f'Regime transition is not identifiable: {transition_regime}')

        self._response = response_lr(y=self._variances, X=self._means, bandwidth=self.bandwidth)

        self.fitted = True
        return self

    def detect(self) -> tuple[bool, int, float]:
        """Detect transition time via estimated local response function
          and critical load via estimated local means.

        Returns
        -------
        tuple[bool, int, float]
            Namedtuple with fields
            'status': True (detected) | False (not detected)
            't_crit': Start index of longest plateau in response
            'load_crit': Critical load value
        """
        if not self.fitted:
            raise RuntimeError('Cannot detect yet. Call `fit` first.')

        t_crit = -1
        length = 0
        if self.window is not None and self._response is not None:
            response_ind = self._response <= 0.0
            # Find longest earliest plateau
            for l in range(self.window, self._response.shape[0] + 1, self.window):
                t_crit_l = plateau_earliest(response_ind, l)
                if t_crit_l > t_crit:
                    t_crit = t_crit_l
                    length = l # Shortest for found t_crit
            self._length = length
            self._t_crit = t_crit
        else:
            raise RuntimeError('Cannot detect yet. Call `fit` first.')
        if self._t_crit == -1:
            raise RuntimeError(f'No persistent plateau of length {l} found.')

        result = namedtuple('result', ['status', 't_crit', 'load_crit'])
        if isinstance(self._means, np.ndarray):
            self._load_crit = self._means[self._t_crit]
        if isinstance(self._means, pd.Series):
            self._load_crit = self._means.iloc[self._t_crit]

        self.detected = True
        return result(self.detected, self.t_crit, self.load_crit)

    def predict(self, X: Any = None) -> tuple[bool, int, float]: # pylint: disable=C0103, W0613
        """Standard ML alias for detect method.

        Parameters
        ----------
        X : Any
            Ignored, for compatibility only.

        Returns
        -------
        tuple[bool, int, float]
            Namedtuple with fields
            'status': True (detected) | False (not detected)
            't_crit': Start index of longest plateau in response
            'load_crit': 
        """
        return self.detect()

    def summary(self) -> dict[type[Self], dict[str, Any]]:
        """Show summary of all attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary of all attributes
        """
        return {self.__class__: vars(self)}

    @property
    def trend(self):
        if self._trend is None:
            raise RuntimeError('Trend is not estimated yet. Call `fit` first.')
        return self._trend

    @property
    def means(self):
        if self._means is None:
            raise RuntimeError('Local means are not estimated yet. Call `fit` first.')
        return self._means

    @property
    def variances(self):
        if self._variances is None:
            raise RuntimeError('Local variances are not estimated yet. Call `fit` first.')
        return self._variances

    @property
    def mean_max(self):
        if self._mean_max is None:
            raise RuntimeError('Local mean max is not estimated yet. Call `fit` first.')
        return self._mean_max

    @property
    def response(self):
        if self._response is None:
            raise RuntimeError('Local response function is not estimated yet. Call `fit` first.')
        return self._response

    @property
    def t_crit(self):
        if self._t_crit is None:
            raise RuntimeError('Transition time is not detected yet. Call `detect` first.')
        return self._t_crit

    @property
    def load_crit(self):
        if self._load_crit is None:
            raise RuntimeError('Critical load is not detected yet. Call `detect` first.')
        return self._load_crit

    @property
    def length(self):
        if self._length is None:
            raise RuntimeError('Plateau is not detected yet. Call `detect` first.')
        return self._length
