"""Load Saturation Detection Package.
Data makers.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pyloadsat import validate_positive, validate_type, validate_array_ndim, erlangb_carriedv


def make_demand_exp(rate: float, size: int) -> np.ndarray:
    """Generate demand with exponential growth.

    Parameters
    ----------
    rate : float
        Growth rate
    size : int
        Number of observations

    Returns
    -------
    np.ndarray
        Generated demand values
    """
    rate = validate_positive(rate, 'rate')
    size = validate_type(size, int, 'size')
    size = validate_positive(size, 'size')

    t = np.arange(size)
    demand = np.exp(rate * t)

    return demand


def make_carried(model: str,
                 demand: list | np.ndarray | pd.Series,
                 cap: float = 0.75,
                 ar_noise: tuple[float, int, float] | None = (0.3, 10, 1.0),
                 outliers: int | None = 5,
                 seed: int | None = None) -> tuple[np.ndarray|pd.Series, np.ndarray|pd.Series]:
    """Generate carried load and capacity data with the given service model.

    Parameters
    ----------
    model : str
        Carried load service model
        One of 'erlangb', 'hardcap'
    demand : list | np.ndarray | pd.Series
        Demand time series
    cap : float, optional
        System capacity positive dimensionless parameter, by default 0.75
        Use it to configure capacity constraint in the system
        For large cap, system operates in elastic-only regime
        For small cap, system operates in saturated-only regime
        For some value between large and small, system is transition-observable
    ar_noise : tuple[float, int, float] | None, optional
        Add autocorrelated seasonal noise AR(1)_w, by default (0.3, 10, 1.0)
        Tuple of three noise parameters:
            phi : float
                AR(1)_w coefficient, by default 0.3
            w : int
                AR(1)_w seasonality, by default 10
            sigma : float
                AR noise standard deviation, by default 1.0
    outliers : int | None, optional
        Insert a number of outliers at random indices, by default 5
    seed : int | None, optional
        RNG seed, by default None

    Returns
    -------
    tuple[np.ndarray|pd.Series, np.ndarray|pd.Series]
        Generated carried load and capacity values
    """
    if model not in ('erlangb', 'hardcap'):
        raise ValueError("model must be one of 'erlangb', 'hardcap'")
    demand = demand.copy()
    if isinstance(demand, list):
        demand = np.array(demand)
    demand = validate_array_ndim(demand, 1, 'demand')
    size = demand.shape[0]
    cap = validate_positive(cap, 'cap')
    # Rescale to constant capacity in load units
    capacity = (cap * demand[-1] * np.ones(size)).astype(int)

    if model == 'erlangb':
        carried = erlangb_carriedv(demand, capacity) # Erlang B model
    elif model == 'hardcap':
        carried = np.minimum(demand, capacity) # Hard capacity model
    else:
        raise ValueError("model must be one of 'erlangb', 'hardcap'")

    if ar_noise is not None:
        rng = np.random.default_rng(seed=seed)
        phi, w, sigma = ar_noise
        coeffs = [1.0] + [0.0] * (w - 1) + [-phi]
        noise = sm.tsa.ArmaProcess(ar=coeffs).generate_sample(nsample=size,
                                            distrvs=lambda size: rng.normal(0.0, sigma, size=size))
        carried = carried + noise

    if outliers is not None:
        outliers = validate_type(outliers, int, 'outliers')
        outliers = validate_positive(outliers, 'outliers')
        rng = np.random.default_rng(seed=seed)
        t = np.arange(size)
        t_outliers = rng.choice(t, outliers, False)
        carried[t_outliers] *= 0.5

    return (carried, capacity)
