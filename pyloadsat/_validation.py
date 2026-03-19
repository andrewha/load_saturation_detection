"""Load Saturation Detection Package.
Argument validation functions.
"""
from typing import Any
import numpy as np
import pandas as pd


def validate_type(val: Any, expected_type: Any, name: str = 'argument') -> Any:
    """Ensure correct argument's type.

    Parameters
    ----------
    val : Any
        Inspected argument's value
    expected_type : Any
        Expected argument's type
    name : str, optional
        Inspected argument's name, by default 'argument'

    Returns
    -------
    Any
        Inspected argument's value, if no exceptions

    Raises
    ------
    TypeError
        If wrong type.
    """
    if not isinstance(val, expected_type):
        raise TypeError(f'{name} must be {expected_type}, got {type(val)}.')
    return val


def validate_array_ndim(arr: np.ndarray | pd.Series,
                         expected_ndim: Any,
                         name: str = 'array') -> np.ndarray | pd.Series:
    """Ensure correct array's ndim.

    Parameters
    ----------
    arr : np.ndarray | pd.Series
        Inspected array
    expected_ndim : Any
        Expected array's ndim
    name : str, optional
        Inspected array's name, by default 'array'

    Returns
    -------
    np.ndarray | pd.Series
        Inspected array, if no exceptions

    Raises
    ------
    TypeError
        If wrong type.
    ValueError
        If wrong ndim.
    """
    if not hasattr(arr, 'ndim'):
        raise TypeError(f'{name} must be array-like, got {type(arr)}.')
    if arr.ndim != expected_ndim:
        raise ValueError(f'{name} must have ndim {expected_ndim}, got {arr.ndim}.')
    return arr


def validate_arrays_shape_equal(arr1: np.ndarray | pd.Series,
                                arr2: np.ndarray | pd.Series,
                                name1: str = 'array1',
                                name2: str = 'array2') -> tuple[np.ndarray|pd.Series,
                                                                np.ndarray|pd.Series]:
    """Ensure two arrays have equal shape.

    Parameters
    ----------
    arr1 : np.ndarray | pd.Series
        Inspected array 1
    arr2 : np.ndarray | pd.Series
        Inspected array 2
    name1 : str, optional
        Inspected array1's name, by default 'array1'
    name2 : str, optional
        Inspected array2's name, by default 'array2'

    Returns
    -------
    tuple[np.ndarray|pd.Series, np.ndarray|pd.Series]
        Inspected arrays, if no exceptions

    Raises
    ------
    ValueError
        If wrong shape.
    """
    if arr1.shape != arr2.shape:
        raise ValueError(f'{name1} and {name2} shapes are different: {arr1.shape}, {arr2.shape}.')
    return arr1, arr2


def validate_positive(val: Any, name: str = 'argument') -> Any:
    """Ensure argument is positive.

    Parameters
    ----------
    val : Any
        Inspected argument's value
    name : str, optional
        Inspected argument's name, by default 'argument'

    Returns
    -------
    Any
        Inspected argument's value, if no exceptions

    Raises
    ------
    ValueError
        If not positive
    """
    if val <= 0:
        raise ValueError(f'{name} must be positive, got {val}.')
    return val


def validate_in_range(val: Any, min_val: Any, max_val: Any, name: str = 'argument') -> Any:
    """Ensure argument's value is in range.

    Parameters
    ----------
    val : Any
        Inspected argument's value
    min_val : Any
        Inspected argument's min allowed value
    max_val : Any
        Inspected argument's max allowed value
    name : str, optional
        Inspected argument's name, by default 'argument'

    Returns
    -------
    Any
        Inspected argument's value, if no exceptions.

    Raises
    ------
    ValueError
        If not in range
    """
    if val < min_val:
        raise ValueError(f'{name} must be >= {min_val}, got {val}.')
    if val > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {val}")
    return val
