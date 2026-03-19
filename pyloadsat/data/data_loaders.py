"""Load Saturation Detection Package.
Data loaders.
"""
import os

import numpy as np
import pandas as pd

from pyloadsat import validate_array_ndim, validate_arrays_shape_equal


def load_carried(model: str) -> tuple[np.ndarray|pd.Series, np.ndarray|pd.Series]:
    """Load synthetic carried load and capacity data from NumPy file f'synth_carried_{model}.npy'.
    First row: carried load with 1000 points
    Second row: system capacity with 1000 points

    Parameters
    ----------
    model : str
        Carried load service model
        One of 'erlangb', 'hardcap'

    Returns
    -------
    tuple[np.ndarray|pd.Series, np.ndarray|pd.Series]
        Loaded values.
    """
    if model not in ('erlangb', 'hardcap'):
        raise ValueError("model must be one of 'erlangb', 'hardcap'")
    # Save current working directory
    current_working_directory = os.getcwd()
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    os.chdir(dir_path)
    with open(f'synth_carried_{model}.npy', 'rb') as f:
        data = np.load(f)
    # Restore current working directory
    os.chdir(current_working_directory)
    carried = data[0]
    capacity = data[1]
    carried = validate_array_ndim(carried, 1, 'carried')
    capacity = validate_array_ndim(capacity, 1, 'capacity')
    carried, capacity = validate_arrays_shape_equal(carried, capacity, 'carried', 'capacity')

    return (carried, capacity)
