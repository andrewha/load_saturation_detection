"""Load Saturation Detection Package.
Blocking calculation functions.
"""
import numpy as np

from ._validation import validate_type, validate_positive


def erlangb_prob(offered: float, capacity: int) -> float:
    """Calculate blocking probability using Erlang B.

    Parameters
    ----------
    offered : float
        Offered load in Erlangs
    capacity : int
        System capacity (number of channels)

    Returns
    -------
    float
        Blocking probability
    """
    offered = validate_positive(offered, 'offered')
    capacity = validate_type(capacity, int, 'capacity')
    capacity = validate_positive(capacity, 'capacity')
    InvB = 1.0
    for j in range(1, capacity + 1):
        InvB = 1.0 + InvB * (j / offered)

    return 1.0 / InvB


# Vectorized implementation of erlangb_prob
erlangb_probv = np.vectorize(erlangb_prob, otypes=['float'])


def erlangb_carried(offered: float, capacity: int) -> float:
    """Convert offered load to carried load for given capacity using Erlang B.

    Parameters
    ----------
    offered : np.ndarray
        Offered load in Erlangs
    capacity : int
        System capacity (number of channels)

    Returns
    -------
    float
        Carried load in Erlangs
    """
    offered = validate_positive(offered, 'offered')
    capacity = validate_type(capacity, int, 'capacity')
    capacity = validate_positive(capacity, 'capacity')
    carried = offered * (1 - erlangb_prob(offered, capacity))
    return carried


# Vectorized implementation of erlangb_carried
erlangb_carriedv = np.vectorize(erlangb_carried, otypes=['float'])


# Convert capacity to offered load for given blocking probability
erlangb_offered = {2: # blocking 2%
                   {6: # channels
                    2.25, # Erlangs
                    14: 8.2, 20: 13.15, 29: 21.0}
                }
