"""
Tabular function approximation.
"""

from typing import Iterable, Tuple, Union

import numpy as np
from numpy.random import RandomState

from ..helpers.schedule import Schedule
from .approximator import Approximator


class Tabular(Approximator):
    """
    A tabular approximation of a function. Uses gradient descent to update each
    repeated discrete observation using squared loss. Does not interpolate/
    extrapolate for unseen data points. The table has the shape `dim_sizes x outdim`.

    Args:
    * indim (int): Number of input features in a training example.
    * outdim (int): Number of output targets in a training example.
    * dim_sizes (int, Tuple[int]): Size of each feature's dimension in the table.
    If `int`, all feature dimensions get the same size.
    * lrate (float/`Schedule`). Constant or decaying learning rate [0, 1].
    * low (Tuple[int]): The lowest limits for each dimension. Defaults 0. Inclusive.
    * high (Tuple[int]): The highest limits for each dimension. Defaults to
    dimension sizes. Inclusive.
    * random_state (int, `np.random.RandomState`): Random seed.
    """

    def __init__(self, indim: int, outdim: int=1, dim_sizes: Union[int, Tuple[int]]=5,\
        lrate: Union[float, Schedule]=1e-2,low: Union[float, Tuple[int]]=0,\
        high: Union[float, Tuple[int]]=None,random_state: Union[int, RandomState]=None):

        self.lrate = Schedule(lrate) if not isinstance(lrate, Schedule) else lrate

        self.indim = indim
        self.outdim = outdim    
        self.dim_sizes = np.asarray(dim_sizes)
        self.low = np.asarray(low)
        self.high = np.asarray(dim_sizes) if high is None else np.asarray(high) 
        self.range = self.high - self.low
        self.random = random_state if isinstance(random_state, RandomState)\
                      else RandomState(random_state)
    
        dim_sizes = tuple([dim_sizes] * indim) if isinstance(dim_sizes, int) \
                    else dim_sizes
        self.table = self.random.uniform(-0.5, 0.5, size=(*dim_sizes, outdim))


    def discretize(self, key: Iterable[Union[float, int]]) -> np.ndarray:
        """
        Converts `x` feature tuples into valid indices that can be used to
        store/access the table.

        Args:
        * key: The features tuples/arrays.

        Returns:
        * An array of the corresponding indices into `Tabular.table`.
        """
        key = (np.asarray(key) - self.low) * self.dim_sizes / self.range
        key = np.clip(key, 0, self.dim_sizes-1)
        return key.astype(int)


    def update(self, x: Union[np.ndarray, Tuple], y: Union[np.ndarray, Tuple]):
        """
        Incrementally update function approximation using stochastic gradient
        descent.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance in
        each row.
        * y (Tuple, ndarray): A *2D* array of values to be learned at that point
        corresponding to each row of features in x.
        """
        x, y = np.asarray(x), np.asarray(y)
        keys = list(self.discretize(x))
        indices = list(zip(*keys))
        error = self.table[indices] - y
        self.table[indices] -= self.lrate() * error


    def predict(self, x: Union[np.ndarray, Tuple]) -> np.ndarray:
        """
        Predict value from the learned function given the input x.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance in
        each row.

        Returns:
        * A *2D* array of predictions for each instance in `x`.
        """
        x = np.asarray(x)
        keys = list(self.discretize(x_) for x_ in x)
        indices = list(zip(*keys))
        return self.table[indices].reshape(-1, self.outdim)


    @property
    def weights(self) -> np.ndarray:
        """
        Returns the whole table.
        """
        return self.table


    @weights.setter
    def weights(self, w: np.ndarray):
        if self.table.shape == w.shape:
            self.table[:] = w[:]
        else:
            raise TypeError('Shape mismatch.')
