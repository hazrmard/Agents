"""
Implemants fully-connected artificial neural network function approximation.
"""

from typing import Tuple, Union, List

import numpy as np
from numpy.random import RandomState
from sklearn.neural_network import MLPRegressor
from sklearn.utils import check_random_state

from .approximator import Approximator


class Neural(Approximator):
    """
    A fully-connected neural network to approximate a function.

    Args:
    * hidden_layer_sizes: A tuple of ints representing number of units in each
    hidden layer.
    * random_state: Integer seed or `np.random.RandomState` instance.
    * norm(Tuple[Tuple[float, float]]): Bounds to use to normalize input between
    [0, 1]. Must be same length as features. Of the form ((low, high), (low, high)...)
    * default (float): The default value to return if predict called before fit.
    * kwargs: Any keyword arguments to be fed to `sklearn.neural_network.MPLRegressor`
    which fits to the function. Hard-coded arguments are `warm_start`, `max_iter`.
    """


    def __init__(self, indim: int, outdim: int, hidden_layer_sizes: Tuple[int],\
        norm: Tuple[Tuple[float, float]] = None, default = 0.,\
        random_state: Union[int, RandomState] = None, **kwargs):
        self.default = default

        self.indim = indim
        self.outdim = outdim

        kwargs['random_state'] = random_state
        self.model = MLPRegressor(hidden_layer_sizes, **kwargs)
        # the model does not initialize weights/random states until the first
        # call to 'fit()'. Setting up everything early so the weights property
        # is accessible from the get-go. Once initialized, the model won't
        # reset values. See sklearn.utils.check_random_state() and
        # MLPRegressor._fit()
        self.model._random_state = check_random_state(self.model.random_state)
        self.model._initialize(y=np.ones((1, outdim)),
                               layer_units=[indim, *hidden_layer_sizes, outdim])

        self.bounds = np.asarray(norm).T if norm is not None else None
        self.range = self.bounds[1] - self.bounds[0] if norm is not None else None


    def _project(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes an array of features. Truncates them to `indim` features per
        instance.
        """
        if self.bounds is not None:
            return (x - self.bounds[0]) / self.range
        return x


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
        # If number of output classes is 1, scipy wants a 1D array instead of
        # a 2D array with 1 column.
        if self.outdim == 1:
            self.model.partial_fit(self._project(x), y.ravel())
        else:
            self.model.partial_fit(self._project(x), y)


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
        projection = self._project(x)
        return self.model.predict(projection).reshape(-1, self.outdim)


    @property
    def weights(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Returns a tuple of:

        * A list of # hidden layers vectors where each element is the bias for
        a unit in a layer.
        * A list # hidden layers weight matrices.
        """
        return (self.model.intercepts_, self.model.coefs_)


    @weights.setter
    def weights(self, w: Tuple[List[np.ndarray], List[np.ndarray]]):
            self.model.intercepts_[:] = w[0][:]
            self.model.coefs_[:] = w[1][:]
