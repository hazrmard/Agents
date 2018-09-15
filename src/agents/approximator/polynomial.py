"""
Implements a polynomial function approximation.
"""

from typing import Tuple, Union, List

import numpy as np
from numpy.random import RandomState
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures

from .approximator import Approximator


class Polynomial(Approximator):
    """
    Incremental approximation of a function using polynomial basis.

    Args:
    * indim (int): Number of input features in a training example.
    * outdim (int): Number of output targets in a training example.
    * order (int): The order of the polynomial (>=1).
    * random_state: Integer seed or `np.random.RandomState` instance.
    * default (float): The default value to return if predict called before fit.
    * kwargs: Any keyword arguments to be fed to sklearn.linear_model.SGDRegressor
    which fits to the function. Hard-coded arguments are `fit_intercept=True`.

    Attributes:
    * weights: 
    """

    def __init__(self, indim: int, outdim: int, order: int, default=0.,\
        tol: float=1e-3, random_state: Union[int, RandomState] = None, **kwargs):
        
        self.indim = indim
        self.outdim = outdim
        self.default = default
        self.order = order
        
        self.transformer = PolynomialFeatures(degree=order, include_bias=False)
        # self.powers = np.arange(1, self.order + 1)[:, None]
        
        kwargs['random_state'] = random_state   # to be passed to SGDRegressor
        kwargs['fit_intercept'] = True          # learn bias term for polynomial
        n_features = len(self._project(np.zeros((1, indim)))[0])
        self.models = []
        for _ in range(outdim):
            model = SGDRegressor(tol=tol, **kwargs)
            # SGDRegressor does not assign weight arrays until the first call
            # to 'fit()'. Allocating them early so the weights property is
            # accessible from the get-go. One initialized, they won't be reset
            # by the model.
            model._allocate_parameter_mem(n_classes=outdim, n_features=n_features)
            self.models.append(model)


    def _project(self, x: np.ndarray) -> np.ndarray:
        """
        Converts an input instance into higher dimensions consistent with the
        order of the polynomial being approximated.

        Args:
        * x (np.ndarray): A *2D* array representing a single instance.

        Returns:
        * a 2D np.ndarray with the projected features.
        """
        # use broadcasting to calculate powers of x in a 2D array, then reshape
        # it into a single array.
        # See: https://stackoverflow.com/q/50428668/4591810
        # order='f' : [x1^1, x2^1, x1^2, x2^2...],
        # order='c' : [x1^1, x1^2, ..., x2^1, x2^2]
        # return (x[:, None] ** self.powers).reshape(-1, order='f')  # 2D
        # return (x[:, None] ** self.powers).reshape(x.shape[0], -1, order='c')

        # TODO: A faster polynomial feature generator.
        return self.transformer.fit_transform(x)


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
        for i, model in enumerate(self.models):
            model.partial_fit(self._project(x), y[:, i])


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
        return np.asarray([model.predict(projection) for model in self.models]).T


    @property
    def weights(self) -> List[Tuple[float, np.ndarray]]:
        """
        Returns a tuple of:

        * The intercept term of the polynomial fitting as a numpy array (float),
        * Weights assigned to each feature generated after projecting inputs
        into feature-space using _project() (np.ndarray).
        """
        return [(float(model.intercept_), model.coef_) for model in self.models]


    @weights.setter
    def weights(self, w: List[Tuple[float, np.ndarray]]):
        for model, weight in zip(self.models, w):
            model.intercept_ = np.asarray(weight[0])
            model.coef_[:] = weight[1][:]
