"""
Implements the base Approximator class. Cannot be used as-is. Must be subclassed.
"""
from typing import Tuple, Union
import numpy as np


class Approximator:
    """
    A machine learning model that can learn mapping between instances of features
    and values (x, y). All feature/targets are 2D numpy arrays where each row is
    an instance and each column is a variable/class.

    Args:
    * indim (int): Number of input features in a training example.
    * outdim (int): Number of output targets in a training example.
    """


    def __init__(self, indim: int, outdim: int, *args, **kwargs):
        pass


    def update(self, x: Union[np.ndarray, Tuple], y: Union[np.ndarray, Tuple]):
        """
        Update function approximation using stochastic gradient
        descent.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance in
        each row.
        * y (Tuple, ndarray): A *2D* array of values to be learned at that point
        corresponding to each row of features in x.
        """
        pass
    

    def predict(self, x: Union[np.ndarray, Tuple]) -> np.ndarray:
        """
        Predict value from the learned function given the input x.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance.

        Returns:
        * A *2D* array of predictions for each feature in `x`.
        """
        pass
    

    @property
    def weights(self):
        """
        Property method that returns the weights of an approximator. The type of
        weights for different classes need not be consistent (one can be list, 
        other can be array).
        """
        pass


    @weights.setter
    def weights(self, w):
        """
        Complementary setter method for weights property. Sets weights.
        """
        pass
    

    def __getitem__(self, x):
        return self.predict(x)
    

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)