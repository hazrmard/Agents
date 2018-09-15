import unittest
from numbers import Number
from itertools import product

import numpy as np

from . import Neural, Polynomial, Tabular


def gen_data(order, num, indim, outdim=1, domain=(0, 1)):
    """
    Generates polynomial data points of specified order and output classes.

    * order (int): Order of base polynomial,
    * num (int): Number of samples to generate
    * indim (int): Number of variables in polynomial
    * outdim (int): Number of output classes i.e. number of polynomials
    * domain (Tuple[float, float]): Domain over which to sample polynomial.

    Returns:

    * The domain points/ training inputs -> X (2D array)
    * The domain points projected into feature space i.e powers/interaction terms
    * The target values / training outputs -> Y (2D array)
    * The weight matrix multiplied with feature points to generate Y
    """
    # generating sample points and sorting them in increasing order
    x = np.random.uniform(*domain, (num, indim))
    # projecting points into higher dimension
    x_ = (x[:, None] ** np.arange(1, order + 1)[:, None]).reshape(x.shape[0], -1, order='c')
    # generating sample weights
    w = np.random.rand(order * indim, outdim) - 0.5
    # calculating target variable
    y = x_ @ w + np.random.rand(outdim) - 0.5  # add biases
    return x, x_, y, w



class TestModels(unittest.TestCase):


    def setUp(self):
        pass


    def model_tester(self, model, indim=2, outdim=1, order=2, epochs=2, N=50):
        X, _, Y, _ = gen_data(order, N, indim, outdim)
        errs = []
        # training loop
        for i in range(epochs):
            for x, y in zip(X, Y):
                x = np.asarray(x).reshape(1, -1)       # converting to 2D array
                y = np.asarray(y).reshape(-1, outdim)  # converting to 2D array
                model.update(x, y)

            # testing loop after each epoch of training
            err = 0
            for x, y in zip(X, Y):
                x = np.asarray(x).reshape(1, -1)       # converting to 2D array
                y = np.asarray(y).reshape(-1, outdim)  # converting to 2D array
                ypred = model.predict(x)
                err += ((ypred - y)**2).sum()
                # testing that prediction returns copy and not reference
                ypred[0,0] = np.inf
                self.assertNotEqual(model.predict(x)[0,0], np.inf)
                # Testing prediction dimensions
                self.assertEqual(ypred.ndim, 2)             # 2D array and..
                self.assertEqual(ypred.shape[0], 1)         # ..2D row vector..
                self.assertEqual(ypred.shape[1], outdim)    # ..with outdim cols
            errs.append(err)
        self.assertTrue(errs[-1] < errs[0])


    def test_polynomial_model(self):
        for indim, outdim, order in product((1,2), (1,2), (2,3)):
            model = Polynomial(indim=indim, outdim=outdim, order=order)
            self.model_tester(model, indim=indim, outdim=outdim, order=order)


    def test_neural_model(self):
        for indim, outdim, order in product((1,2), (1,2), (2,3)):
            model = Neural(indim=indim, outdim=outdim, hidden_layer_sizes=(2, 3))
            self.model_tester(model, indim=indim, outdim=outdim, order=order)


    def test_tabular_model(self):
        dimsize = 10   # size of each dimension in table
        for indim, outdim, order in product((1,2), (1,2), (2,3)):
            dim_sizes = tuple([dimsize] * indim)
            model = Tabular(indim=indim, outdim=outdim, dim_sizes=dim_sizes,
                            lrate=0.3, low=0, high=1.)
            self.model_tester(model, indim=indim, outdim=outdim, order=order)


    def test_polynomial_weights(self):
        model = Polynomial(indim=3, outdim=2, order=1)
        old_w = model.weights
        self.assertIsInstance(old_w, list)          # list of coefs for sub-models
        self.assertIsInstance(old_w[0], tuple)      # sub-model is tuple..
        self.assertIsInstance(old_w[0][0], Number)      # .. of intercept
        self.assertIsInstance(old_w[0][1], np.ndarray)  # .. and feature weights

        new_w = [(np.random.rand(), np.random.rand(*old_w[0][1].shape))]
        model.weights = new_w
        self.assertEqual(model.weights[0][0], new_w[0][0])
        self.assertTrue(np.array_equal(model.weights[0][1], new_w[0][1]))


    def test_neural_weights(self):
        model = Neural(indim=3, outdim=2, hidden_layer_sizes=(3,))
        old_w = model.weights
        self.assertIsInstance(old_w, tuple)     # tuple of intercepts & weights
        self.assertEqual(len(old_w), 2)         # i.e. len==2
        self.assertIsInstance(old_w[0], list)   # intercepts and weights are..
        self.assertIsInstance(old_w[1], list)   # ..in a list for each hidden layer
        self.assertEqual(len(old_w[0]), 2)      # 1 hidden + 1 output layers
        self.assertEqual(len(old_w[1]), 2)      # 2 weight matrices in -> hid -> out
        self.assertIsInstance(old_w[0][0], np.ndarray)
        self.assertIsInstance(old_w[0][1], np.ndarray)

        new_w = ([np.ones(3)], [np.ones(3)])
        model.weights = new_w
        self.assertTrue(np.array_equal(np.ones(3), model.weights[0][0]))
        self.assertTrue(np.array_equal(np.ones(3), model.weights[1][0]))


    def test_tabular_weights(self):
        dimsize = 10   # size of each dimension in table
        indim, outdim = 3, 2
        dim_sizes = tuple([dimsize] * indim)

        model = Tabular(indim=indim, outdim=outdim, dim_sizes=dim_sizes)
        new_w = np.random.rand(*model.weights.shape)
        self.assertTupleEqual(model.weights.shape, (*dim_sizes, outdim))
        model.weights = new_w
        self.assertTrue(np.array_equal(new_w, model.weights))



if __name__ == '__main__':
    unittest.main(verbosity=0)
