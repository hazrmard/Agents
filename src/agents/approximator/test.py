import unittest
import numpy as np
from . import Tabular
from . import Polynomial
from . import Neural



def gen_data(order, num, indim, outdim=1):
    # generating sample points and sorting them in increasing order
    x = np.random.uniform(0, 10, (num, indim))
    # projecting points into higher dimension
    x_ = (x[:, None] ** np.arange(1, order + 1)[:, None]).reshape(x.shape[0], -1, order='c')
    # generating sample weights
    w = np.random.rand(order * indim, outdim) - 0.5
    # calculating target variable
    y = x_ @ w + np.random.rand(outdim) - 0.5  # add biases
    return x, x_, y, w



class TestModels(unittest.TestCase):


    def setUp(self):
        dimsize = 10
        ndims = 3
        x, _, y, _ = gen_data(2, 50, ndims, 1)
        self.shape = tuple([dimsize] * ndims)
        self.x = x
        self.y = y.ravel()
        self.epochs = 2


    def model_tester(self, model):
        # updates model 1 reading at a time
        errs = []
        for i in range(self.epochs):
            for x, y in zip(self.x, self.y):
                x = np.asarray(x).reshape(1, -1)  # converting to 2D array
                y = np.asarray(y).reshape(1)      # converting to 1D array
                model.update(x, y)
            err = 0
            for x, y in zip(self.x, self.y):
                x = np.asarray(x).reshape(1, -1)  # converting to 2D array
                y = np.asarray(y).reshape(1)      # converting to 1D array
                ypred = model.predict(x)
                err += ((ypred - y)**2).sum()
            errs.append(err)
        self.assertTrue(errs[-1] < errs[0])


    def test_polynomial(self):
        model = Polynomial(2, 20, 10)
        self.model_tester(model)


    def test_neural(self):
        model = Neural((2, 3))
        self.model_tester(model)


    def test_tabular(self):
        model = Tabular(self.shape, lrate=0.3, low=0, high=1.)
        self.model_tester(model)
        model = Tabular(self.shape, lrate=0.3, low=0, high=(1.,))
        self.model_tester(model)



if __name__ == '__main__':
    unittest.main(verbosity=0)
