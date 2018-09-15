import unittest
from numbers import Number

import numpy as np

from . import Dummy1DContinuous
from . import DummyIdentity
from . import DummySwitch
from . import TicTacToe
from .. import Environment



class TestDummyEnv(unittest.TestCase):

    def env_tester(self, env: Environment):
        type_s = type(env.reset())
        for i in range(env.maxsteps):
            s, r, d, _ = env.step(env.action_space.sample())
            self.assertEqual(type_s, type(s))
            self.assertTrue(isinstance(d, (bool, np.bool_, np.bool)))
            self.assertIsInstance(r, Number)


    def test_Dummy1DContinuous(self):
        env = Dummy1DContinuous()
        self.env_tester(env)


    def test_DummySwitch(self):
        env = DummySwitch()
        self.env_tester(env)


    def test_DummyIdentity(self):
        env = DummyIdentity()
        self.env_tester(env)


    def test_TicTacToe(self):
        env = TicTacToe()
        self.env_tester(env)



if __name__ == '__main__':
    unittest.main(verbosity=0)
