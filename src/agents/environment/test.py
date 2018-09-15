import unittest

import numpy as np
from gym.spaces import MultiDiscrete

from . import Environment
from .dummy.test import *


class TestEnvironment(unittest.TestCase):


    def setUp(self):
        env_size = 10
        act_size = 4
        num_goals = 3
        state_space = MultiDiscrete([env_size,])
        action_space = MultiDiscrete([act_size,])
        tmatrix = np.random.randint(0, env_size, (env_size, act_size))
        rmatrix = np.random.rand(env_size)
        goals = np.random.randint(0, env_size, num_goals)
        is_goal = lambda s: s in goals
        
        self.transition = lambda s, a: np.asarray(tmatrix[(s, a)])
        self.reward = lambda s, a, ns: rmatrix[int(ns)]
        self.env = Environment(self.reward, self.transition, state_space,\
                                action_space, is_goal)
    

    def test_transitions(self):
        for i in range(100):
            a = self.env.action_space.sample()
            ns, r, g, _ = self.env.step(a)
            self.assertIsInstance(ns, np.ndarray)
            self.assertIsInstance(r, float)
            self.assertIsInstance(g, bool)


    def test_reset(self):
        self.env.reset()
        self.assertEqual(self.env.t, 0)
        for i in range(10):
            self.env.step(self.env.action_space.sample())
        self.assertEqual(self.env.t, 10)
        self.env.reset()
        self.assertEqual(self.env.t, 0)



if __name__ == '__main__':
    unittest.main(verbosity=0)
