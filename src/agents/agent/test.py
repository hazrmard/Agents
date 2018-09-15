import unittest
from collections import namedtuple

import gym
import numpy as np

from .memory import Memory
from .gpi.test import *



class TestMemory(unittest.TestCase):


    def test_sample(self):
        for x in range(1, 5):
            for y in range(1, 5):
                m = Memory(x, y)
                for i in range(10):
                    m.append(i)
                    s = m.sample()
                    self.assertEqual(len(s), y)
                    self.assertTrue(all([z in m for z in s]))



if __name__ == '__main__':
    unittest.main(verbosity=0)
