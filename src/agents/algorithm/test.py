import unittest
import random
from typing import Tuple, Union

from ..agent import Agent
from ..environment import dummy
from ..approximator import Tabular
from ..algorithm import q, nsteptd, nstepsarsa



class DummyEnv(dummy.DummyIdentity):
    """
    An identity environment with only 2 states. Taking action=0 gives reward=-1,
    taking action=1 gives reward=0. Episode terminates when `maxsteps` actiosn
    have been taken.
    """

    NSTATES = 2
    GOAL_STATES = (1,)

    def reset(self):
        self.state = 0
        return 0



class DummyAgent(Agent):
    """
    An agent with pre-defined policy. Picks actions randomly from a tuple
    of action choices provided at instantiation.
    """

    def __init__(self, actions: Tuple[Tuple[Union[int, float]]], env, value_function):
        self.action_pool = actions
        super().__init__(env=env, value_function=value_function)



    def set_action_selection_policy(self, policy):
        self.next_action = lambda x: random.choice(self.action_pool)



class TestGPIAlgorithms(unittest.TestCase):

    def setUp(self):
        self.lrate = 1e-1
        self.maxsteps = 3

        self.env = DummyEnv(maxsteps=self.maxsteps)
        self.value = Tabular(dims=(2,2), lrate=self.lrate)
        self.agent = DummyAgent(actions=((0,),), env=self.env, value_function=self.value)
        # self.agent1 = DummyAgent(actions=((1,),), env=self.env, value_function=self.value)
        
        self.value.table *= 0



    def test_q(self):
        r = self.agent.learn(algorithm=q, episodes=1, discount=1)
        expected = -0.1 + (-0.09) + (-0.081)
        actual = self.agent.value.table[0,0]
        self.assertAlmostEqual(expected, actual)



    def test_nstepsarsa(self):
        r = self.agent.learn(algorithm=nstepsarsa, episodes=1, discount=1, steps=1)
        expected = 0
        expected += self.lrate*(-1-1+0)
        expected += self.lrate*((-1-1+expected) - expected)
        expected += self.lrate*((-1+expected) - expected)    # terminal state reached, lookahead contracted
        actual = self.agent.value.table[0,0]
        self.assertAlmostEqual(expected, actual)



    def test_nsteptd(self):
        r = self.agent.learn(algorithm=nsteptd, episodes=1, discount=1, steps=1)
        expected = 0
        expected += self.lrate*(-1-1+0)
        expected += self.lrate*((-1-1+0) - expected)
        expected += self.lrate*((-1+0) - expected)    # terminal state reached, lookahead contracted
        actual = self.agent.value.table[0,0]
        self.assertAlmostEqual(expected, actual)



if __name__ == '__main__':
    unittest.main(verbosity=0)
