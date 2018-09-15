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
    taking action=1 gives reward=0. Episode terminates when `maxsteps` actions
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
        # A value function with 2D features (state, action) mapping to value.
        self.value = Tabular(indim=2, dim_sizes=(2,2), outdim=1, lrate=self.lrate)
        # DummyAgent only takes action=0. This means it never reaches goal state
        # and always stays in state=0. The environment ends the episode after
        # maxsteps.
        self.agent = DummyAgent(actions=((0,),), env=self.env, value_function=self.value)
        # Resetting values to 0 so random initialization effects can be discarded.
        self.value.table *= 0



    def test_q(self):
        d = 1
        lr = self.lrate
        r = self.agent.learn(algorithm=q, episodes=1, discount=d)
        max_v = 0                   # maximum action value (0 by design)
        v0 = 0                      # initial value estimate of action=0

        r0 = -1                     # first step reward of action=0
        ret0 = r0 + d * max_v       # estimate of return
        v1 = v0 + lr * (ret0 - v0)  # value update

        r1 = -1                     # second step reward of action=0
        ret1 = r1 + d * max_v       # estimate of return
        v2 = v1 + lr * (ret1 - v1)  # value update
    
        r2 = -1                     # third step reward of action=0
        ret2 = r2 + d * max_v       # estimate of return
        v3 = v2 + lr * (ret2 - v2)  # value update

        actual = self.agent.value.predict(((0,0),))[0,0]
        self.assertAlmostEqual(r0+r1+r2, r)
        self.assertAlmostEqual(v3, actual)



    def test_nstepsarsa(self):
        r = self.agent.learn(algorithm=nstepsarsa, episodes=1, discount=1, steps=1)
        # Assuming discount=1,
        # Return with 0-step lookahead is:
        #   reward(t=0) + value(next_state/action)
        # Return with 1-step lookahead is:
        #   rewatd(t=-1) + reward(t=0) + value(next_state/action)
        expected = 0                    # initial value estimate of state=0, action=0
        expected += self.lrate*((-1-1+expected) - expected) # after first step
        expected += self.lrate*((-1-1+expected) - expected) # after second step
        expected += self.lrate*((-1+expected) - expected)   # terminal state reached, lookahead contracted
        actual = self.agent.value.predict(((0,0),))[0,0]
        self.assertAlmostEqual(expected, actual)



    def test_nsteptd(self):
        r = self.agent.learn(algorithm=nsteptd, episodes=1, discount=1, steps=1)
        # Assuming discount=1,
        # Return with 0-step lookahead is:
        #   reward(t=0) + max_value(next_state/action)
        # Return with 1-step lookahead is:
        #   rewatd(t=-1) + reward(t=0) + max_value(next_state/action)
        max_v = 0       # always 0, since never updated due to policy. All others are < 0
        expected = 0
        expected += self.lrate*((-1-1+max_v) - expected)
        expected += self.lrate*((-1-1+max_v) - expected)
        expected += self.lrate*((-1+max_v) - expected)    # terminal state reached, lookahead contracted
        actual = self.agent.value.predict(((0,0),))[0,0]
        self.assertAlmostEqual(expected, actual)



if __name__ == '__main__':
    unittest.main(verbosity=0)
