from gym.spaces import Discrete

from .dummyswitch import DummySwitch



class DummyIdentity(DummySwitch):
    """
    An identity environment. The state is whatever the action is.

    S=1 + A=100 ==> S=100 ...

    Args:

    * `maxsteps (int)`: Maximum number of steps before episode ends.
    * `random_state (int, np.random.RandomState)`: Random seed.
    """

    @classmethod
    def transition_func(cls, state: int, action: int) -> int:
        return action


    def __init__(self, maxsteps=3, random_state=None):
        observation_space = Discrete(self.NSTATES)
        action_space = Discrete(self.NSTATES)
        super().__init__(maxsteps=maxsteps, random_state=random_state)
