import numpy as np
from gym.spaces import Discrete, MultiDiscrete

from ..environment import Environment



class TicTacToe(Environment):
    """
    A classic Tic-Tac-Toe game.

     0 | 1 | 2
    ___|___|___
     3 | 4 | 5
    ___|___|___
     6 | 7 | 8

    A 3x3 tic-tac-toe board. Agent is 'x', opponent is 'o'. There are 9 possible
    actions representing where to put the 'x' (0, 1, 2...8) from left to
    right and top row to bottom in that order (action 3 = 1st col, 2nd row).
    The state of the board is represented by a 9-vector. 0 means blank, 1 means
    opponent has put 'o', 2 means agent has put 'x' for the location corresponding
    to the index in the vector.
    """

    win_combos = np.asarray(((0, 1, 2),
                             (3, 4, 5),
                             (6, 7, 8),
                             (0, 3, 6),
                             (1, 4, 7),
                             (2, 5, 8),
                             (0, 4, 8),
                             (2, 4, 6)))

    @classmethod
    def win(cls, state: np.ndarray) -> int:
        win_spaces = state[cls.win_combos]
        if np.any(np.all(win_spaces == 1, axis=1)):     # opponent wins
            return 1
        elif np.any(np.all(win_spaces == 2, axis=1)):   # agent wins
            return 2
        else:
            return 0


    @classmethod
    def isgoal(cls, state: np.ndarray) -> bool:
        return cls.win(state) == 2


    @classmethod
    def opponent(cls, state: np.ndarray, action: int) -> np.ndarray:
        if state[action] == 0:
            state[action] = 2


        return state


    @classmethod
    def reinforcement(cls, state: np.ndarray, action: int, nstate: np.ndarray) -> float:
        if state[action] == 0:     # valid action (i.e. on blank space)
            winner = cls.win(nstate)
            if winner == 1:         # opponent wins
                return -5
            elif winner == 2:       # player wins
                return 5
            else:                   # valid move, game continuing
                return 0
        return -1                   # invalid action reinforcement


    def __init__(self, random_state=None):
        action_space = Discrete(9)
        observation_space = MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3])
        max_steps = 5
        super().__init__(reward=self.reinforcement, transition=self.opponent,
                         observation_space=observation_space,
                         action_space=action_space,
                         maxsteps=max_steps,
                         random_state=random_state, goal=self.isgoal)