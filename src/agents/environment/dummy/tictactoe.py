import numpy as np
from gym.spaces import Discrete, MultiBinary

from ..environment import Environment



class TicTacToe(Environment):
    """
     0 | 1 | 2
    ___|___|___
     3 | 4 | 5
    ___|___|___
     6 | 7 | 8
    
    A 3x3 tic-tac-toe board. Agent is 'x', opponent is 'o'. There are 9 possible
    actions representing where to put the 'x' (0, 1, 2...8) from left to
    right and top row to bottom in that order (action 3 = 1st col, 2nd row).
    The state of the board is represented by a 9-vector. -1 means blank, 0 means
    agent has put 'x', 1 means opponent has put 'o' for the location corresponding
    to the index in the vector.
    """

    def __init__(self):
        action_space = Discrete(9)
        observation_space = MultiBinary(9)