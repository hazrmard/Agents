"""
Defines functions to compute maximum of a function over a discrete or
continuous space.

The functions should be of the form:

    [[argument]]    =>  f   => [[result]]

i.e. all inputs and outputs in a 2D array (see Approximators)
"""

from typing import Callable, Tuple, Iterable, Union

import numpy as np
from scipy.optimize import minimize



def max_discrete(func: Callable[[Tuple], np.ndarray], over: Iterable[Tuple],\
    state: Tuple[Union[int, float]]) -> Tuple[float, Tuple[Union[int, float]], None]:
    """
    Calculates the maximum value of a function over a discrete space.
    The function is of the form `f([[state,...,action]]) -> [[float]]`.

    Args:
    * func: The function that accepts a tuple/2D array of arguments and returns
    a float to maximize.
    * over: An iterable of argument tuples to maximize over. I.e. the iterable
    enumerates all the arguments to compare.
    * state: The prefix argument i.e. the state over which to explore action space.

    Returns a tuple of:
    * The maximum value,
    * The corresponding argument tuple.
    * None
    """
    vals = [func(np.asarray((*state, *action)).reshape(1, -1))[0, 0] for action in over]
    maximum = max(vals)
    return (maximum, over[vals.index(maximum)], None)



def max_continuous(func: Callable[[Tuple], np.ndarray], over: Iterable[Tuple],\
    state: Tuple[Union[int, float]]) -> Tuple[float, Tuple[Union[int, float]], None]:
    """
    Calculates the maximum value of a function over a continuous
    space. The function is of the form `f([[state,...,action]]) -> [[float]]`.

    Args:
    * func: The function that accepts a tuple of arguments and returns a float
    to maximize.
    * over: An iterable of tuples describing the "box" to maximize over. The
    number of tuples should equal the number of arguments given to function.
    For e.g.: func = f(x1, x2) will have over=((x1min, x1max), (x2min, x2max))
    * state: The prefix argument i.e. the state over which to explore action space.

    Returns a tuple of:
    * The maximum value,
    * The corresponding argument tuple.
    * None
    """
    statebounds = tuple(zip(state, state))
    init = tuple([*state, *np.random.uniform(*zip(*over))])
    funcarg = lambda x: -func(np.asarray(x).reshape(1,-1))[0, 0]
    res = minimize(funcarg, x0=init, bounds=(*statebounds, *over))
    return (-funcarg(res.x), tuple(res.x[len(state):]), None)



def max_hybrid(func: Callable[[Tuple], np.ndarray], over: Tuple[Tuple],\
    state: Tuple[Union[int, float]], cont: Tuple[bool],\
    actions: Iterable[Tuple]) -> Tuple[float, Tuple[Union[int, float]], None]:
    """
    Calculates the maximum value of a function over a discrete-continuous hybrid
    space. The function is of the form `f([[state,...,action]]) -> [[float]]`.

    Args:
    * func: The function that accepts a tuple of arguments and returns a float
    to maximize.
    * over: An iterable of tuples describing the "box" to maximize over. The
    number of tuples should equal the number of arguments given to function.
    For e.g.: func = f(x1, x2) will have over=((x1min, x1max), (x2min, x2max))
    * cont: For each variable in action space, True if continuous, else False.
    * state: The prefix argument i.e. the state over which to explore action space.
    * actions: An iterable of tuples of actions. Result of `enumerate_discrete_space`.

    Returns a tuple of:
    * The maximum value,
    * The corresponding action argument tuple.
    * None
    """
    best = -np.inf
    bestarg = None
    funcarg = lambda x: -func(np.asarray(x).reshape(1, -1))[0, 0]
    statebounds = tuple(zip(state, state))
    for act in actions:
        actbounds = [b if c else (a, a) for a, c, b in zip(act, cont, over)]
        init = tuple([*state, *np.random.uniform(*zip(*over))])
        res = minimize(funcarg, x0=init, bounds=(*statebounds, *actbounds))
        val = -funcarg(res.x)
        if val > best:
            best = val
            bestarg = res.x
    return (best, tuple([float(v) if c else int(v) for v, c in \
                            zip(bestarg[len(state):], cont)]), None)



def max_array(func: Callable[[Tuple], np.ndarray], state: Tuple[Union[int, float]]) \
    -> Tuple[float, Tuple[int], np.ndarray]:
    """
    Calcualtes the maximum value and corresponding index/argument over a
    supplied 1D array. The function is of the form `f([[state,...,action]]) -> [[float]]`.
    That is, computes the maximum value for a function that returns an array of
    values.

    Returns a tuple of:
    * The maximum value,
    * The corresponding action argument tuple,
    * The whole array (1D).
    """
    arr = func((state,)).ravel()
    max_index = np.argmax(arr)
    return (arr[max_index], (max_index,), arr)
