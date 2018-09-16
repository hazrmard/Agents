"""
Implements the standard q-learning algorithm, known as TD(0) for a single
episode. Q-learning is an off-policy temporal difference learning algorithm.
"""
import numpy as np

from ..helpers.spaces import to_space, to_tuple


def q(agent: 'Agent', memory: 'Memory', discount: float, maxsteps: int=np.inf,\
    **kwargs) -> float:
    """
    Q-learning: Off-policy Temporal difference learning with no look-ahead.
    Uses value iteration to learn policy. Value function is incrementaly learned.
    New estimate of value is:

        `Q'(s, a) = reward + discount * max_{a'}Q(s', a')`

    Note: Temporal difference methods with off-policy and non-tabular value
    function approximations may not converge [4.2 Ch. 11.3 - Deadly Triad].

    Args:
    * agent: The agent calling the learning function.
    * memory: A Memory instance that can store and sample past observations.
    * discount: The discount level for future rewards. Between 0 and 1.
    * maxsteps: Number of steps at most to take if episode continues.
    * kwargs: All other keyword arguments discarded silently.
    """
    state = to_tuple(agent.env.observation_space, agent.env.reset())
    rewards = []        # history of rewards
    done = False
    t = 0   # keeping track of steps

    # preallocate arrays for states (X) -> value targets (Y) for approximator
    batchX = np.zeros((memory.batchsize, agent.value.indim))
    batchY = np.zeros((memory.batchsize, agent.value.outdim))

    while (not done) and (t < maxsteps):
        # select exploratory action
        action = agent.next_action(state)
        # observe next state and rewards
        nstate, reward, done, _ = agent.env.step(to_space(agent.env.action_space, action))
        nstate = to_tuple(agent.env.observation_space, nstate)

        # memorize experience
        memory.append((state, action, reward, nstate))

        # Replay experience from memory and calculate new estimate of return.
        # An approximator with outdim=1, the output value will only be for
        # the action taken (since it is passed as argument). So only that
        # action value needs to be updated.
        if agent.value.outdim == 1:
            for i, (s, a, r, ns) in enumerate(memory.sample()):
                # calculate new estimate of return
                nvalue, _, _ = agent.maximum(ns)
                ret = r + discount * nvalue
                # fill batch with state/actions -> values
                batchX[i] = [*s, *a]
                batchY[i] = ret
        # An approximator with outdim > 1 will return a multi-column row
        # vector which has action values for all actions. So a new vector
        # is constructed which only updates the estimate for the action
        # taken and uses prior estimates for other actions.
        else:
             for i, (s, a, r, ns) in enumerate(memory.sample()):
                # calculate new estimate of return
                nvalue, _, predictions = agent.maximum(ns)
                ret = r + discount * nvalue
                # fill batch with states -> action values
                batchX[i] = s
                batchY[i] = predictions
                batchY[i, a] = ret

        # update value function with new estimate
        agent.value.update(batchX, batchY)
        state = nstate
        rewards.append(reward)
        t += 1

    return np.sum(rewards)
