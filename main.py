#! /usr/bin/env python

import numpy as np
from scipy.misc import logsumexp

n_states = 15
n_actions = 4
batch_size = 10
transitions = np.random.random([n_states, n_actions, n_states])

def softmax(array, axis=None):
    denominator = logsumexp(array, axis=axis)
    if axis:
        denominator = np.expand_dims(denominator, axis)
    return np.exp(array - denominator)

def next_state_distribution(actions, states, stochastic_actions=False):
    """
    :returns a distribution over next states for the given actions.
    """
    assert transitions.shape == (n_states, n_actions, n_states)
    assert states.shape == batch_size
    if stochastic_actions:
        assert actions.shape == batch_size, n_actions
        return np.dot(actions, transitions)
    else:
        return transitions[states, actions]

def step(actions, states):
    return np.array(list(map(
        np.random.choice, next_state_distribution(actions, states)
        )))

def act(states, value_matrix):
    """
    :returns a distribution over next states for the given actions.
    """
    return np.argmax(
        np.dot(np.dot(actions, transitions[states, :, :]), value_matrix),
        axis=-1
            )

def interpolate(a, b):
    return alpha * a + (1 - alpha) * b

def update(value_matrix, rewards, states, next_states):
    return interpolate(
            value_matrix,
            rewards[states] + gamma * np.dot(value_matrix, next_states)
            )


