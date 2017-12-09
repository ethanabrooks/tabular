#! /usr/bin/env python

import numpy as np
from scipy.misc import logsumexp
from scipy.ndimage import gaussian_filter


def softmax(array, axis=None):
    denominator = logsumexp(array, axis=axis)
    if axis:
        denominator = np.expand_dims(denominator, axis)
    return np.exp(array - denominator)


GAMMA = .95
ALPHA = .9
N_STATES = 4
N_BATCH = 4
N_ACTIONS = 2
TRANSITIONS = np.stack([gaussian_filter(
    np.eye(N_STATES)[:, np.roll(np.arange(N_STATES), shift)], .5)
    for shift in [-1, 1]])  # shifted and blurred I matrices
REWARDS = np.random.choice(2, N_STATES,
                           p=np.array([N_STATES - 2, 2]) / N_STATES)
EPISODES = 200
MAX_TIMESTEPS = 100


def step(actions, states):
    n_batch, = actions.shape
    assert states.shape == (n_batch,)

    next_state_distribution = TRANSITIONS[actions, states]
    assert next_state_distribution.shape == (n_batch, N_STATES)

    next_states = np.array([np.random.choice(N_STATES, p=row)
                            for row in next_state_distribution])
    assert next_states.shape == (n_batch,)

    rewards = REWARDS[next_states]
    assert rewards.shape == next_states.shape

    return next_states, rewards


def act(states, value_matrix):
    n_batch, = states.shape
    assert TRANSITIONS.shape == (N_ACTIONS, N_STATES, N_STATES),\
        TRANSITIONS.shape
    assert value_matrix.shape == (n_batch, N_STATES)

    meshgrid = np.meshgrid(range(N_ACTIONS), states)
    assert len(meshgrid) == 2
    for grid in meshgrid:
        assert grid.shape == (n_batch, N_ACTIONS)

    next_states = TRANSITIONS[meshgrid]
    assert next_states.shape == (n_batch, N_ACTIONS, N_STATES)

    transposed_value_matrix = np.expand_dims(value_matrix, 2)
    assert transposed_value_matrix.shape == (n_batch, N_STATES, 1)

    next_values = np.matmul(next_states, transposed_value_matrix)
    assert next_values.shape == (n_batch, N_ACTIONS, 1)

    actions = np.argmax(next_values, axis=1).flatten()
    assert actions.shape == (n_batch,)
    return actions


def update(value_matrix, states, next_states):
    n_batch, = states.shape
    assert value_matrix.shape == (n_batch, N_STATES)
    assert next_states.shape == (n_batch,)

    indexes = np.arange(n_batch), next_states

    rewards = REWARDS[states]
    assert rewards.shape == states.shape

    next_values = value_matrix[indexes]
    assert next_values.shape == states.shape

    value_matrix *= ALPHA
    value_matrix[indexes] += (1 - ALPHA) * (rewards + next_values)
    return value_matrix


if __name__ == '__main__':
    value_matrix = np.zeros((N_BATCH, N_STATES))
    states = np.random.choice(N_STATES, N_BATCH)
    for _ in range(EPISODES):
        cum_reward = np.zeros(N_BATCH)
        for _ in range(MAX_TIMESTEPS):
            actions = act(states, value_matrix)
            next_states, reward = step(actions, states)
            cum_reward += reward
            value_matrix = update(value_matrix, states, next_states)
            print(value_matrix, REWARDS)
            states = next_states[next_states != None]
        print(cum_reward.mean())

