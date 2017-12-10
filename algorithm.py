#! /usr/bin/env python

import numpy as np
from scipy.misc import logsumexp


def softmax(array, axis=None):
    denominator = logsumexp(array, axis=axis)
    if axis:
        denominator = np.expand_dims(denominator, axis)
    return np.exp(array - denominator)


class Agent:
    def __init__(self, gamma, alpha, n_states, n_batch, n_actions,
                 transitions, rewards, max_timesteps):
        self.gamma = gamma
        self.alpha = alpha
        self.value_matrix = np.zeros((n_batch, n_states))
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_batch = n_batch
        self.transitions = transitions
        self.max_timesteps = max_timesteps
        self.rewards = rewards
        self.timestep = None
        self.reset()

    def reset(self):
        self.timestep = 0
        return np.random.choice(self.n_states, size=self.n_batch)

    def step_sim(self, actions, states):
        self.timestep += 1
        n_batch, = actions.shape
        assert states.shape == (n_batch,)

        next_state_distribution = self.transitions[actions, states]
        assert next_state_distribution.shape == (n_batch, self.n_states)

        next_states = np.array([np.random.choice(self.n_states, p=row)
                                for row in next_state_distribution])
        assert next_states.shape == (n_batch,)

        rewards = self.rewards[range(self.n_batch), next_states]
        assert rewards.shape == next_states.shape

        return next_states, rewards

    def act(self, states, value_matrix):
        n_batch, = states.shape
        assert self.transitions.shape == (
            self.n_actions, self.n_states, self.n_states), \
            self.transitions.shape
        assert value_matrix.shape == (n_batch, self.n_states)

        meshgrid = np.meshgrid(range(self.n_actions), states)
        assert len(meshgrid) == 2
        for grid in meshgrid:
            assert np.shape(grid) == (n_batch, self.n_actions)

        next_states = self.transitions[meshgrid]
        assert next_states.shape == (
            n_batch, self.n_actions, self.n_states)

        transposed_value_matrix = np.expand_dims(value_matrix, 2)
        assert transposed_value_matrix.shape == (n_batch, self.n_states, 1)

        next_values = np.matmul(next_states, transposed_value_matrix)
        assert next_values.shape == (n_batch, self.n_actions, 1)

        actions = np.argmax(next_values, axis=1).flatten()
        assert actions.shape == (n_batch,)
        return actions

    def update(self, value_matrix, states, next_states):
        n_batch, = states.shape
        assert value_matrix.shape == (n_batch, self.n_states)
        assert next_states.shape == (n_batch,)

        rewards = self.rewards[range(self.n_batch), states]
        assert rewards.shape == states.shape

        next_values = value_matrix[range(n_batch), next_states]
        assert next_values.shape == states.shape

        value_matrix *= self.alpha
        value_matrix[range(n_batch), states] += (1 - self.alpha) * (
            rewards + next_values)
        return value_matrix

    def step(self, states):
        actions = self.act(states, self.value_matrix)
        next_states, reward = self.step_sim(actions, states)
        self.value_matrix = self.update(self.value_matrix, states, next_states)
        return actions, states, next_states, reward


class SingleAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, states):
        actions = self.act(states, self.value_matrix)
        next_states, reward = self.step_sim(actions, states)
        next_states = np.ones_like(next_states) * next_states[0]
        self.value_matrix = self.update(self.value_matrix, states, next_states)
        return actions, states, next_states, reward

    def reset(self):
        self.timestep = 0
        return np.random.choice(self.n_states) * np.ones(self.n_batch,
                                                         dtype=int)
