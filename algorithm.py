#! /usr/bin/env python

import numpy as np
from scipy.misc import logsumexp
from scipy.ndimage import gaussian_filter


def softmax(array, axis=None):
    denominator = logsumexp(array, axis=axis)
    if axis:
        denominator = np.expand_dims(denominator, axis)
    return np.exp(array - denominator)


class Agent:
    def __init__(self, gamma, alpha, n_states, n_batch, n_actions,
                 transitions, rewards, episodes, max_timesteps):
        self.gamma = gamma
        self.alpha = alpha
        self.episodes = episodes
        self.max_timesteps = max_timesteps
        self.states = np.random.choice(n_states, n_batch)
        self.value_matrix = np.zeros((n_batch, n_states))
        self.sim = Sim(n_states=n_states,
                       n_actions=n_actions,
                       n_batch=n_batch,
                       transitions=transitions,
                       rewards=rewards)

    def act(self, states, value_matrix):
        n_batch, = states.shape
        assert self.sim.transitions.shape == (
            self.sim.n_actions, self.sim.n_states, self.sim.n_states), \
            self.sim.transitions.shape
        assert value_matrix.shape == (n_batch, self.sim.n_states)

        meshgrid = np.meshgrid(range(self.sim.n_actions), states)
        assert len(meshgrid) == 2
        for grid in meshgrid:
            assert grid.shape == (n_batch, self.sim.n_actions)

        next_states = self.sim.transitions[meshgrid]
        assert next_states.shape == (
        n_batch, self.sim.n_actions, self.sim.n_states)

        transposed_value_matrix = np.expand_dims(value_matrix, 2)
        assert transposed_value_matrix.shape == (n_batch, self.sim.n_states, 1)

        next_values = np.matmul(next_states, transposed_value_matrix)
        assert next_values.shape == (n_batch, self.sim.n_actions, 1)

        actions = np.argmax(next_values, axis=1).flatten()
        assert actions.shape == (n_batch,)
        return actions

    def update(self, value_matrix, states, next_states):
        n_batch, = states.shape
        assert value_matrix.shape == (n_batch, self.sim.n_states)
        assert next_states.shape == (n_batch,)

        rewards = self.sim.rewards[range(self.sim.n_batch), states]
        assert rewards.shape == states.shape

        next_values = value_matrix[range(n_batch), next_states]
        assert next_values.shape == states.shape

        value_matrix *= self.alpha
        value_matrix[range(n_batch), states] += (1 - self.alpha) * (
            rewards + next_values)
        return value_matrix


class Sim:
    def __init__(self, n_states, n_actions, n_batch, transitions, rewards):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_batch = n_batch
        self.transitions = transitions
        self.rewards = rewards

    def step(self, actions, states):
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


if __name__ == '__main__':
    n_states = n_batch = 4
    rewards = np.zeros((n_batch, n_states))
    rewards[range(n_batch), np.random.randint(n_states, size=n_batch)] = 1
    transitions = np.stack([gaussian_filter(
        np.eye(n_states)[:, np.roll(np.arange(n_states), shift)], .5)
        for shift in [-1, 1]])  # shifted and blurred I matrices
    sim = Sim(n_states=n_states,
              n_actions=2,
              transitions=transitions,
              rewards=rewards)

    for _ in range(sim.episodes):
        for _ in range(sim.max_timesteps):
            actions = sim.act(sim.states, sim.value_matrix)
            next_states, reward = sim.step(actions, sim.states)
            value_matrix = sim.update(sim.value_matrix, sim.states, next_states)
            sim.states = next_states
