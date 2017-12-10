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
        self.sim = Sim(n_states=n_states,
                       n_actions=n_actions,
                       n_batch=n_batch,
                       transitions=transitions,
                       rewards=rewards,
                       max_timesteps=max_timesteps)
        self.sim.reset()

    def act(self, states, value_matrix):
        n_batch, = states.shape
        assert self.sim.transitions.shape == (
            self.sim.n_actions, self.sim.n_states, self.sim.n_states), \
            self.sim.transitions.shape
        assert value_matrix.shape == (n_batch, self.sim.n_states)

        meshgrid = np.meshgrid(range(self.sim.n_actions), states)
        assert len(meshgrid) == 2
        for grid in meshgrid:
            assert np.shape(grid) == (n_batch, self.sim.n_actions)

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

    def step(self, states):
        actions = self.act(states, self.value_matrix)
        next_states, reward = self.sim.step(actions, states)
        self.value_matrix = self.update(self.value_matrix, states, next_states)
        return actions, states, next_states, reward


class SingleAgent(Agent):
    def step(self, states):
        actions = self.act(states, self.value_matrix)
        next_states, reward = self.sim.step(actions, states)
        next_states = np.ones_like(next_states) * next_states[0]
        self.value_matrix = self.update(self.value_matrix, states, next_states)
        return actions, states, next_states, reward


class Sim:
    def __init__(self, n_states, n_actions, n_batch, transitions, rewards, max_timesteps):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_batch = n_batch
        self.transitions = transitions
        self.max_timesteps = max_timesteps
        self.rewards = rewards
        self.states = self.timestep = None
        self.reset()

    def reset(self):
        self.states = np.random.choice(self.n_states, self.n_batch)
        self.timestep = 0

    def step(self, actions, states):
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
