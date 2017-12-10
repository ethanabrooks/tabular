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
        self.reset()

    def reset(self):
        return np.random.choice(self.n_states, size=self.n_batch)

    def step_sim(self, actions, states):
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

        indexes = range(n_batch), states

        rewards = self.rewards[indexes]
        assert rewards.shape == states.shape

        next_values = value_matrix[range(n_batch), next_states]
        assert next_values.shape == states.shape

        value_matrix[indexes] *= self.alpha
        value_matrix[indexes] += (1 - self.alpha) * (
            rewards + next_values)
        return value_matrix

    def step(self, states):
        actions = self.act(states, self.value_matrix)
        next_states, reward = self.step_sim(actions, states)
        self.value_matrix = self.update(self.value_matrix, states, next_states)
        return actions, next_states, reward


class SingleAgent(Agent):
    def step(self, states):
        actions = self.act(states, self.value_matrix)
        next_states, reward = self.step_sim(actions, states)
        next_states = np.ones_like(next_states) * next_states[0]
        self.value_matrix = self.update(self.value_matrix, states, next_states)
        return actions, next_states, reward

    def reset(self):
        return np.random.choice(self.n_states) * np.ones(self.n_batch,
                                                         dtype=int)


class OptimizedAgent(Agent):
    def __init__(self, n_states, rewards, **kwargs):
        self.goal_ids = np.arange(n_states)
        n_batch = int(n_states + 1)
        self.goal_ids = np.random.choice(n_states, n_batch)
        primary_goal = np.random.choice(n_batch)
        self.goal_ids[[0, primary_goal]] = self.goal_ids[[primary_goal, 0]]
        # rewards = np.vstack([np.eye(n_states), rewards])
        if rewards is None:
            rewards = np.zeros((n_batch, n_states))
            rewards[[range(n_batch), self.goal_ids]] = 1
        super().__init__(rewards=rewards, n_states=n_states,
                         n_batch=n_batch, **kwargs)

    def update(self, value_matrix, states, next_states):
        value_matrix = super().update(value_matrix, states, next_states)
        value_matrix = np.minimum(value_matrix, 1)
        values1 = value_matrix[range(self.n_batch), states]
        values2 = value_matrix[[0] * self.n_batch, self.goal_ids]
        product_values = np.ones((self.n_batch + 1, self.n_states)) * -np.inf
        product_values[range(self.n_batch), states] = values1 * values2
        product_values[self.n_batch] = value_matrix[0]
        value_matrix[0] = product_values.max(axis=0)
        return value_matrix


class OptimizedSingleAgent(OptimizedAgent):
    # def update(self, value_matrix, states, next_states):
    #     value_matrix = super().update(value_matrix, states, next_states)
    #     value_matrix = np.minimum(value_matrix, 1)
    #     values1 = value_matrix[range(self.n_states), states]
    #     values2 = value_matrix[[0] * self.n_states, self.goal_ids]
    #     product_values = np.zeros((self.n_states + 1, self.n_states))
    #     product_values[range(self.n_states), states] = values1 * values2
    #     product_values[self.n_states] = value_matrix[0]
    #     value_matrix[0] = product_values.sum(axis=0)
    #     return value_matrix

    def step(self, states):
        actions = self.act(states, self.value_matrix)
        next_states, reward = self.step_sim(actions, states)
        next_states = np.ones_like(next_states) * next_states[0]
        self.value_matrix = self.update(self.value_matrix, states, next_states)
        return actions, next_states, reward

    def reset(self):
        return np.random.choice(self.n_states) * np.ones(self.n_batch,
                                                         dtype=int)


def init():
    np.set_printoptions(precision=2)
    n_states = 10
    n_batch = 3
    rewards = np.zeros((n_batch, n_states))
    rewards[range(n_batch), np.random.randint(n_states, size=n_batch)] = 1
    transitions = np.stack([
        np.eye(n_states)[:, np.roll(np.arange(n_states), shift)]
        for shift in [-1, 1]])  # shifted and blurred I matrices
    transitions[[0, 1], [0, n_states - 1], [n_states - 1, 0]] = 0
    transitions[[0, 1], [0, n_states - 1], [0, n_states - 1]] = 1
    transitions = gaussian_filter(transitions, .3)
    agent = OptimizedAgent(gamma=.95,
                           alpha=.95,
                           n_states=n_states,
                           n_actions=2,
                           transitions=transitions,
                           max_timesteps=n_states,
                           # n_batch=n_batch,
                           # rewards=rewards,
                           )

    state = np.random.choice(agent.n_states)
    states = state * np.ones(agent.n_batch, dtype=int)
    return agent, states


if __name__ == '__main__':
    agent1, states1 = init()
    timestep = 0
    while True:
        if timestep == agent1.max_timesteps:
            states1 = agent1.reset()
            timestep = 0
        else:
            actions, states1, reward = agent1.step(states1)
            timestep += 1
