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
    def __init__(self, gamma, alpha, n_states, n_agents, n_actions,
                 transitions, rewards, terminal, max_timesteps):
        self.gamma = gamma
        self.alpha = alpha
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.transitions = transitions
        self.max_timesteps = max_timesteps
        self.rewards = rewards
        self.value_matrix = rewards

        assert terminal.shape == (n_agents,)
        self.terminal = terminal
        self.reset()

    def reset(self):
        return np.random.choice(self.n_states, size=self.n_agents)

    def act(self, states, value_matrix):
        n_agents, = states.shape
        assert self.transitions.shape == (
            self.n_actions, self.n_states, self.n_states), \
            self.transitions.shape
        assert value_matrix.shape == (n_agents, self.n_states)

        meshgrid = np.meshgrid(range(self.n_actions), states)
        assert len(meshgrid) == 2
        for grid in meshgrid:
            assert np.shape(grid) == (n_agents, self.n_actions)

        next_states = self.transitions[meshgrid]
        assert next_states.shape == (
            n_agents, self.n_actions, self.n_states)

        transposed_value_matrix = np.expand_dims(value_matrix, 2)
        assert transposed_value_matrix.shape == (n_agents, self.n_states, 1)

        next_values = np.matmul(next_states, transposed_value_matrix)
        assert next_values.shape == (n_agents, self.n_actions, 1)

        actions = np.argmax(next_values, axis=1).flatten()
        assert actions.shape == (n_agents,)
        return actions

    def step_sim(self, actions, states):
        n_agents, = actions.shape
        assert states.shape == (n_agents,)

        next_state_distribution = self.transitions[actions, states]
        assert next_state_distribution.shape == (n_agents, self.n_states)

        next_states = np.array([np.random.choice(self.n_states, p=row)
                                for row in next_state_distribution])
        assert next_states.shape == (n_agents,)

        rewards = self.rewards[range(self.n_agents), next_states]
        assert rewards.shape == next_states.shape

        terminal = self.terminal == states  # type: np.ndarray
        assert terminal.shape == states.shape
        next_states[terminal] = states[terminal]
        assert np.array_equal(states[terminal], next_states[terminal])

        return next_states, rewards

    def update(self, value_matrix, states, next_states):
        n_agents, = states.shape
        assert value_matrix.shape == (n_agents, self.n_states)
        assert next_states.shape == (n_agents,)
        assert self.terminal.shape == (n_agents, )

        nonterminal = states != self.terminal
        indexes = np.arange(n_agents)[nonterminal], states[nonterminal]

        rewards = self.rewards[indexes]
        assert rewards.shape == (nonterminal.sum(), )

        next_values = value_matrix[np.arange(n_agents), next_states][nonterminal]
        assert next_values.shape == (nonterminal.sum(), )

        value_matrix[indexes] *= self.alpha
        value_matrix[indexes] += (1 - self.alpha) * (rewards + next_values)
        return value_matrix

    def step(self, states):
        actions = self.act(states, self.value_matrix)
        next_states, reward = self.step_sim(actions, states)
        self.value_matrix = self.update(self.value_matrix, states, next_states)

        terminal = self.terminal == states
        assert np.array_equal(states[terminal], next_states[terminal])
        return actions, next_states, reward


class SingleAgent(Agent):
    def step(self, states):
        actions = self.act(states, self.value_matrix)
        next_states, reward = self.step_sim(actions, states)
        next_states[states != self.terminal] = next_states[0]
        self.value_matrix = self.update(self.value_matrix, states, next_states)

        terminal = self.terminal == states
        assert np.array_equal(states[terminal], next_states[terminal])
        return actions, next_states, reward

    def reset(self):
        return np.random.choice(self.n_states) * np.ones(self.n_agents, dtype=int)


def all_goals_rewards(n_states, rewards):
    goal_ids = np.arange(n_states)
    rewards = np.vstack([rewards, np.eye(n_states)])
    return goal_ids, rewards


def random_rewards(n_states, n_agents, rewards):
    goal_ids = np.random.choice(n_states, n_agents - 1)
    alt_rewards = np.zeros((n_agents, n_states))
    alt_rewards[[range(n_agents - 1), goal_ids]] = 1
    rewards = np.vstack([rewards, alt_rewards])
    return goal_ids, rewards


class OptimizedAgent(Agent):
    def __init__(self, n_states, rewards, **kwargs):
        assert rewards.shape == (n_states, )
        # n_agents = int(n_states / 2)
        n_agents = n_states + 1
        self.goal_ids, rewards = all_goals_rewards(n_states, rewards)
        assert self.goal_ids.shape == (n_agents - 1, )
        assert rewards.shape == (n_agents, n_states), rewards

        terminal = rewards.argmax(axis=1)
        assert terminal.shape == (n_agents, )

        super().__init__(rewards=rewards, terminal=terminal,
                         n_states=n_states, n_agents=n_states + 1, **kwargs)

    def update(self, value_matrix, states, next_states):
        assert self.goal_ids.shape == (self.n_agents - 1,)

        value_matrix = super().update(value_matrix, states, next_states)
        value_matrix = np.minimum(value_matrix, 1)  # TODO: delete
        values1 = value_matrix[range(1, self.n_agents), states[1:]]  # V_g'(s)
        values2 = value_matrix[0, self.goal_ids]  # V_g(g')
        product_values = np.ones((self.n_agents, self.n_states)) * -np.inf
        product_values[range(1, self.n_agents), states[1:]] = values1 * values2  # V_g'(s) * V_g(g')
        product_values[0] = value_matrix[0]
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
        terminal = self.terminal == states
        next_states[np.logical_not(terminal)] = next_states[0]
        self.value_matrix = self.update(self.value_matrix, states, next_states)

        assert np.array_equal(states[terminal], next_states[terminal])
        return actions, next_states, reward

    def reset(self):
        return np.random.choice(self.n_states) * np.ones(self.n_agents,
                                                         dtype=int)


def init():
    np.set_printoptions(precision=2)
    n_states = 10
    n_agents = 3
    rewards = np.zeros((n_agents, n_states))
    rewards[range(n_agents), np.random.randint(n_states, size=n_agents)] = 1
    transitions = np.stack([
        np.eye(n_states)[:, np.roll(np.arange(n_states), shift)]
        for shift in [-1, 1]])  # shifted and blurred I matrices
    transitions[[0, 1], [0, n_states - 1], [n_states - 1, 0]] = 0
    transitions[[0, 1], [0, n_states - 1], [0, n_states - 1]] = 1
    transitions = gaussian_filter(transitions, .3)
    rewards = np.zeros(n_states)
    # rewards[[0, -1]] = [.001, .999]
    rewards[np.random.choice(n_states)] = 1
    agent = OptimizedSingleAgent(
        gamma=.95,
        alpha=.95,
        n_states=n_states,
        n_actions=2,
        transitions=transitions,
        max_timesteps=n_states,
        # n_agents=n_agents,
        rewards=rewards,
    )

    state = np.random.choice(agent.n_states)
    states = state * np.ones(agent.n_agents, dtype=int)
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


def stochastic_stepwise_transitions(sigma, n_states):
    transitions = np.stack([
        np.eye(n_states)[:, np.roll(np.arange(n_states), shift)]
        for shift in [-1, 1]])  # shifted and blurred I matrices
    transitions[[0, 1], [0, n_states - 1], [n_states - 1, 0]] = 0
    transitions[[0, 1], [0, n_states - 1], [0, n_states - 1]] = 1
    return gaussian_filter(transitions, sigma)


def combination_lock_transitions(sigma, n_states):
    left = np.zeros((n_states, n_states))
    left[:, 0] = 1
    transitions = np.stack([
        left,
        np.eye(n_states)[:, np.roll(np.arange(n_states), 1)]
    ])
    return gaussian_filter(transitions, sigma)