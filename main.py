import time
from functools import partial
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.ndimage import gaussian_filter

import algorithm


def updatefig(ax, agent, states, speed):
    assert states.shape == (agent.n_batch,), \
        (states.shape, agent.n_batch)
    im = ax.imshow(agent.value_matrix, vmin=0, vmax=1,
                   cmap='Oranges', animated=True)
    im.set_zorder(0)
    timestep_text = ax.text(.5, 0, '',
                            verticalalignment='bottom',
                            horizontalalignment='center',
                            transform=ax.transAxes, zorder=2)

    circles = []
    texts = []
    for i in range(agent.n_batch):
        for j in range(agent.n_states):
            texts.append(ax.text(j, i, int(round(agent.rewards[i, j])),
                                 zorder=2))
        y = agent.n_states - i - 1
        color = 'gray' if i == 0 else 'lightgray'
        circle = plt.Circle((states[i], y), radius=0.2, facecolor=color,
                            zorder=1, edgecolor='white')
        circles.append(circle)
        ax.add_patch(circle)

    total_reward = 0
    for episode in itertools.count(1):
        states = agent.reset()
        pos = states.astype(float)
        for timestep in range(agent.max_timesteps):
            avg_reward = round(total_reward / float(episode), ndigits=2)
            timestep_text.set_text(
                'timestep: {}, avg. reward: {}'.format(
                    timestep, avg_reward))
            im.set_array(agent.value_matrix)
            actions, next_states, reward = agent.step(states)
            total_reward += reward[0]
            step_size = (next_states - states) * speed
            states = next_states
            while not np.allclose(pos, next_states):
                pos += step_size
                for i, j in enumerate(pos):
                    circles[i].center = (j, i)
                yield [im, timestep_text] + texts + circles
        time.sleep(.5)


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


def identity(x):
    return x


if __name__ == '__main__':
    np.set_printoptions(precision=1)
    n_states = 10
    transitions = stochastic_stepwise_transitions(sigma=.5, n_states=n_states)
    rewards = np.zeros(n_states)
    # rewards[[0, -1]] = [.001, .999]
    rewards[np.random.choice(n_states)] = 1
    agent1 = algorithm.OptimizedSingleAgent(
        gamma=.95,
        alpha=.95,
        n_states=n_states,
        n_actions=2,
        transitions=transitions,
        max_timesteps=n_states,
        # n_batch=n_batch,
        rewards=rewards,
    )

    # agent2 = algorithm.Agent(
    #     gamma=.95,
    #     alpha=.95,
    #     n_states=n_states,
    #     n_batch=1,
    #     n_actions=2,
    #     transitions=transitions,
    #     max_timesteps=n_states,
    #     rewards=agent1.rewards,
    # )
    agent2 = algorithm.OptimizedAgent(
        gamma=.95,
        alpha=.95,
        n_states=n_states,
        n_actions=2,
        transitions=transitions,
        max_timesteps=n_states,
        # n_batch=n_batch,
        rewards=agent1.rewards,
    )

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title('Optimized Agent')
    ax2.set_title('Baseline Agent')
    speed = 1 / 4


    def animate(ax, agent):
        ax.axis('off')
        ax.set_ylim([-1, agent.n_batch])
        states = agent.reset()
        frames = partial(updatefig, ax, agent, states, speed)
        return animation.FuncAnimation(fig, identity, frames,
                                       interval=.1, blit=True)


    a1 = animate(ax1, agent1)
    a2 = animate(ax2, agent2)

    plt.show()
