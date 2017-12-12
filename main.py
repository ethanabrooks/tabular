import time
from functools import partial
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import algorithm
from algorithm import stochastic_stepwise_transitions


def circle_color(i, terminal):
    if terminal:
        return 'white'
    elif i == 0:
        return 'black'
    else:
        return 'gray'


def updatefig(ax, agent, states, speed):
    assert states.shape == (agent.n_agents,), \
        (states.shape, agent.n_agents)
    im = ax.imshow(agent.value_matrix, vmin=0, vmax=1,
                   cmap='Oranges', animated=True)
    im.set_zorder(0)
    timestep_text = ax.text(.5, 0, '',
                            verticalalignment='bottom',
                            horizontalalignment='center',
                            transform=ax.transAxes, zorder=2)

    circles = []
    texts = []
    for i in range(agent.n_agents):
        for j in range(agent.n_states):
            texts.append(ax.text(j, i, int(round(agent.rewards[i, j])),
                                 zorder=2))
        y = agent.n_states - i - 1
        circle = plt.Circle((states[i], y), radius=0.2,
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
            for i, (state, terminal, circle) in enumerate(zip(states, agent.terminal, circles)):
                circle.set_facecolor(circle_color(i, state == terminal))

            actions, next_states, reward = agent.step(states)
            terminal = states == agent.terminal

            total_reward += reward[0]
            step_size = (next_states - states) * speed
            assert np.all(step_size[terminal] == 0)

            states = next_states
            while not np.allclose(pos, next_states):
                pos += step_size
                for i, j in enumerate(pos):
                    circles[i].center = (j, i)
                yield [im, timestep_text] + texts + circles
        time.sleep(.5)


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
        alpha=.5,
        n_states=n_states,
        n_actions=2,
        transitions=transitions,
        max_timesteps=15,
        # n_agents=n_agents,
        rewards=rewards,
    )

    agent2 = algorithm.Agent(
        gamma=.95,
        alpha=.95,
        n_states=n_states,
        n_agents=1,
        n_actions=2,
        transitions=transitions,
        max_timesteps=n_states,
        rewards=agent1.rewards[[0]],
        terminal=agent1.rewards[[0]].argmax(axis=1),
    )
    # agent2 = algorithm.OptimizedAgent( gamma=.95,
    #     alpha=.95,
    #     n_states=n_states,
    #     n_actions=2,
    #     transitions=transitions,
    #     max_timesteps=15,
    #     # n_agents=n_agents,
    #     rewards=agent1.rewards[0],
    # )

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title('Optimized Agent')
    ax2.set_title('Baseline Agent')
    speed = 1 / 20


    def animate():
        ax1.axis('off')
        ax1.set_ylim([-1, agent1.n_agents])
        states = agent1.reset()
        states2 = agent2.reset()
        while True:
            l1 = next(updatefig(ax1, agent1, states, speed))
            l2 = next(updatefig(ax2, agent2, states2, speed))
            yield l1 + l2

    a1 = animation.FuncAnimation(fig, identity, animate,
                                 interval=1, blit=True)
    # a2 = animate(ax2, agent2)

    plt.show()
