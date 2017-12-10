import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.ndimage import gaussian_filter

from algorithm import Agent, SingleAgent, softmax

if __name__ == '__main__':
    n_states = 10
    n_batch = 3
    rewards = np.zeros((n_batch, n_states))
    rewards[range(n_batch), np.random.randint(n_states, size=n_batch)] = 1
    transitions = np.stack([
        np.eye(n_states)[:, np.roll(np.arange(n_states), shift)]
        for shift in [-1, 1]])  # shifted and blurred I matrices
    transitions[[0, 1], [0, n_states - 1], [n_states - 1, 0]] = 0
    transitions[[0, 1], [0, n_states - 1], [0, n_states - 1]] = 1
    transitions = gaussian_filter(transitions, .1)
    agent = SingleAgent(gamma=.95,
                        alpha=.9,
                        n_states=n_states,
                        n_batch=n_batch,
                        n_actions=2,
                        transitions=transitions,
                        rewards=rewards,
                        max_timesteps=5)
    sim = agent.sim


    def x(j):
        return j


    def y(i):
        return sim.n_states - i - 1


    fig = plt.figure()
    ax = plt.axes()

    value_matrix = np.zeros((sim.n_batch, sim.n_states),
                            dtype=np.float)
    im = plt.imshow(value_matrix, vmin=0, vmax=1, cmap='Oranges', animated=True)
    im.set_zorder(0)
    states = np.random.choice(sim.n_states) * np.ones(sim.n_batch, dtype=int)

    next_states = states
    pos = states.astype(float)
    step_size = 0
    circles = []
    texts = []
    for i in range(sim.n_batch):
        color = 'black' if i == 0 else 'gray'
        circle = plt.Circle((states[i], y(i)), radius=0.2, facecolor=color,
                            zorder=1, edgecolor='black')
        circles.append(circle)
        ax.add_patch(circle)
        for j in range(sim.n_states):
            texts.append(ax.text(j, i, int(sim.rewards[i, j]), zorder=2))

    timestep_text = ax.text(.5, 0, 'timestep = {}'.format(0),
                            verticalalignment='bottom',
                            horizontalalignment='center',

                            transform=ax.transAxes, zorder=2)


    def updatefig(_):
        global pos, states, next_states, step_size, agent
        if sim.timestep == sim.max_timesteps:
            sim.reset()
            next_states = states = sim.states.astype(int)
            pos = sim.states.astype(float)
            step_size = 0
        else:
            if np.allclose(pos, next_states):
                # print('before step', pos)
                actions, states, next_states, reward = agent.step(states)
                step_size = (next_states - states) / 30
                # print('step size', step_size)
                states = next_states
                im.set_array(agent.value_matrix)
            pos += step_size
        for i, j in enumerate(pos):
            circles[i].center = (x(j), i)
        timestep_text.set_text('timestep = {}'.format(sim.timestep))
        return [im, timestep_text] + texts + circles


    ani = animation.FuncAnimation(fig, updatefig, interval=.01, blit=True)
    plt.show()
