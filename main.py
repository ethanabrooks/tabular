import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import algorithm

if __name__ == '__main__':
    agent, value_matrix, states, next_states = algorithm.init()

    def x(j):
        return j


    def y(i):
        return agent.n_states - i - 1


    fig = plt.figure()
    ax = plt.axes()

    im = plt.imshow(value_matrix, vmin=0, vmax=1, cmap='Oranges', animated=True)
    im.set_zorder(0)
    pos = states.astype(float)
    step_size = 0
    circles = []
    texts = []
    for i in range(agent.n_batch):
        color = 'black' if i == 0 else 'gray'
        circle = plt.Circle((states[i], y(i)), radius=0.2, facecolor=color,
                            zorder=1, edgecolor='black')
        circles.append(circle)
        ax.add_patch(circle)
        for j in range(agent.n_states):
            texts.append(ax.text(j, i, int(agent.rewards[i, j]), zorder=2))

    timestep_text = ax.text(.5, 0, 'timestep = {}'.format(0),
                            verticalalignment='bottom',
                            horizontalalignment='center',

                            transform=ax.transAxes, zorder=2)


    def updatefig(_):
        global pos, states, next_states, step_size, agent
        if agent.timestep == agent.max_timesteps:
            states = agent.reset()
            next_states = states.astype(int)
            pos = states.astype(float)
            step_size = 0
            time.sleep(1)
        else:
            if np.allclose(pos, next_states):
                # print('before step', pos)
                actions, states, next_states, reward = agent.step(states)
                step_size = (next_states - states) / 10
                # print('step size', step_size)
                states = next_states
                im.set_array(agent.value_matrix)
            pos += step_size
        for i, j in enumerate(pos):
            circles[i].center = (x(j), i)
        timestep_text.set_text('timestep = {}'.format(agent.timestep))
        return [im, timestep_text] + texts + circles


    ani = animation.FuncAnimation(fig, updatefig, interval=.01, blit=True)
    plt.show()
