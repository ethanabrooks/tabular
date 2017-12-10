import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import algorithm

if __name__ == '__main__':
    agent1, value_matrix1, states1, next_states1 = algorithm.init()
    agent2, value_matrix2, states2, next_states2 = algorithm.init()

    def x(j):
        return j


    def y(i):
        return agent1.n_states - i - 1


    fig, (ax1, ax2) = plt.subplots(2)

    for ax, agent, value_matrix, states, next_states in (
            [[ax1, agent1, value_matrix1, states1, next_states1]]):
            # [ax2, agent2, value_matrix2, states2, next_states2]):
        im = ax.imshow(value_matrix, vmin=0, vmax=1, cmap='Oranges', animated=True)
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
        global pos, states1, next_states1, step_size, agent1
        if agent1.timestep == agent1.max_timesteps:
            states1 = agent1.reset()
            next_states1 = states1
            pos = states1.astype(float)
            step_size = 0
            time.sleep(1)
        else:
            if np.allclose(pos, next_states1):
                actions, states1, next_states1, reward = agent1.step(states1)
                step_size = (next_states1 - states1) / 5
                states1 = next_states1
                im.set_array(agent1.value_matrix)
            pos += step_size
        for i, j in enumerate(pos):
            circles[i].center = (x(j), i)
        timestep_text.set_text('timestep = {}'.format(agent1.timestep))
        return [im, timestep_text] + texts + circles


    ani = animation.FuncAnimation(fig, updatefig, interval=.01, blit=True)
    plt.show()
