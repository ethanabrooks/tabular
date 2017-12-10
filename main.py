import time

import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from matplotlib import animation

import algorithm


def updatefig(ax, agent, states, speed):
    im = ax.imshow(agent.value_matrix, vmin=0, vmax=1,
                   cmap='Oranges', animated=True)
    im.set_zorder(0)
    timestep_text = ax.text(.5, 0, 'timestep = {}'.format(0),
                            verticalalignment='bottom',
                            horizontalalignment='center',
                            transform=ax.transAxes, zorder=2)
    circles = []
    texts = []
    for i in range(agent.n_batch):
        for j in range(agent.n_states):
            texts.append(ax.text(j, i, int(agent.rewards[i, j]), zorder=2))
        y = agent.n_states - i - 1
        color = 'gray' if i == 0 else 'lightgray'
        circle = plt.Circle((states[i], y), radius=0.2, facecolor=color,
                            zorder=1, edgecolor='white')
        circles.append(circle)
        ax.add_patch(circle)

    while True:
        states = agent.reset()
        pos = states.astype(float)
        for timestep in range(agent.max_timesteps):
            timestep_text.set_text('timestep = {}'.format(agent.timestep))
            im.set_array(agent.value_matrix)
            actions, next_states, reward = agent.step(states)
            step_size = (next_states - states) * speed
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
    fig, (ax1, ax2) = plt.subplots(2)
    speed = 1 / 20

    def animate(ax):
        frames = partial(updatefig, ax, *algorithm.init(), speed)
        return animation.FuncAnimation(fig, identity, frames,
                                       interval=.01, blit=True)
    a1 = animate(ax1)
    a2 = animate(ax2)
    plt.show()
