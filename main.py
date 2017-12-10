import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import algorithm


class Animation:
    def __init__(self, ax, agent, states, speed=1 / 20):
        assert speed < 1
        self.speed = speed
        self.agent = agent
        self.ax = ax
        self.im = ax.imshow(agent.value_matrix, vmin=0, vmax=1, cmap='Oranges',
                            animated=True)
        self.im.set_zorder(0)
        self.next_states = states
        self.pos = states.astype(float)
        self.step_size = 0
        self.circles = []
        self.texts = []
        for i in range(agent.n_batch):
            color = 'black' if i == 0 else 'gray'
            y = self.agent.n_states - i - 1
            circle = plt.Circle((states[i], y), radius=0.2, facecolor=color,
                                zorder=1, edgecolor='black')
            self.circles.append(circle)
            ax.add_patch(circle)
            for j in range(agent.n_states):
                self.texts.append(
                    ax.text(j, i, int(agent.rewards[i, j]), zorder=2))

        self.timestep_text = ax.text(.5, 0, 'timestep = {}'.format(0),
                                     verticalalignment='bottom',
                                     horizontalalignment='center',
                                     transform=ax.transAxes, zorder=2)

    def updatefig(self, _):
        pos = [circle.center[0] for circle in self.circles]
        if self.agent.timestep == self.agent.max_timesteps:
            states = self.agent.reset()
            self.next_states = states
            pos = states.astype(float)
            self.step_size = 0
            time.sleep(1)
        else:
            if np.allclose(pos, self.next_states):
                states = self.next_states
                actions, self.next_states, reward = self.agent.step(states)
                self.step_size = (self.next_states - states) * self.speed
                self.im.set_array(self.agent.value_matrix)
            pos += self.step_size
        for i, j in enumerate(pos):
            self.circles[i].center = (j, i)
        self.timestep_text.set_text('timestep = {}'.format(self.agent.timestep))
        return [self.im, self.timestep_text] + self.texts + self.circles

    def animate(self, fig):
        return animation.FuncAnimation(fig, self.updatefig, interval=.01,
                                       blit=True)


if __name__ == '__main__':
    agent1, value_matrix1, states1, next_states1 = algorithm.init()
    agent2, value_matrix2, states2, next_states2 = algorithm.init()
    fig, (ax1, ax2) = plt.subplots(2)
    _ = Animation(ax2, agent2, states2).animate(fig)
    _ = Animation(ax1, agent1, states1).animate(fig)
    plt.show()
