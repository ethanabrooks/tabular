import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.ndimage import gaussian_filter

from algorithm import Sim



n_states = n_batch = 4
rewards = np.zeros((n_batch, n_states))
rewards[range(n_batch), np.random.randint(n_states, size=n_batch)] = 1
transitions = np.stack([gaussian_filter(
    np.eye(n_states)[:, np.roll(np.arange(n_states), shift)], .5)
    for shift in [-1, 1]])  # shifted and blurred I matrices
sim = Sim(gamma=.95,
          alpha=.9,
          n_states=n_states,
          n_batch=n_batch,
          n_actions=2,
          transitions=transitions,
          rewards=rewards,
          episodes=200,
          max_timesteps=100)


def x(j):
    return j


def y(i):
    return sim.n_states - i - 1


fig = plt.figure()
ax = plt.axes()


value_matrix = np.zeros((sim.n_batch, sim.n_states), dtype=np.float)
im = plt.imshow(value_matrix, vmin=0, vmax=1, cmap='Oranges', animated=True)
im.set_zorder(0)
states = np.random.choice(sim.n_states) * np.ones(sim.n_states, dtype=int)
next_states = states
pos = states.astype(float)
step_size = 0
circles = []
texts = []
for i in range(sim.n_batch):
    circle = plt.Circle((states[i], y(i)), radius=0.2, color='black', zorder=1)
    circles.append(circle)
    ax.add_patch(circle)
    for j in range(sim.n_states):
        texts.append(ax.text(j, i, int(sim.rewards[i, j]), zorder=2))


def updatefig(_):
    global pos, states, next_states, value_matrix, step_size
    if np.allclose(pos, next_states):
        actions = sim.act(states, value_matrix)
        next_states, reward = sim.step(actions, states)
        value_matrix = sim.update(value_matrix, states, next_states)
        step_size = (next_states - states) / 20
        states = next_states
        im.set_array(value_matrix)
    pos += step_size
    for i, j in enumerate(pos):
        circles[i].center = (x(j), i)
    return [im] + texts + circles

ani = animation.FuncAnimation(fig, updatefig, interval=.01, blit=True)
plt.show()
