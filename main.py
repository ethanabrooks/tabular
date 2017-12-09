import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from algorithm import N_BATCH, N_STATES, act, step, \
    update, REWARDS


def x(j):
    return j


def y(i):
    return N_STATES - i - 1


fig = plt.figure()
ax = plt.axes()

value_matrix = np.zeros((N_BATCH, N_STATES), dtype=np.float)
im = plt.imshow(value_matrix, vmin=0, vmax=1, cmap='Oranges', animated=True)
im.set_zorder(0)
states = np.random.choice(N_STATES, N_BATCH)
next_states = states
pos = states.astype(float)
step_size = 0
circles = []
texts = []
for i in range(N_BATCH):
    circle = plt.Circle((states[i], y(i)), radius=0.2, color='black', zorder=1)
    circles.append(circle)
    ax.add_patch(circle)
    for j in range(N_STATES):
        texts.append(ax.text(j, i, REWARDS[j], zorder=2))



def updatefig(_):
    global pos, states, next_states, value_matrix, step_size
    if np.allclose(pos, next_states):
        actions = act(states, value_matrix)
        next_states, reward = step(actions, states)
        value_matrix = update(value_matrix, states, next_states)
        step_size = (next_states - states) / 20
        states = next_states
        im.set_array(value_matrix)
    pos += step_size
    for i, j in enumerate(pos):
        circles[i].center = (x(j), i)
    return [im] + texts + circles


ani = animation.FuncAnimation(fig, updatefig, interval=.01, blit=True)
plt.show()
