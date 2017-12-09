import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from algorithm import N_BATCH, N_STATES, act, step, \
    update

fig = plt.figure()
ax = plt.axes()

value_matrix = np.zeros((N_BATCH, N_STATES), dtype=np.float)
im = plt.imshow(value_matrix, vmin=0, vmax=1, cmap='Oranges', animated=True)
states = np.random.choice(N_STATES, N_BATCH)
next_states = states
pos = states.astype(float)
step_size = 0
circles = []
for i, state in enumerate(states):
    circle = plt.Circle((state, i), radius=0.2, color='black')
    circles.append(circle)
    ax.add_patch(circle)


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
    for i, x in enumerate(pos):
        circles[i].center = (x, i)
    return [im] + circles


ani = animation.FuncAnimation(fig, updatefig, interval=.01, blit=True)
plt.show()
