import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from algorithm import N_BATCH, N_STATES, act, step, \
    update

fig = plt.figure()
ax = plt.axes()

value_matrix = np.zeros((N_BATCH, N_STATES), dtype=np.float)
im = plt.imshow(value_matrix, vmin=0, vmax=1, animated=True)
states = np.random.choice(N_STATES, N_BATCH)
circles = []
for i, state in enumerate(states):
    circle = plt.Circle((state, i), radius=0.2, color='white')
    circles.append(circle)
    ax.add_patch(circle)

def updatefig(_):
    global states, value_matrix
    actions = act(states, value_matrix)
    next_states, reward = step(actions, states)
    value_matrix = update(value_matrix, states, next_states)
    states = next_states
    im.set_array(value_matrix)
    for i, state in enumerate(states):
        circles[i].center = (state, i)
    return [im] + circles


ani = animation.FuncAnimation(fig, updatefig, blit=True)
plt.show()
