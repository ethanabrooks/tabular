import matplotlib.pyplot as plt
import numpy as np

from algorithm import N_BATCH, N_STATES, EPISODES, MAX_TIMESTEPS, act, step, \
    update, REWARDS

from matplotlib import animation

fig = plt.figure()
ax = plt.axes()

value_matrix = np.zeros((N_BATCH, N_STATES))
im = plt.imshow(value_matrix, animated=True)
states = np.random.choice(N_STATES, N_BATCH)

circles = []
for i, state in enumerate(states):
    circle = plt.Circle((state, i), 0.2, color='red', )
    circles.append(circle)
    ax.add_patch(circle)


def updatefig(_):
    global states, value_matrix
    actions = act(states, value_matrix)
    next_states, reward = step(actions, states)
    value_matrix = update(value_matrix, states, next_states)
    print(value_matrix)
    states = next_states

    im.set_array(value_matrix)
    for i, state in enumerate(states):
        circles[i].center = (state, i)
    return [im] + circles


ani = animation.FuncAnimation(fig, updatefig, blit=True)
plt.show()
