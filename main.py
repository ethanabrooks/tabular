import matplotlib.pyplot as plt
import numpy as np

from algorithm import N_BATCH, N_STATES, EPISODES, MAX_TIMESTEPS, act, step, \
    update, REWARDS

from matplotlib import animation


def f(x, y):
    return np.sin(x) + np.cos(y)


fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

value_matrix = np.zeros((N_BATCH, N_STATES))
im = plt.imshow(value_matrix, animated=True)
states = np.random.choice(N_STATES, N_BATCH)


circles = [plt.Circle((5, 5), 0.2, color='red', ) for _ in range(N_BATCH)]
for circle in circles:
    ax.add_patch(circle)



def updatefig(i):
    global states, value_matrix
    actions = act(states, value_matrix)
    next_states, reward = step(actions, states)
    value_matrix = update(value_matrix, states, next_states)
    print(value_matrix)
    # im.set_array(value_matrix)
    # circle.center =
    global x, y
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(value_matrix)
    circle.center = (i, i)
    return im, circle


ani = animation.FuncAnimation(fig, updatefig, blit=True)
plt.show()
