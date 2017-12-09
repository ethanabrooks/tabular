import matplotlib.pyplot as plt
import numpy as np

from algorithm import N_BATCH, N_STATES, EPISODES, MAX_TIMESTEPS, act, step, \
    update, REWARDS

from matplotlib import animation


# if __name__ == '__main__':
#     value_matrix = np.zeros((N_BATCH, N_STATES))
#     states = np.random.choice(N_STATES, N_BATCH)
#     for _ in range(EPISODES):
#         for _ in range(MAX_TIMESTEPS):
#             actions = act(states, value_matrix)
#             next_states, reward = step(actions, states)
#             value_matrix = update(value_matrix, states, next_states)
#             states = next_states[next_states != None]

# if __name__ == '__main__':
#     # Just some example data (random)
#     x = 0
#     for _ in range(50):
#         x += .01
#         data[0] += .01
#         # image.set_data(data)
#         circle = plt.Circle((x, 0), radius=.01)
#         plt.gca().add_patch(circle)
#         plt.pause(.1)
#         plt.draw()

def f(x, y):
    return np.sin(x) + np.cos(y)


fig = plt.figure()
ax = plt.axes()


x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

im = plt.imshow(f(x, y), animated=True)
circle = plt.Circle((5, 5), 0.2, color='red', )
ax.add_patch(circle)


def updatefig(i):
    global x, y
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(f(x, y))
    circle.center = (i, i)
    return im, circle


ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()
