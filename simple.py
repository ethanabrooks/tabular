import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x = np.linspace(0, 15, 100)

fig, (p1, p2) = plt.subplots(2)

p1.set_xlim([0, 15])
p1.set_ylim([0, 100])

p2.set_xlim([0, 15])
p2.set_ylim([0, 100])

# set up empty lines to be updates later on
l1, = p1.plot([], [], 'b')
l2, = p2.plot([], [], 'r')


def gen1():
    i = 0.5
    while True:
        y = i * x
        l1.set_data(x, y)
        yield l1,
        i += 0.1


def gen2():
    j = 0
    while True:
        yield j
        j += 1


def run1(c):
    return c
    # y = c * x
    # l1.set_data(x, y)


def run2(c):
    y = c * x
    l2.set_data(x, y)


ani1 = animation.FuncAnimation(fig, run1, gen1, interval=1, blit=True)
ani2 = animation.FuncAnimation(fig, run2, gen2, interval=1)
plt.show()
