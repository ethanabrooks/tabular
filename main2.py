from collections import namedtuple
from typing import Tuple, List

import numpy as np
import pygame
import scipy.misc
from pygame import surfarray, rect
from pygame.locals import *

import algorithm
from algorithm import stochastic_stepwise_transitions

Shape = namedtuple('Shape', 'height width')


    # pygame.display.set_caption(name)


def update():
    if pygame.event.wait().type == QUIT:
        raise SystemExit()


def array_to_image(array, color, scale):
    shape = np.array(array.shape) * scale
    array = scipy.misc.imresize(array, shape, interp='nearest')
    return color.reshape(1, 1, 3) * np.expand_dims(array, 2)


def get_subsurfaces(screen: pygame.Surface, pad, array_shapes: List[Shape]):
    subsurfaces = []
    left, top = pad, pad
    for shape in array_shapes:
        subsurfaces.append(
            screen.subsurface([left, top, shape.width, shape.height])
        )
        top += shape.height + pad
    return subsurfaces


def get_screen_shape(pad, shapes):
    height = pad + sum([height + pad for height, width in shapes])
    width = 2 * pad + max([width for height, width in shapes])
    return width, height


def rescale_array(array, scale):
    shape = np.array(array.shape) * scale
    return scipy.misc.imresize(array, np.array(shape), interp='nearest')


def main():
    """show various surfarray effects
    If arraytype is provided then use that array package. Valid
    values are 'numeric' or 'numpy'. Otherwise default to NumPy,
    or fall back on Numeric if NumPy is not installed.
    """

    ### Agents

    np.set_printoptions(precision=1)
    n_states = 4
    transitions = stochastic_stepwise_transitions(sigma=.5, n_states=n_states)
    rewards = np.zeros((1, n_states))
    # rewards[[0, -1]] = [.001, .999]
    rewards[0, np.random.choice(n_states)] = 1
    agent1 = algorithm.Agent(
        gamma=.95,
        alpha=.95,
        n_states=n_states,
        n_agents=1,
        n_actions=2,
        transitions=transitions,
        max_timesteps=n_states,
        rewards=rewards,
        terminal=rewards.argmax(axis=1),
    )
    agent2 = algorithm.Agent(
        gamma=.95,
        alpha=.95,
        n_states=n_states,
        n_agents=1,
        n_actions=2,
        transitions=transitions,
        max_timesteps=n_states,
        rewards=agent1.rewards[[0]],
        terminal=agent1.rewards[[0]].argmax(axis=1),
    )
    scale = 40
    pad = 80

    pygame.init()
    print('Using %s' % surfarray.get_arraytype().capitalize())
    print('Press the mouse button to advance image.')
    print('Press the "s" key to save the current image.')

    array1 = np.expand_dims(np.ones(4), 0)
    print(array1)
    array1[0, 2] = 0
    array2 = np.expand_dims(np.ones(8), 1)
    array2[2, 0] = 0

    shape1 = Shape(*(np.array(array1.shape) * scale))
    shape2 = Shape(*(np.array(array2.shape) * scale))
    array1 = rescale_array(array1, scale)
    array2 = rescale_array(array2, scale)
    # array1 = scipy.misc.imresize(agent1.array1, shape1, interp='nearest')
    orange = np.array([1, .4, 0])
    array1 = orange.reshape(1, 1, 3) * np.expand_dims(array1, 2)
    array2 = orange.reshape(1, 1, 3) * np.expand_dims(array2, 2)

    shapes = [Shape(*array.shape[:2])
              for array in [array1, array2]]

    screen_shape = get_screen_shape(pad, shapes)
    screen = pygame.display.set_mode(screen_shape)
    for subsurface, array in zip(get_subsurfaces(screen, pad, shapes), [array1, array2]):
        surfarray.blit_array(subsurface, array.transpose(1, 0, 2))
    # pygame.draw.rect(s1, (255, 0, 0), pygame.Rect((30, 40, 30, 50)))

    # screen2 = pygame.display.set_mode(shape)
    # surfarray.blit_array(screen2, array)

    pygame.display.flip()

    # init(array1, 'striped')
    while True:
        update()


if __name__ == '__main__':
    main()
