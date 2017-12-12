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
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit()


def array_to_image(array, color, scale):
    shape = np.array(array.shape) * scale
    array = scipy.misc.imresize(array, shape, interp='nearest')
    return color.reshape(1, 1, 3) * np.expand_dims(array, 2)


def get_screen_shape(pad, shapes):
    height = pad + sum([height + pad for height, width in shapes])
    width = 2 * pad + max([width for height, width in shapes])
    return width, height


def get_subsurfaces(screen: pygame.Surface, pad, array_shapes: List[Shape]):
    subsurfaces = []
    left, top = pad, pad
    for shape in array_shapes:
        subsurfaces.append(
            screen.subsurface([left, top, shape.width, shape.height])
        )
        top += shape.height + pad
    return subsurfaces


def blit_image(subsurface, image):
    surfarray.blit_array(subsurface, image.transpose(1, 0, 2))


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
    pad = 10

    array1 = agent1.rewards
    array1[0, 2] = 0
    array2 = np.expand_dims(np.ones(8), 1)
    array2[2, 0] = 0
    orange = np.array([1, .4, 0])

    images = []
    shapes = []
    for array in [array1, array2]:
        image = array_to_image(array, orange, scale)
        shapes.append(Shape(*image.shape[:2]))
        images.append(image)
    screen_shape = get_screen_shape(pad, shapes)
    screen = pygame.display.set_mode(screen_shape)
    subsurfaces = get_subsurfaces(screen, pad, shapes)
    for subsurface, image in zip(subsurfaces, images):
        blit_image(subsurface, image)

    pygame.init()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit()

        blit_image(subsurface, np.random.random(image.shape) * 255)
        pygame.display.flip()


if __name__ == '__main__':
    main()
