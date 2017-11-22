#######################################################################
# Copyright (C)                                                       #
# 2017 Ji Yang(jyang7@ualberta.ca)                                    #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

"""
Simulation of applying the in-place version of iterative policy evaluation algorithm in
the grid world, from Sutton & Barto's RL Book, Section 4.1 Policy Evaluation (Prediction)
"""

from __future__ import print_function
import numpy as np

WORLD_SIZE = 4
REWARD = -1.0  # a constant reward for moving to non-terminal states
ACTION_PROB = 0.25  # random policy
GAMMA = 1.0  # episode task, no discount

world = np.zeros((WORLD_SIZE, WORLD_SIZE))

# left, up, right, down
actions = ['L', 'U', 'R', 'D']

# here we generate the possible next state for each cell in the grid world
next_state = []
for i in range(0, WORLD_SIZE):
    next_state.append([])
    for j in range(0, WORLD_SIZE):
        next = dict()
        # uncomment this print and the last print in this loop if you haven't got it
        # print('Grid cell (col, row):', i, j)

        if i == 0:
            next['U'] = [i, j]
        else:
            next['U'] = [i - 1, j]

        if i == WORLD_SIZE - 1:
            next['D'] = [i, j]
        else:
            next['D'] = [i + 1, j]

        if j == 0:
            next['L'] = [i, j]
        else:
            next['L'] = [i, j - 1]

        if j == WORLD_SIZE - 1:
            next['R'] = [i, j]
        else:
            next['R'] = [i, j + 1]

        # print(next)

        next_state[i].append(next)

# generate state coordinates in the grid word, except for the terminal state
states = []
for i in range(0, WORLD_SIZE):
    for j in range(0, WORLD_SIZE):
        # handle the top-left and bottom-right terminal state
        if (i == 0 and j == 0) or (i == WORLD_SIZE - 1 and j == WORLD_SIZE - 1):
            continue
        else:
            states.append([i, j])

#######################################################################
# Figure 4.1
k = 0
figure_plot_ks = [0, 1, 2, 3, 10]
threshold_delta = 1e-9  # a very small positive number

# keep iterating until convergence
while True:
    new_world = np.zeros((WORLD_SIZE, WORLD_SIZE))
    if k in figure_plot_ks:
        print('k = {}'.format(k))
        print(world)
    k += 1
    for i, j in states:
        for action in actions:
            new_state = next_state[i][j][action]
            # expected update by the Bellman equation
            new_world[i, j] += ACTION_PROB * GAMMA * (REWARD + world[new_state[0], new_state[1]])

    # convergence check
    if np.sum(np.abs(world - new_world)) < threshold_delta:
        print('Random Policy')
        print(new_world)
        break
    world = new_world
