#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

WORLD_SIZE = 4
REWARD = -1.0
ACTION_PROB = 0.25

world = np.zeros((WORLD_SIZE, WORLD_SIZE))

# left, up, right, down
actions = ['L', 'U', 'R', 'D']

nextState = []
for i in range(0, WORLD_SIZE):
    nextState.append([])
    for j in range(0, WORLD_SIZE):
        next = dict()
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

        nextState[i].append(next)

states = []
for i in range(0, WORLD_SIZE):
    for j in range(0, WORLD_SIZE):
        if (i == 0 and j == 0) or (i == WORLD_SIZE - 1 and j == WORLD_SIZE - 1):
            continue
        else:
            states.append([i, j])

def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(image):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = 'white'

        tb.add_cell(i, j, width, height, text=val, 
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(range(len(image))):
        tb.add_cell(i, -1, width, height, text=label+1, loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(range(len(image))):
        tb.add_cell(-1, j, width, height/2, text=label+1, loc='center', 
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    plt.show()

# for figure 4.1
num_iter = 0
while True:
    # keep iteration until convergence
    newWorld = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i, j in states:
        for action in actions:
            newPosition = nextState[i][j][action]
            # bellman equation
            newWorld[i, j] += ACTION_PROB * (REWARD + world[newPosition[0], newPosition[1]])
    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('Random Policy')
        draw_image(np.round(newWorld, decimals=2))
        break
    world = newWorld
    num_iter += 1
print('number of iterations: {}'.format(num_iter))


# softmax function
def softmax(Z):
    exps = np.exp(Z)
    return exps / np.sum(exps)

# update the actionProbs along the way in every iteration k
world = np.zeros((WORLD_SIZE, WORLD_SIZE))
num_iter = 0
while True:
    # keep iteration until convergence
    newWorld = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i, j in states:
        newPositions = nextState[i][j]
        currentVals = [world[newPositions[action][0], newPositions[action][1]] for action in actions]
        actionProbs = {action: prob for action, prob in zip(actions, softmax(currentVals))}
        
        for action in actions:
            newPosition = nextState[i][j][action]
            # bellman equation
            newWorld[i, j] += actionProbs[action] * (REWARD + world[newPosition[0], newPosition[1]]) # notice that here updated actionProbs is taken instead of equiprobable actions
    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('Random Policy')
        draw_image(np.round(newWorld, decimals=2))
        break
    world = newWorld
    num_iter += 1

print('number of iterations: {}'.format(num_iter))