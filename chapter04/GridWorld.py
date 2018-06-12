#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np

WORLD_SIZE = 4
ACTION_PROB = 0.25
REWARD = -1

def next_state(state, action):
    i, j = state
    if i == 0 and action == 'U':
        return [i, j]
    if i == WORLD_SIZE - 1 and action == 'D':
        return [i, j]
    if j == 0 and action == 'L':
        return [i, j]
    if j == WORLD_SIZE - 1 and action == 'R':
        return [i, j]

    if action == 'U':
        return [i - 1, j]
    if action == 'D':
        return [i + 1, j]
    if action == 'R':
        return [i, j + 1]
    if action == 'L':
        return[i, j - 1]

states = []
for i in range(WORLD_SIZE):
    for j in range(WORLD_SIZE):
        if (i == 0 and j == 0) or (i == WORLD_SIZE-1 and j == WORLD_SIZE-1):
            continue
        states.append([i, j])

actions = ['U', 'D', 'R', 'L']

def compute_state_values(in_place=False):
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    state_values = new_state_values.copy()
    iteration = 1
    while True:
        src = new_state_values if in_place else state_values
        for (i, j) in states:
            value = 0
            for action in actions:
                next_i, next_j = next_state([i, j], action)
                value += ACTION_PROB * (REWARD + src[next_i, next_j])
            new_state_values[i, j] = value

        if np.sum(np.abs(new_state_values - state_values)) < 1e-4:
            state_values = new_state_values.copy()
            break

        state_values = new_state_values.copy()
        iteration += 1

    return state_values, iteration

if __name__ == '__main__':
    state_values, iteration = compute_state_values(in_place=True)
    print('In-place:')
    print('State values under random policy after %d iterations' % (iteration))
    print(state_values)

    state_values, iteration = compute_state_values(in_place=False)
    print('Synchronous:')
    print('State values under random policy after %d iterations' % (iteration))
    print(state_values)






































