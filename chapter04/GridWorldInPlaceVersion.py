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
ACTION_PORB = 0.25
REWARD = -1


def Next_State(state, action):
	i, j = state
	if i == 0 and action == 'U':
		return [i, j]
	if i == WORLD_SIZE-1 and action == 'D':
		return [i, j]
	if j == 0 and action == 'L':
		return [i, j]
	if j == WORLD_SIZE-1 and action == 'R':
		return [i, j]

	if action == 'U':
		return [i-1, j]
	if action == 'D':
		return [i+1, j]
	if action == 'R':
		return [i, j+1]
	if action == 'L':
		return[i, j-1]


states = []
for i in range(WORLD_SIZE):
	for j in range(WORLD_SIZE):
		if (i == 0 and j == 0) or (i == WORLD_SIZE-1 and j == WORLD_SIZE-1):
			continue
		states.append([i, j])

actions = ['U', 'D', 'R', 'L']

new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
state_values = new_state_values.copy()

iteration = 1
while True:
	for (i, j) in states:
		value = 0
		for action in actions:
			next_i, next_j = Next_State([i, j], action)
			value += ACTION_PORB * (REWARD + new_state_values[next_i, next_j])
		new_state_values[i, j] = value

	if np.sum(np.abs(new_state_values - state_values)) < 1e-4:
		state_values = new_state_values.copy()
		break

	state_values = new_state_values.copy()
	iteration += 1

print('Random Policy')
print(state_values)
print('Iteration Times')
print(iteration)






































