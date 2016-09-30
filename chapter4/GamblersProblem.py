#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from utils import *
import matplotlib.pyplot as plt

GOAL = 100
states = np.arange(GOAL + 1)
headProb = 0.4
policy = np.zeros(GOAL + 1)
stateValue = np.zeros(GOAL + 1)
stateValue[GOAL] = 1.0

k = 0
while k < 500:
    newStateValue = np.zeros(GOAL + 1)
    for state in states[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        actionReturns = []
        for action in actions:
            actionReturns.append(headProb * stateValue[state + action] + (1 - headProb) * stateValue[state - action])
        newStateValue[state] = np.max(actionReturns)
    newStateValue[0] = 0
    newStateValue[GOAL] = 1
    # print np.sum(np.abs(newStateValue - stateValue))
    # if np.sum(newStateValue != stateValue) == 0:
    # if np.sum(np.abs(newStateValue - stateValue)) < 1e-4:
    #     stateValue = newStateValue
    #     break
    k += 1
    stateValue = newStateValue

for state in states[1:GOAL]:
    actions = np.arange(min(state, GOAL - state) + 1)
    actionReturns = []
    for action in actions:
        actionReturns.append(headProb * stateValue[state + action] + (1 - headProb) * stateValue[state - action])
    maxReturns = np.max(actionReturns)
    print [actions[i] for i in range(0, len(actionReturns)) if actionReturns[i] == maxReturns]
    # policy[state] = actions[argmax(actionReturns)]

# print stateValue
# plt.plot(stateValue)
# plt.show()
# print policy
# plt.scatter(states, policy)
# plt.plot(policy)
# plt.show()