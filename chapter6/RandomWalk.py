#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from utils import *
import matplotlib.pyplot as plt

# 0 is the left terminal state
# 6 is the right terminal state
# 1 ... 5 represents A ... E
states = np.zeros(7)
states[1:6] = 0.5
# For convenience, we assume all rewards are 0
# and the left terminal state has value 0, the right terminal state has value 1
# This trick has been used in Gambler's Problem
states[6] = 1

# set up true state values
trueValue = np.zeros(7)
trueValue[1:6] = np.arange(1, 6) / 6.0
trueValue[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1

def temporalDifference(states, alpha=0.1):
    state = 3
    while True:
        oldState = state
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        # Assume all rewards are 0
        reward = 0
        # TD update
        states[oldState] += alpha * (reward + states[state] - states[oldState])
        if state == 6 or state == 0:
            break
    return states

def monteCarlo(states, alpha=0.1):
    state = 3
    trajectory = [3]
    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    returns = 0
    while True:
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break
    for state_ in trajectory[:-1]:
        # MC update
        states[state_] += alpha * (returns - states[state_])
    return states

# Figure 6.2 left
def stateValue():
    episodes = [0, 1, 10, 100]
    currentStates = np.copy(states)
    plt.figure(1)
    axisX = np.arange(0, 7)
    for i in range(0, episodes[-1] + 1):
        if i in episodes:
            plt.plot(axisX, currentStates, label=str(i) + ' episodes')
        temporalDifference(currentStates)
    plt.plot(axisX, trueValue, label='true values')
    plt.xlabel('state')
    plt.legend()

# Figure 6.2 right
def RMSError():
    TDAlpha = [0.15, 0.1, 0.05]
    MCAlpha = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    plt.figure(2)
    axisX = np.arange(0, episodes)
    for alpha in TDAlpha + MCAlpha:
        totalErrors = np.zeros(episodes)
        if alpha in TDAlpha:
            method = 'TD'
        else:
            method = 'MC'
        for run in range(0, runs):
            errors = []
            currentStates = np.copy(states)
            for i in range(0, episodes):
                errors.append(np.sqrt(np.sum(np.power(trueValue - currentStates, 2)) / 5.0))
                if method == 'TD':
                    temporalDifference(currentStates, alpha=alpha)
                else:
                    monteCarlo(currentStates, alpha=alpha)
            totalErrors += np.asarray(errors)
        totalErrors /= runs
        plt.plot(axisX, totalErrors, label=method + ', alpha=' + str(alpha))
    plt.xlabel('episodes')
    plt.legend()

stateValue()
RMSError()
plt.show()