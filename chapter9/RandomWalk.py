#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from utils import *
import matplotlib.pyplot as plt
import pickle

# # of states except for terminal states
N_STATES = 1000

# true state values, just a promising guess
trueStateValues = np.arange(-1001, 1003, 2) / 1001.0

# all states
states = np.arange(1, N_STATES + 1)

# start from a central state
START_STATE = 500

# terminal states
END_STATES = [0, N_STATES + 1]

# possible actions
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action
STEP_RANGE = 100

# Dynamic programming to find the true state values, based on the promising guess above
# Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states
while True:
    oldTrueStateValues = np.copy(trueStateValues)
    for state in states:
        trueStateValues[state] = 0
        for action in ACTIONS:
            for step in range(1, STEP_RANGE + 1):
                step *= action
                newState = state + step
                newState = max(min(newState, N_STATES + 1), 0)
                # asynchronous update for faster convergence
                trueStateValues[state] += 1.0 / (2 * STEP_RANGE) * trueStateValues[newState]
    error = np.sum(np.abs(oldTrueStateValues - trueStateValues))
    print error
    if error < 1e-2:
        break
# correct the state value for terminal states to 0
trueStateValues[0] = trueStateValues[-1] = 0

# take an @action at @state, return new state and reward for this transition
def takeAction(state, action):
    step = np.random.randint(1, STEP_RANGE + 1)
    step *= action
    state += step
    state = max(min(state, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
    return state, reward

# get an action, following random policy
def getAction():
    if np.random.binomial(1, 0.5) == 1:
        return 1
    return -1

# a wrapper class for aggregation value function
class ValueFunction:
    # @numOfGroups: # of aggregations
    def __init__(self, numOfGroups):
        self.numOfGroups = numOfGroups
        self.groupSize = N_STATES / numOfGroups

        # thetas
        self.params = np.zeros(numOfGroups)

    # get the value of @state
    def value(self, state):
        if state in END_STATES:
            return 0
        groupIndex = (state - 1) / self.groupSize
        return self.params[groupIndex]

    # update parameters
    # Notice that there is only 1 parameter to be updated each time in aggregation value function,
    # so there is no difference between synchronous and asynchronous update
    # @delta: step size * (return - old estimation)
    # @state: state of current sample
    def update(self, delta, state):
        groupIndex = (state - 1) / self.groupSize
        self.params[groupIndex] += delta

# gradient Mento Carlo algorithm
# @valueFunction: an instance of class ValueFunction
# @alpha: step size
# @distribution: array to store the distribution statistics
def gradientMentoCarlo(valueFunction, alpha, distribution=None):
    currentState = START_STATE
    trajectory = [currentState]

    # We assume gamma = 1, so return is just the same as the latest reward
    reward = 0.0
    while currentState not in END_STATES:
        action = getAction()
        newState, reward = takeAction(currentState, action)
        trajectory.append(newState)
        currentState = newState

    # Gradient update for each state in this trajectory
    for state in trajectory[:-1]:
        delta = alpha * (reward - valueFunction.value(state))
        valueFunction.update(delta, state)
        if distribution is not None:
            distribution[state] += 1

# Figure 9.1, gradient Monte Carlo algorithm
def figure9_1():
    nEpisodes = int(1e5)
    alpha = 2e-5

    # we have 10 aggregations in this example, each has 100 states
    valueFunction = ValueFunction(10)
    distribution = np.zeros(N_STATES + 2)
    for episode in range(0, nEpisodes):
        print 'episode:', episode
        gradientMentoCarlo(valueFunction, alpha, distribution)

    distribution /= np.sum(distribution)
    stateValues = [valueFunction.value(i) for i in states]

    plt.figure(0)
    plt.plot(states, stateValues, label='Approximate MC value')
    plt.plot(states, trueStateValues[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')

    plt.figure(1)
    plt.plot(states, distribution[1: -1], label='State distribution')
    plt.xlabel('State')
    plt.ylabel('Distribution')
    plt.legend()

figure9_1()
plt.show()
