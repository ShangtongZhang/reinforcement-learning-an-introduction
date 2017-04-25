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
from mpl_toolkits.mplot3d import Axes3D

# all states: state 0-5 are upper states
STATES = np.arange(0, 7)
# state 6 is lower state
LOWER_STATE = 6
# discount factor
DISCOUNT = 0.99

# each state is represented by a vector of length 8
FEATURE_SIZE = 8
FEATURES = np.zeros((len(STATES), FEATURE_SIZE))
for i in range(LOWER_STATE):
    FEATURES[i, i] = 2
    FEATURES[i, 7] = 1
FEATURES[LOWER_STATE, 6] = 1
FEATURES[LOWER_STATE, 7] = 2

# all possible actions
DASHED = 0
SOLID = 1
ACTIONS = [DASHED, SOLID]

# reward is always zero
REWARD = 0

# take @action at @state, return the new state
def takeAction(state, action):
    if action == SOLID:
        return LOWER_STATE
    return np.random.choice(STATES[: LOWER_STATE])

# target policy
def targetPolicy(state):
    return SOLID

# behavior policy
BEHAVIOR_SOLID_PROBABILITY = 1.0 / 7
def behaviorPolicy(state):
    if np.random.binomial(1, BEHAVIOR_SOLID_PROBABILITY) == 1:
        return SOLID
    return DASHED

# Semi-gradient off-policy temporal difference
# @state: current state
# @weights: weight for each component of the feature vector
# @alpha: step size
# @return: next state
def semiGradientOffPolicyTD(state, weights, alpha):
    action = behaviorPolicy(state)
    nextState = takeAction(state, action)
    # get the importance ration
    if action == DASHED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
    delta = REWARD + DISCOUNT * np.dot(FEATURES[nextState, :], weights) - \
            np.dot(FEATURES[state, :], weights)
    delta *= rho * alpha
    # derivatives happen to be the same matrix due to the linearity
    weights += FEATURES[state, :] * delta
    return nextState

# Semi-gradient DP
# @weights: weight for each component of the feature vector
# @alpha: step size
def semiGradientDP(weights, alpha):
    delta = 0.0
    # go through all the states
    for currentState in STATES:
        expectedReturn = 0.0
        # compute bellman error for each state
        for nextState in STATES:
            if nextState == LOWER_STATE:
                expectedReturn += REWARD + DISCOUNT * np.dot(weights, FEATURES[nextState, :])
        bellmanError = expectedReturn - np.dot(weights, FEATURES[currentState, :])
        # accumulate gradients
        delta += bellmanError * FEATURES[currentState, :]
    # derivatives happen to be the same matrix due to the linearity
    weights += alpha / len(STATES) * delta

figureIndex = 0

# Figure 11.2(a), Baird's counterexample
def figure11_2_a():
    # Initial the weights
    weights = np.ones(FEATURE_SIZE)
    weights[6] = 10

    alpha = 0.01

    steps = 1000
    thetas = np.zeros((FEATURE_SIZE, steps))
    state = np.random.choice(STATES)
    for step in range(steps):
        state = semiGradientOffPolicyTD(state, weights, alpha)
        thetas[:, step] = weights

    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    for i in range(FEATURE_SIZE):
        plt.plot(thetas[i, :], label='theta' + str(i + 1))
    plt.xlabel('Steps')
    plt.ylabel('Theta value')
    plt.legend()

def figure11_2_b():
    # Initial the weights
    weights = np.ones(FEATURE_SIZE)
    weights[6] = 10

    alpha = 0.01

    sweeps = 1000
    thetas = np.zeros((FEATURE_SIZE, sweeps))
    for sweep in range(sweeps):
        semiGradientDP(weights, alpha)
        thetas[:, sweep] = weights

    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    for i in range(FEATURE_SIZE):
        plt.plot(thetas[i, :], label='theta' + str(i + 1))
    plt.xlabel('Sweeps')
    plt.ylabel('Theta value')
    plt.legend()

if __name__ == '__main__':
    figure11_2_a()
    figure11_2_b()
    plt.show()