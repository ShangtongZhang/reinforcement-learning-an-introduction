#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# all states: state 0-5 are upper states
STATES = np.arange(0, 8)
# state 6 is lower state
LOWER_STATE = 6
# state 7 is terminal state
TERMINAL_STATE = 7

# probability to the terminal state for a transition
TERMINAL_PROBABILITY = 0.01

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

# take @action at @state, return the new state
def takeAction(state, action):
    if np.random.binomial(1, TERMINAL_PROBABILITY) == 1:
        return TERMINAL_STATE
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
# @weights: weight for each component of the feature vector
# @alpha: step size
def semiGradientOffPolicyTD(weights, alpha):
    # start from a random state
    currentState = np.random.choice(STATES[: TERMINAL_STATE])
    while currentState != TERMINAL_STATE:
        action = behaviorPolicy(currentState)
        newState = takeAction(currentState, action)
        # get the importance ration
        if action == DASHED:
            rho = 0.0
        else:
            rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
        delta = np.dot(FEATURES[newState, :], weights) - \
                np.dot(FEATURES[currentState, :], weights)
        delta *= rho * alpha
        # derivatives happen to be the same matrix due to the linearity
        weights += FEATURES[currentState, :] * delta
        currentState = newState

# Figure 11.1, Baird's counterexample
def figure11_1():
    # Initial the weights
    weights = np.ones(FEATURE_SIZE)
    weights[6] = 10

    # pick up a small alpha
    alpha = 1e-3

    episodes = 100
    thetas = np.zeros((FEATURE_SIZE, episodes))
    for episode in range(episodes):
        semiGradientOffPolicyTD(weights, alpha)
        thetas[:, episode] = weights

    plt.figure(0)
    for i in range(FEATURE_SIZE):
        plt.plot(thetas[i, :], label='theta' + str(i + 1))
    plt.xlabel('Episodes')
    plt.ylabel('Theta value')
    plt.legend()

figure11_1()
plt.show()