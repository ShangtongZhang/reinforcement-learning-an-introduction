#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from utils import *
import matplotlib.pyplot as plt

# This example is adapted from example 5.5, Infinite Variance
# I made following modifications to the original example to demonstrate the advantage
# of partial return importance sampling and per-reward importance sampling
# 1. discount now is 0.5 instead of 1
# 2. the action BACK will always has reward +1, no matter what state it leads to
# In this setting, the true state value is around 1.81

# all possible actions
ACTION_BACK = 0
ACTION_END = 1
ACTIONS = [ACTION_BACK, ACTION_END]

# 2 states
TERMINAL_STATE = 0
START_STATE = 1

# action BACK will lead to terminal state w.p. 0.1
BACK_STAY_PROBABILITY = 0.9

# discount now is 0.5
DISCOUNT = 0.5

# behavior policy stays unchanged
def behaviorPolicy(state):
    return np.random.binomial(1, 0.5)

# target policy stays unchanged
def targetPolicy(state):
    return ACTION_BACK

def takeAction(state, action):
    # END will lead to terminal state with reward 0
    if action == ACTION_END:
        return TERMINAL_STATE, 0.0
    # BACK will always have reward +1
    if np.random.binomial(1, BACK_STAY_PROBABILITY) == 1:
        return state, 1.0
    return TERMINAL_STATE, 1.0

# play for an episode
# @return: trajectory of this episode
def play():
    currentState = START_STATE
    trajectory = []
    while currentState != TERMINAL_STATE:
        action = behaviorPolicy(currentState)
        newState, reward = takeAction(currentState, action)
        trajectory.append([currentState, action, reward])
        currentState = newState
    return trajectory

# base class for importance sampling
class ImportanceSampling:
    def __init__(self):
        self.name = ''
        # track the episode
        self.episodes = 0
        self.sumOfReturns = [0]

    # learn from a new trajectory
    def learn(self, trajectory):
        self.sumOfReturns.append(self.sumOfReturns[-1] + self.returnValue(trajectory))
        self.episodes += 1

    # derived class must override this to correctly calculate the return value
    def returnValue(self, trajectory):
        return 0.0

    # get the estimations over all episodes
    def estimations(self):
        del self.sumOfReturns[0]
        return np.asarray(self.sumOfReturns) / np.arange(1, self.episodes + 1)

    # reinitialize some variables for a new episode
    def clear(self):
        self.episodes = 0
        self.sumOfReturns = [0]

# classical ordinary importance sampling
class OrdinaryImportanceSampling(ImportanceSampling):
    def __init__(self):
        ImportanceSampling.__init__(self)
        self.name = 'ordinary importance sampling'

    def returnValue(self, trajectory):
        T = len(trajectory)
        # assume all rewards are 1, otherwise rho will be 0
        reward = (1 - pow(DISCOUNT, T)) / (1 - DISCOUNT)
        if trajectory[-1][1] == ACTION_END:
            rho = 0.0
        else:
            rho = 1.0 / pow(0.5, T)
        return rho * reward

# classical weighted importance sampling
class WeightedImportanceSampling(ImportanceSampling):
    def __init__(self):
        ImportanceSampling.__init__(self)
        self.name = 'weighted importance sampling'
        # track sum of rhos until current episode
        self.sumOfRhos = [0]

    def returnValue(self, trajectory):
        T = len(trajectory)
        # assume all rewards are 1, otherwise rho will be 0
        reward = (1 - pow(DISCOUNT, T)) / (1 - DISCOUNT)
        if trajectory[-1][1] == ACTION_END:
            rho = 0.0
        else:
            rho = 1.0 / pow(0.5, T)
        self.sumOfRhos.append(self.sumOfRhos[-1] + rho)
        return rho * reward

    def estimations(self):
        del self.sumOfReturns[0]
        del self.sumOfRhos[0]
        estimations = []
        for i in range(len(self.sumOfReturns)):
            if self.sumOfRhos[i] == 0:
                estimations.append(0.0)
            else:
                estimations.append(self.sumOfReturns[i] / self.sumOfRhos[i])
        return np.asarray(estimations)

    def clear(self):
        ImportanceSampling.__init__(self)
        self.sumOfRhos = [0]

# per-reward importance sampling
class PerRewardImportanceSampling(ImportanceSampling):
    def __init__(self):
        ImportanceSampling.__init__(self)
        self.name = 'per-reward importance sampling'

    def returnValue(self, trajectory):
        perRewardReturn = 0.0
        # get the power of gamma and rho incrementally
        rho = 1.0
        gamma = 1.0
        for state, action, reward in trajectory:
            if action == ACTION_END:
                rho *= 0
            else:
                rho *= 1 / 0.5
            perRewardReturn += rho * gamma * reward
            gamma *= DISCOUNT
        return perRewardReturn

# discounting-aware ordinary importance sampling
class DiscountingAwareOrdinaryImportanceSampling(ImportanceSampling):
    def __init__(self):
        ImportanceSampling.__init__(self)
        self.name = 'discounting-aware ordinary importance sampling'

    def returnValue(self, trajectory):
        # get the power of gamma and rho incrementally
        gamma = 1.0
        rho = 1.0
        partialReturn = 0.0
        rewards = 0.0
        for (state, action, reward), i in zip(trajectory, range(len(trajectory))):
            rewards += reward
            if action == ACTION_END:
                rho *= 0
            else:
                rho *= 1 / 0.5
            coef = gamma * rho
            if i < len(trajectory) - 1:
                coef *= 1 - DISCOUNT
            partialReturn += coef * rewards
            gamma *= DISCOUNT
        return partialReturn

# discounting-aware weighted importance sampling
class DiscountingAwareWeightedImportanceSampling(ImportanceSampling):
    def __init__(self):
        ImportanceSampling.__init__(self)
        self.name = 'discounting-aware weighted importance sampling'
        # track sum of rhos until current episode
        self.sumOfRhos = [0.0]

    def returnValue(self, trajectory):
        # get the power of gamma and rho incrementally
        gamma = 1.0
        rho = 1.0
        sumOfRhos = 0.0
        partialReturn = 0.0
        rewards = 0.0
        for (state, action, reward), i in zip(trajectory, range(len(trajectory))):
            rewards += reward
            if action == ACTION_END:
                rho *= 0
            else:
                rho *= 1 / 0.5
            coef = gamma * rho
            if i < len(trajectory) - 1:
                coef *= 1 - DISCOUNT
            partialReturn += coef * rewards
            sumOfRhos += coef
            gamma *= DISCOUNT
        self.sumOfRhos.append(self.sumOfRhos[-1] + sumOfRhos)
        return partialReturn

    def estimations(self):
        del self.sumOfReturns[0]
        del self.sumOfRhos[0]
        estimations = []
        for i in range(len(self.sumOfReturns)):
            if self.sumOfRhos[i] == 0:
                estimations.append(0.0)
            else:
                estimations.append(self.sumOfReturns[i] / self.sumOfRhos[i])
        return np.asarray(estimations)

    def clear(self):
        ImportanceSampling.__init__(self)
        self.sumOfRhos = [0]

figureIndex = 0
def mentoCarloSampling(method):
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    plt.title(method.name)
    runs = 5
    episodes = int(1e5)
    for run in range(runs):
        method.clear()
        for episode in range(episodes):
            print method.name, 'episode:', episode
            trajectory = play()
            method.learn(trajectory)
        plt.plot(np.arange(episodes), method.estimations())
    plt.xlabel('Episodes')
    plt.ylabel('State value')
    plt.xscale('log')
    return

def figure():
    methods = [OrdinaryImportanceSampling,
               WeightedImportanceSampling,
               PerRewardImportanceSampling,
               DiscountingAwareOrdinaryImportanceSampling,
               DiscountingAwareWeightedImportanceSampling]
    for method in methods:
        mentoCarloSampling(method())

figure()
plt.show()
