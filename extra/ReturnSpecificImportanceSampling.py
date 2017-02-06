#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt

# This example is adapted from example 5.5, Infinite Variance
# I made following modifications to the original example to demonstrate the advantage
# of return specific importance sampling
# 1. discount now is 0.5 instead of 1
# 2. the transition from state s to state s will have reward +2
# 3. the action BACK will lead to state s or terminal state with equal probability
# In this setting, the true state value is 2

# all possible actions
ACTION_BACK = 0
ACTION_END = 1
ACTIONS = [ACTION_BACK, ACTION_END]

# 2 states
TERMINAL_STATE = 0
START_STATE = 1

# action BACK will lead to the state itself w.p. 0.5
BACK_STAY_PROBABILITY = 0.5

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
    if np.random.binomial(1, BACK_STAY_PROBABILITY) == 1:
        # BACK with transition to itself has reward +2
        return state, 2.0
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

    # reinitialize some variables for a new independent run
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
        returns = 0.0
        gamma = 1.0
        for state, action, reward in trajectory:
            returns += gamma * reward
            gamma *= DISCOUNT
        if trajectory[-1][1] == ACTION_END:
            rho = 0.0
        else:
            rho = 1.0 / pow(0.5, T)
        return rho * returns

# classical weighted importance sampling
class WeightedImportanceSampling(ImportanceSampling):
    def __init__(self):
        ImportanceSampling.__init__(self)
        self.name = 'weighted importance sampling'
        # track sum of rhos until current episode
        self.sumOfRhos = [0]

    def returnValue(self, trajectory):
        T = len(trajectory)
        returns = 0.0
        gamma = 1.0
        for state, action, reward in trajectory:
            returns += gamma * reward
            gamma *= DISCOUNT
        if trajectory[-1][1] == ACTION_END:
            rho = 0.0
        else:
            rho = 1.0 / pow(0.5, T)
        self.sumOfRhos.append(self.sumOfRhos[-1] + rho)
        return rho * returns

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
        ImportanceSampling.clear(self)
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

# per-reward weighted importance sampling (PDWIS) is biased and inconsistent, so it doesn't work
# consistent per-reward weighted importance sampling (CPDWIS) is still biased however consistent
# View Philip Thomas's PhD thesis http://psthomas.com/papers/Thomas2015c.pdf (3.9 3.10)
# for more mathematical discussions

# PDWIS doesn't work
class PerRewardWeightedImportanceSampling(ImportanceSampling):
    def __init__(self):
        ImportanceSampling.__init__(self)
        self.name = 'per-reward weighted importance sampling (PDWIS)'
        # track sum of rhos until current episode
        self.sumOfRhos = [0]

    def returnValue(self, trajectory):
        perRewardReturn = 0.0
        # get the power of gamma and rho incrementally
        rho = 1.0
        gamma = 1.0
        sumOfRhos = 0.0
        for state, action, reward in trajectory:
            if action == ACTION_END:
                rho *= 0
            else:
                rho *= 1 / 0.5
            perRewardReturn += rho * gamma * reward
            sumOfRhos += rho * gamma
            gamma *= DISCOUNT
        self.sumOfRhos.append(self.sumOfRhos[-1] + sumOfRhos)
        return perRewardReturn

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
        ImportanceSampling.clear(self)
        self.sumOfRhos = [0]

# CPDWIS works but is biased
# CPDWIS can't be fully incremental
# I try to calculate it as incrementally as possible,
# so readability may be hurt to some extent
class ConsistentPerRewardWeightedImportanceSampling(ImportanceSampling):
    def __init__(self):
        ImportanceSampling.__init__(self)
        self.name = 'consistent per-reward weighted importance sampling (CPDWIS)'
        # track nominator
        self.weightedRewards = []
        # track denominator
        self.weights = []
        # track the length of trajectories
        self.trajectoriesLength = []

    def learn(self, trajectory):
        rho = 1.0
        gamma = 1.0
        self.weightedRewards.append([])
        self.weights.append([])
        self.trajectoriesLength.append(len(trajectory))
        for state, action, reward in trajectory:
            if action == ACTION_END:
                rho *= 0
            else:
                rho *= 1 / 0.5
            self.weightedRewards[-1].append(gamma * reward * rho)
            self.weights[-1].append(rho)
            gamma *= DISCOUNT

    def estimations(self):
        maxTime = np.max(self.trajectoriesLength)
        self.weightedRewards = np.asarray([pad(array, maxTime) for array in self.weightedRewards])
        self.weights = np.asarray([pad(array, maxTime) for array in self.weights])
        for i in range(1, self.weightedRewards.shape[0]):
            self.weightedRewards[i, :] += self.weightedRewards[i - 1, :]
            self.weights[i, :] += self.weights[i - 1, :]
        for i in range(0, self.weightedRewards.shape[0]):
            for j in range(0, self.weightedRewards.shape[1]):
                if self.weights[i, j] == 0:
                    self.weights[i, j] = 1
                    self.weightedRewards[i, j] = 0
        estimations = []
        for i in range(self.weightedRewards.shape[0]):
            estimations.append(np.sum(self.weightedRewards[i, :] / self.weights[i, :]))
        return np.asarray(estimations)

    def clear(self):
        ImportanceSampling.clear(self)
        self.weightedRewards = []
        self.weights = []
        self.trajectoriesLength = []

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
        ImportanceSampling.clear(self)
        self.sumOfRhos = [0]

figureIndex = 0
def monteCarloSampling(method):
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    plt.title(method.name)
    runs = 5
    episodes = int(1e5)
    for run in range(runs):
        method.clear()
        for episode in range(episodes):
            print(method.name, 'episode:', episode)
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
               DiscountingAwareWeightedImportanceSampling,
               PerRewardWeightedImportanceSampling,
               ConsistentPerRewardWeightedImportanceSampling]
    for method in methods:
        monteCarloSampling(method())

figure()
plt.show()
