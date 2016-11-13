#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from utils import *
import matplotlib.pyplot as plt

# all states
N_STATES = 19

# all states but terminal states
states = np.arange(1, N_STATES + 1)

# start from the middle state
START_STATE = 10

# two terminal states
# an action leading to the left terminal state has reward -1
# an action leading to the right terminal state has reward 1
END_STATES = [0, N_STATES + 1]

# true state values from Bellman equation
realStateValues = np.arange(-20, 22, 2) / 20.0
realStateValues[0] = realStateValues[N_STATES + 1] = 0.0

class ValueFunction:
    def __init__(self, rate, stepSize, numOfTilings=8, maxSize=2048):
        self.maxSize = maxSize
        self.rate = rate
        self.numOfTilings = numOfTilings
        self.stepSize = stepSize / numOfTilings

        self.hashTable = IHT(maxSize)
        self.weights = np.zeros(maxSize)

        self.scale = self.numOfTilings / float(N_STATES - 1)

    def getActiveTiles(self, state):
        return tiles(self.hashTable, self.numOfTilings, [self.scale * state])

    def value(self, state):
        if state in END_STATES:
            return 0.0
        activeTiles = self.getActiveTiles(state)
        return np.sum(self.weights[activeTiles])

class OffLineLambdaReturn(ValueFunction):
    def __init__(self, rate, stepSize):
        ValueFunction.__init__(self, rate, stepSize)
        self.trajectory = [START_STATE]
        self.reward = 0.0
        self.rateTruncate = 1e-3

    def learn(self, state, reward):
        self.trajectory.append(state)
        if state in END_STATES:
            self.reward = reward
            self.T = len(self.trajectory) - 1
            self.offLineLearn()

    def nStepReturnFromTime(self, n, time):
        # gamma is always 1
        endTime = min(time + n, self.T)
        returns = self.value(self.trajectory[endTime])
        if endTime == self.T:
            returns += self.reward
        return returns

    def lambdaReturnFromTime(self, time):
        returns = 0.0
        lambdaPower = 1
        for n in range(1, self.T - time):
            returns += lambdaPower * self.nStepReturnFromTime(n, time)
            lambdaPower *= self.rate
            if lambdaPower < self.rateTruncate:
                break
        returns *= 1 - self.rate
        if lambdaPower >= self.rateTruncate:
            returns += lambdaPower * self.reward
        return returns

    def offLineLearn(self):
        for time in range(self.T):
            state = self.trajectory[time]
            delta = self.lambdaReturnFromTime(time) - self.value(state)
            delta *= self.stepSize
            activeTiles = self.getActiveTiles(state)
            for activeTile in activeTiles:
                self.weights[activeTile] += delta
        self.trajectory = []

def randomWalk(valueFunction):
    currentState = START_STATE
    while currentState not in END_STATES:
        newState = currentState + np.random.choice([-1, 1])
        if newState == 0:
            reward = -1
        elif newState == N_STATES + 1:
            reward = 1
        else:
            reward = 0
        valueFunction.learn(newState, reward)
        currentState = newState

figureIndex = 0
def figure12(valueFunctionGenerator, runs):
    global figureIndex
    truncateValue = 0.55
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = np.arange(0, 1.1, 0.1)

    episodes = 10
    errors = np.zeros((len(lambdas), len(alphas)))
    for run in range(runs):
        for lambdaIndex, rate in zip(range(len(lambdas)), lambdas):
            for alphaIndex, alpha in zip(range(len(alphas)), alphas):
                valueFunction = valueFunctionGenerator(rate, alpha)
                for episode in range(episodes):
                    print 'run:', run, 'lambda:', rate, 'alpha:', alpha, 'episode:', episode
                    randomWalk(valueFunction)
                    stateValues = [valueFunction.value(state) for state in states]
                    errors[lambdaIndex, alphaIndex] += np.sqrt(np.mean(np.power(stateValues - realStateValues[1: -1], 2)))

    errors /= episodes * runs
    errors[errors > truncateValue] = truncateValue
    plt.figure(figureIndex)
    figureIndex += 1
    for i in range(len(lambdas)):
        plt.plot(alphas, errors[i, :], label='lambda = ' + str(lambdas[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.legend()

def figure12_3():
    figure12(OffLineLambdaReturn, 10)

figure12_3()
plt.show()


