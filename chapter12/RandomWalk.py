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
    def __init__(self, rate, stepSize, numOfTilings=8):
        self.rate = rate
        self.stepSize = stepSize
        self.weights = np.zeros(N_STATES + 2)

    def value(self, state):
        return self.weights[state]

    def learn(self, state, reward):
        return

class OffLineLambdaReturn(ValueFunction):
    def __init__(self, rate, stepSize):
        ValueFunction.__init__(self, rate, stepSize)
        self.rateTruncate = 1e-3

    def newEpisode(self):
        self.trajectory = [START_STATE]
        self.reward = 0.0

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
            self.weights[state] += delta

class TemporalDifferenceLambda(ValueFunction):
    def __init__(self, rate, stepSize):
        ValueFunction.__init__(self, rate, stepSize)
        self.newEpisode()

    def newEpisode(self):
        self.eligibility = np.zeros(N_STATES + 2)
        self.lastState = START_STATE

    def value(self, state):
        return self.weights[state]

    def learn(self, state, reward):
        self.eligibility *= self.rate
        self.eligibility[self.lastState] += 1
        delta = reward + self.value(state) - self.value(self.lastState)
        delta *= self.stepSize
        self.weights += delta * self.eligibility
        self.lastState = state

def randomWalk(valueFunction):
    valueFunction.newEpisode()
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
def figure12(valueFunctionGenerator, runs, lambdas, alphas):
    global figureIndex

    episodes = 10
    errors = [np.zeros(len(alphas_)) for alphas_ in alphas]
    for run in range(runs):
        for lambdaIndex, rate in zip(range(len(lambdas)), lambdas):
            for alphaIndex, alpha in zip(range(len(alphas[lambdaIndex])), alphas[lambdaIndex]):
                valueFunction = valueFunctionGenerator(rate, alpha)
                for episode in range(episodes):
                    print 'run:', run, 'lambda:', rate, 'alpha:', alpha, 'episode:', episode
                    randomWalk(valueFunction)
                    stateValues = [valueFunction.value(state) for state in states]
                    errors[lambdaIndex][alphaIndex] += np.sqrt(np.mean(np.power(stateValues - realStateValues[1: -1], 2)))

    for error in errors:
        error /= episodes * runs
    plt.figure(figureIndex)
    figureIndex += 1
    for i in range(len(lambdas)):
        plt.plot(alphas[i], errors[i], label='lambda = ' + str(lambdas[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.legend()

def figure12_3():
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01)]
    figure12(OffLineLambdaReturn, 50, lambdas, alphas)

def figure12_6():
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.99, 0.09),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.33, 0.03),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01),
              np.arange(0, 0.044, 0.004)]
    figure12(TemporalDifferenceLambda, 50, lambdas, alphas)

figure12_3()
# figure12_6()
plt.show()


