#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import functools

# 19-state random walk
N_STATES = 19

# undiscounted
GAMMA = 1

stateValues = np.zeros(N_STATES + 2)

# all states except for terminal states
states = np.arange(1, N_STATES + 1)

# start from the middle
START_STATE = N_STATES // 2 + 1
END_STATES = [0, N_STATES + 1]

# add an extra action STAY besides LEFT and RIGHT
ACTIONS = [-1, 0, 1]
# probability of each action
ACTIONS_PROB = np.asarray([0.25, 0.5, 0.25])

# use DP to get the true state value
trueStateValues = np.copy(stateValues)
trueStateValues[0] = -1.0
trueStateValues[-1] = 1.0
while True:
    delta = 0.0
    for state in states:
        newStateValue = np.sum(ACTIONS_PROB * [trueStateValues[state + action] for action in ACTIONS])
        delta += np.abs(newStateValue - trueStateValues[state])
        trueStateValues[state] = newStateValue
    if delta < 1e-3:
        break
trueStateValues[0] = trueStateValues[-1] = 0
print(np.sqrt(np.mean(np.power(trueStateValues[1: -1] - stateValues[1: -1], 2))))

# go to next state
def nextStep(state):
    newState = state + np.random.choice(ACTIONS, p=ACTIONS_PROB)
    if newState == 0:
        reward = -1.0
    elif newState == N_STATES + 1:
        reward = 1.0
    else:
        reward = 0.0
    return newState, reward

# n-step TD algorithm
# @sumOfTDErrors: False if use n-step TD error
#                 True if use sum of n TD errors
def temproalDifference(stateValues, n, alpha, sumOfTDErrors=False):
    currentState = START_STATE
    states = [currentState]
    rewards = [0.0]

    time = 0
    T = float('inf')

    while True:
        time += 1
        if time < T:
            newState, reward = nextStep(currentState)
            states.append(newState)
            rewards.append(reward)

            if newState in END_STATES:
                T = time

        updateTime = time - n
        if updateTime >= 0:
            stateToUpdate = states[updateTime]
            if sumOfTDErrors:
                # make a copy of current state value
                shadowStateValues = np.copy(stateValues)
                errors = 0.0
                # perform n TD updates on the copy, get the cumulative TD error
                for t in range(updateTime, min(T, updateTime + n)):
                    delta = rewards[t + 1] + shadowStateValues[states[t + 1]] - \
                        shadowStateValues[states[t]]
                    errors += delta
                    shadowStateValues[states[t]] += alpha * delta
            else:
                # n-step TD error
                returns = 0.0
                returns += np.sum(rewards[updateTime + 1: min(T, updateTime + n) + 1])
                if updateTime + n <= T:
                    returns += stateValues[states[updateTime + n]]
                errors = returns - stateValues[stateToUpdate]
            # update the state value
            stateValues[stateToUpdate] += alpha * errors
        if updateTime == T - 1:
            break
        currentState = newState

def figure():
    runs = 100
    episodes = 50
    alpha = 0.1
    labels = ['sum of n TD errors', 'n-step TD error']
    methods = [functools.partial(temproalDifference, n=4, alpha=alpha, sumOfTDErrors=True),
               functools.partial(temproalDifference, n=4, alpha=alpha, sumOfTDErrors=False)]
    errors = np.zeros((len(methods), episodes))
    for run in range(runs):
        for index, method in zip(range(len(methods)), methods):
            # set random seed to make sure the trajectory is the same for both algorithms
            np.random.seed(run)
            currentStateValues = np.copy(stateValues)
            for episode in range(episodes):
                print('run:', run, 'episode:', episode)
                method(currentStateValues)
                errors[index, episode] += np.sqrt(np.mean(np.power(currentStateValues[1: -1] - trueStateValues[1: -1], 2)))
    errors /= runs

    plt.figure()
    for i in range(len(labels)):
        plt.plot(errors[i], label=labels[i])
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.legend()

figure()
plt.show()

