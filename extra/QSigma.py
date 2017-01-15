#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
from utils.utils import *
import matplotlib.pyplot as plt
import functools

# This example is the windy grid world.
# It will compare the performance of Q-Learning, n-step tree backup and n-step Q(sigma)
# I use an extreme case for n-step tree backup and n-step Q(sigma).
# The target policy is greedy policy, the behavior policy is epsilon-greedy policy
# Under this setting, n-step tree backup is simplified significantly
# It looks like "n-step Q-Learning" to some extent

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# reward for each step
REWARD = -1.0

# state action pair value
stateActionValues = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
startState = [3, 0]
goalState = [3, 7]

# set up destinations for each action in each state
actionDestination = []
for i in range(0, WORLD_HEIGHT):
    actionDestination.append([])
    for j in range(0, WORLD_WIDTH):
        destination = dict()
        destination[ACTION_UP] = [max(i - 1 - WIND[j], 0), j]
        destination[ACTION_DOWN] = [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
        destination[ACTION_LEFT] = [max(i - WIND[j], 0), max(j - 1, 0)]
        destination[ACTION_RIGHT] = [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
        actionDestination[-1].append(destination)

def takeAction(state, action):
    x, y = state
    return actionDestination[x][y][action], REWARD

# behavior policy is epsilon-greedy
EPSILON = 0.2
# the probability of choosing best action under epsilon-greedy policy
BEST_ACTION_PROB = 1 - EPSILON + EPSILON / len(ACTIONS)
def behaviorPolicy(state, stateActionValues):
    x, y = state
    if state == goalState or np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        return argmax(stateActionValues[x, y, :])

# play for an episode for Q-learning
def qLearning(stateActionValues, alpha):
    # track the total time steps in this episode
    time = 0

    # initialize state
    currentState = startState
    currentAction = behaviorPolicy(currentState, stateActionValues)

    # keep going until get to the goal state
    while currentState != goalState:
        newState = actionDestination[currentState[0]][currentState[1]][currentAction]
        newAction = behaviorPolicy(newState, stateActionValues)
        stateActionValues[currentState[0], currentState[1], currentAction] += \
            alpha * (REWARD + np.max(stateActionValues[newState[0], newState[1], :]) -
            stateActionValues[currentState[0], currentState[1], currentAction])
        currentState = newState
        currentAction = newAction
        time += 1
    return time

# play for an episode for n-step tree backup
# Note this isn't a general function for all n-step tree backup algorithms
def nStepTreeBackup(stateActionValues, n, alpha):
    time = 0
    currentState = startState
    currentAction = behaviorPolicy(currentState, stateActionValues)
    T = float('inf')
    trajectories = [(currentState, currentAction, 0)]
    while True:
        time += 1
        if time < T:
            newState, reward = takeAction(currentState, currentAction)
            newAction = behaviorPolicy(newState, stateActionValues)
            trajectories.append((newState, newAction, reward))
            if newState == goalState:
                T = time
        updateTime = time - n
        if updateTime >= 0:
            target = 0.0
            coef = 1
            for t in range(updateTime + 1, min(T, updateTime + n)):
                (x, y), action, reward = trajectories[t]
                bestAction = argmax(stateActionValues[x, y, :])
                if action == bestAction:
                    target += coef * reward
                else:
                    target += coef * stateActionValues[x, y, bestAction]
                    coef = 0
            if updateTime + n < T:
                x, y = trajectories[updateTime + n][0]
                target += coef * np.max(stateActionValues[x, y, :])
            state, action, _ = trajectories[updateTime]
            if state != goalState:
                x, y = state
                stateActionValues[x, y, action] += alpha * (target - stateActionValues[x, y, action])
        if updateTime == T - 1:
            break
        currentState = newState
        currentAction = newAction
    return time

# play for an episode for n-step Q(sigma)
# This is a general function for n-step Q(sigma) under arbitrary policies
# This is implemented according to the algorithm box of n-step Q(sigma) in the book,
# so there aren't detailed comments.
# @sigmaFn: a function to generate sigma
def nStepQSigma(stateActionValues, n, alpha, sigmaFn):
    time = 0
    x, y = currentState = startState
    currentAction = behaviorPolicy(currentState, stateActionValues)
    T = float('inf')

    states = [currentState]
    actions = [currentAction]
    values = [stateActionValues[x, y, currentAction]]
    sigmas = [0]
    deltas = []
    PIs = [0]
    RHOs = [0]

    while True:
        if time < T:
            newState, reward = takeAction(currentState, currentAction)
            states.append(newState)
            if newState == goalState:
                T = time + 1
                deltas.append(reward - values[time])
                # append dummy sigma and pi
                sigmas.append(0.0)
                PIs.append(0.0)
            else:
                x, y = newState
                newAction = behaviorPolicy(newState, stateActionValues)
                actions.append(newAction)
                sigma = sigmaFn()
                sigmas.append(sigma)
                values.append(stateActionValues[x, y, newAction])
                deltas.append(reward + sigma * values[-1] +
                              (1 - sigma) * np.max(stateActionValues[x, y, :]) - values[-2])
                if newAction == np.argmax(stateActionValues[x, y, :]):
                    PIs.append(1.0)
                    RHOs.append(1.0 / BEST_ACTION_PROB)
                else:
                    PIs.append(0.0)
                    RHOs.append(0.0)
        updateTime = time - n + 1
        if updateTime >= 0:
            target = values[updateTime]
            coef = 1
            rho = 1
            for t in range(updateTime, min(updateTime + n, T)):
                target += coef * deltas[t]
                coef *= (1 - sigmas[t + 1]) * PIs[t + 1] + sigmas[t + 1]
                rho *= 1 - sigmas[t] + sigmas[t] * RHOs[t]
            x, y = states[updateTime]
            action = actions[updateTime]
            stateActionValues[x, y, action] += alpha * rho * (target - stateActionValues[x, y, action])
        if updateTime == T - 1:
            break
        time += 1
        currentState = newState
        currentAction = newAction
    return time

# print out the optimal policy
def printPolicy(stateActionValues):
    optimalPolicy = []
    for i in range(0, WORLD_HEIGHT):
        optimalPolicy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == goalState:
                optimalPolicy[-1].append('G')
                continue
            bestAction = argmax(stateActionValues[i, j, :])
            if bestAction == ACTION_UP:
                optimalPolicy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimalPolicy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimalPolicy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimalPolicy[-1].append('R')
    for row in optimalPolicy:
        print(row)
    print([str(w) for w in WIND])

def play(method, nEpisodes, alpha):
    episodes = []
    ep = 0
    currentStateActionValues = np.copy(stateActionValues)
    while ep < nEpisodes:
        time = method(currentStateActionValues, alpha=alpha)
        episodes.extend([ep] * time)
        ep += 1
    return np.asarray(episodes)

def figure():
    # simply generate a sigma for each step from a uniform distribution
    sigmaFn = lambda : np.random.rand()
    methods = [qLearning, functools.partial(nStepTreeBackup, n=3), functools.partial(nStepQSigma, n=3, sigmaFn=sigmaFn)]
    labels = ['Q-Learning', '3-Step Tree Backup', '3-Step Q(sigma)']

    # Use a small step size,
    # Although in deterministic environment, alpha = 1 is the optimal selection
    alphas = [0.1, 0.1, 0.1]

    # run each algorithm for certain episodes
    nEpisodes = 120

    # perform several independent runs
    runs = 50

    empiricalLength = 20000
    episodes = np.zeros((len(methods), empiricalLength))
    minLength = empiricalLength
    for run in range(runs):
        print('run:', run)
        for index, method in zip(range(len(methods)), methods):
            episodes_ = play(method, nEpisodes, alphas[index])
            minLength = min(minLength, len(episodes_))
            episodes[index, : minLength] += episodes_[: minLength]
    episodes /= runs
    plt.figure()
    for i in range(len(labels)):
        plt.plot(episodes[i, : minLength], label=labels[i])
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.legend()

figure()
plt.show()
