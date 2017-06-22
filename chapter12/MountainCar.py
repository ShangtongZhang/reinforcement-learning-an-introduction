#######################################################################
# Copyright (C)                                                       #
# 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from TileCoding import *
import pickle

# all possible actions
ACTION_REVERSE = -1
ACTION_ZERO = 0
ACTION_FORWARD = 1
# order is important
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# discount is always 1.0 in these experiments
DISCOUNT = 1.0

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0

# maximum steps per episode
STEP_LIMIT = 5000

# take an @action at @position and @velocity
# @return: new position, new velocity, reward (always -1)
def takeAction(position, velocity, action):
    newVelocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    newVelocity = min(max(VELOCITY_MIN, newVelocity), VELOCITY_MAX)
    newPosition = position + newVelocity
    newPosition = min(max(POSITION_MIN, newPosition), POSITION_MAX)
    reward = -1.0
    if newPosition == POSITION_MIN:
        newVelocity = 0.0
    return newPosition, newVelocity, reward

# I use underline name convention for the following for trace functions to make use of '__name__'

# accumulating trace update rule
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @return: new trace for convenience
def accumulating_trace(trace, activeTiles, lam):
    trace *= lam * DISCOUNT
    trace[activeTiles] += 1
    return trace

# replacing trace update rule
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @return: new trace for convenience
def replacing_trace(trace, activeTiles, lam):
    active = np.in1d(np.arange(len(trace)), activeTiles)
    trace[active] = 1
    trace[~active] *= lam * DISCOUNT
    return trace

# replacing trace update rule, 'clearing' means set all tiles corresponding to non-selected actions to 0
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @clearingTiles: tiles to be cleared
# @return: new trace for convenience
def replacing_trace_with_clearing(trace, activeTiles, lam, clearingTiles):
    active = np.in1d(np.arange(len(trace)), activeTiles)
    trace[~active] *= lam * DISCOUNT
    trace[clearingTiles] = 0
    trace[active] = 1
    return trace

# dutch trace update rule
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @alpha: step size for all tiles
# @return: new trace for convenience
def dutch_trace(trace, activeTiles, lam, alpha):
    coef = 1 - alpha * DISCOUNT * lam * np.sum(trace[activeTiles])
    trace *= DISCOUNT * lam
    trace[activeTiles] += coef
    return trace

# wrapper class for Sarsa(lambda)
class Sarsa:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @maxSize: the maximum # of indices
    def __init__(self, stepSize, lam, traceUpdate=accumulating_trace, numOfTilings=8, maxSize=2048):
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings
        self.traceUpdate = traceUpdate
        self.lam = lam

        # divide step size equally to each tiling
        self.stepSize = stepSize / numOfTilings

        self.hashTable = IHT(maxSize)

        # weight for each tile
        self.weights = np.zeros(maxSize)

        # trace for each tile
        self.trace = np.zeros(maxSize)

        # position and velocity needs scaling to satisfy the tile software
        self.positionScale = self.numOfTilings / (POSITION_MAX - POSITION_MIN)
        self.velocityScale = self.numOfTilings / (VELOCITY_MAX - VELOCITY_MIN)

    # get indices of active tiles for given state and action
    def getActiveTiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        activeTiles = tiles(self.hashTable, self.numOfTilings,
                            [self.positionScale * position, self.velocityScale * velocity],
                            [action])
        return activeTiles

    # estimate the value of given state and action
    def value(self, position, velocity, action):
        if position == POSITION_MAX:
            return 0.0
        activeTiles = self.getActiveTiles(position, velocity, action)
        return np.sum(self.weights[activeTiles])

    # learn with given state, action and target
    def learn(self, position, velocity, action, target):
        activeTiles = self.getActiveTiles(position, velocity, action)
        estimation = np.sum(self.weights[activeTiles])
        delta = target - estimation
        if self.traceUpdate == accumulating_trace or self.traceUpdate == replacing_trace:
            self.traceUpdate(self.trace, activeTiles, self.lam)
        elif self.traceUpdate == dutch_trace:
            self.traceUpdate(self.trace, activeTiles, self.lam, self.stepSize)
        elif self.traceUpdate == replacing_trace_with_clearing:
            clearingTiles = []
            for act in ACTIONS:
                if act != action:
                    clearingTiles.extend(self.getActiveTiles(position, velocity, act))
            self.traceUpdate(self.trace, activeTiles, self.lam, clearingTiles)
        else:
            raise Exception('Unexpected Trace Type')
        self.weights += self.stepSize * delta * self.trace

    # get # of steps to reach the goal under current state value function
    def costToGo(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)

# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def getAction(position, velocity, valueFunction):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(valueFunction.value(position, velocity, action))
    return argmax(values) - 1

# play Mountain Car for one episode based on given method @evaluator
# @return: total steps in this episode
def play(evaluator):
    position = np.random.uniform(-0.6, -0.4)
    velocity = 0.0
    action = getAction(position, velocity, evaluator)
    steps = 0
    while True:
        nextPosition, nextVelocity, reward = takeAction(position, velocity, action)
        nextAction = getAction(nextPosition, nextVelocity, evaluator)
        steps += 1
        target = reward + DISCOUNT * evaluator.value(nextPosition, nextVelocity, nextAction)
        evaluator.learn(position, velocity, action, target)
        position = nextPosition
        velocity = nextVelocity
        action = nextAction
        if nextPosition == POSITION_MAX:
            break
        if steps >= STEP_LIMIT:
            print('Step Limit Exceeded!')
            break
    return steps

figureIndex = 0

# figure 12.10, effect of the lambda and alpha on early performance of Sarsa(lambda)
def figure12_10(load=False):
    runs = 30
    episodes = 50
    alphas = np.arange(1, 8) / 4.0
    lams = [0.99, 0.95, 0.5, 0]

    if load:
        with open('figure12_10.bin', 'rb') as f:
            steps = pickle.load(f)
    else:
        steps = np.zeros((len(lams), len(alphas), runs, episodes))
        for lamInd, lam in enumerate(lams):
            for alphaInd, alpha in enumerate(alphas):
                for run in range(runs):
                    evaluator = Sarsa(alpha, lam, replacing_trace)
                    for ep in range(episodes):
                        step = play(evaluator)
                        steps[lamInd, alphaInd, run, ep] = step
                        print('lambda %f, alpha %f, run %d, episode %d, steps %d' %
                              (lam, alpha, run, ep, step))
        with open('figure12_9.bin', 'wb') as f:
            pickle.dump(steps, f)

    # average over episodes
    steps = np.mean(steps, axis=3)

    # average over runs
    steps = np.mean(steps, axis=2)

    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    for lamInd, lam in enumerate(lams):
        plt.plot(alphas, steps[lamInd, :], label='lambda = %s' % (str(lam)))
    plt.xlabel('alpha * # of tilings (8)')
    plt.ylabel('averaged steps per episode')
    plt.ylim([180, 300])
    plt.legend()

# figure 12.11, summary comparision of Sarsa(lambda) algorithms
# I use 8 tilings rather than 10 tilings
def figure12_11(load=False):
    traceTypes = [dutch_trace, replacing_trace, replacing_trace_with_clearing, accumulating_trace]
    alphas = np.arange(0.2, 2.2, 0.2)
    episodes = 20
    runs = 30
    lam = 0.9
    rewards = np.zeros((len(traceTypes), len(alphas), runs, episodes))

    if load:
        with open('figure12_11.bin', 'rb') as f:
            rewards = pickle.load(f)
    else:
        for traceInd, trace in enumerate(traceTypes):
            for alphaInd, alpha in enumerate(alphas):
                for run in range(runs):
                    evaluator = Sarsa(alpha, lam, trace)
                    for ep in range(episodes):
                        if trace == accumulating_trace and alpha > 0.6:
                            steps = STEP_LIMIT
                        else:
                            steps = play(evaluator)
                        rewards[traceInd, alphaInd, run, ep] = -steps
                        print('%s, step size %f, run %d, episode %d, rewards %d' %
                              (trace.__name__, alpha, run, ep, -steps))
        with open('figure12_10.bin', 'wb') as f:
            pickle.dump(rewards, f)

    # average over episodes
    rewards = np.mean(rewards, axis=3)

    # average over runs
    rewards = np.mean(rewards, axis=2)

    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    for traceInd, trace in enumerate(traceTypes):
        plt.plot(alphas, rewards[traceInd, :], label=trace.__name__)
    plt.xlabel('alpha * # of tilings (8)')
    plt.ylabel('averaged rewards pre episode')
    plt.ylim([-550, -150])
    plt.legend()

if __name__ == '__main__':
    figure12_10()
    figure12_11()
    plt.show()




