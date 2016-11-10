#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ACTION_REVERSE = -1
ACTION_ZERO = 0
ACTION_FORWARD = 1
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]

POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

EPSILON = 0

def takeAction(position, velocity, action):
    newVelocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    newVelocity = min(max(VELOCITY_MIN, newVelocity), VELOCITY_MAX)
    newPosition = position + newVelocity
    newPosition = min(max(POSITION_MIN, newPosition), POSITION_MAX)
    reward = -1.0
    if newPosition == POSITION_MIN:
        newVelocity = 0.0
    return newPosition, newVelocity, reward

class ValueFunction:
    def __init__(self, stepSize, numOfTilings=8, maxSize=2048):
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings
        self.stepSize = stepSize / numOfTilings
        self.hashTable = IHT(maxSize)
        self.weights = np.zeros(maxSize)
        self.positionScale = self.numOfTilings / (POSITION_MAX - POSITION_MIN)
        self.velocityScale = self.numOfTilings / (VELOCITY_MAX - VELOCITY_MIN)

    def getActiveTiles(self, position, velocity, action):
        activeTiles = tiles(self.hashTable, self.numOfTilings,
                            [self.positionScale * position, self.velocityScale * velocity],
                            [action])
        return activeTiles

    def value(self, position, velocity, action):
        if position == POSITION_MAX:
            return 0.0
        activeTiles = self.getActiveTiles(position, velocity, action)
        return np.sum(self.weights[activeTiles])

    def learn(self, position, velocity, action, target):
        activeTiles = self.getActiveTiles(position, velocity, action)
        estimation = np.sum(self.weights[activeTiles])
        delta = self.stepSize * (target - estimation)
        for activeTile in activeTiles:
            self.weights[activeTile] += delta

    def costToGo(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)

def getAction(position, velocity, valueFunction):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(valueFunction.value(position, velocity, action))
    return argmax(values) - 1

def semiGradientNStepSarsa(valueFunction, n=1):
    currentPosition = np.random.uniform(-0.6, -0.4)
    currentVelocity = 0.0
    currentAction = getAction(currentPosition, currentVelocity, valueFunction)

    positions = [currentPosition]
    velocities = [currentVelocity]
    actions = [currentAction]
    rewards = [0.0]

    time = 0
    T = float('inf')

    while True:
        time += 1
        # print 'step:', time
        # if time == 42800:
        #     return

        if time < T:
            newPostion, newVelocity, reward = takeAction(currentPosition, currentVelocity, currentAction)
            newAction = getAction(newPostion, newVelocity, valueFunction)

            positions.append(newPostion)
            velocities.append(newVelocity)
            actions.append(newAction)
            rewards.append(reward)

            if newPostion == POSITION_MAX:
                T = time

        updateTime = time - n
        if updateTime >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(updateTime + 1, min(T, updateTime + n) + 1):
                returns += rewards[t]
            # add state value to the return
            if updateTime + n <= T:
                returns += valueFunction.value(positions[updateTime + n],
                                               velocities[updateTime + n],
                                               actions[updateTime + n])
            # update the value function
            if positions[updateTime] != POSITION_MAX:
                valueFunction.learn(positions[updateTime], velocities[updateTime], actions[updateTime], returns)
        if updateTime == T - 1:
            break
        currentPosition = newPostion
        currentVelocity = newVelocity
        currentAction = newAction

    return time

figureIndex = 0
def prettyPrint(valueFunction, title):
    global figureIndex
    gridSize = 40
    positionStep = (POSITION_MAX - POSITION_MIN) / gridSize
    positions = np.arange(POSITION_MIN, POSITION_MAX + positionStep, positionStep)
    velocityStep = (VELOCITY_MAX - VELOCITY_MIN) / gridSize
    velocities = np.arange(VELOCITY_MIN, VELOCITY_MAX + velocityStep, velocityStep)
    axisX = []
    axisY = []
    axisZ = []
    for position in positions:
        for velocity in velocities:
            axisX.append(position)
            axisY.append(velocity)
            axisZ.append(valueFunction.costToGo(position, velocity))

    fig = plt.figure(figureIndex)
    figureIndex += 1
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(axisX, axisY, axisZ)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')


def figure10_1():
    episodes = 9000
    targetEpisodes = [0, 99, episodes - 1]
    numOfTilings = 8
    alpha = 0.3
    valueFunction = ValueFunction(alpha, numOfTilings)
    for episode in range(0, episodes):
        print 'episode:', episode
        semiGradientNStepSarsa(valueFunction)
        if episode in targetEpisodes:
            prettyPrint(valueFunction, 'Episode: ' + str(episode + 1))

figure10_1()
plt.show()




