#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from utils import *
import matplotlib.pyplot as plt

# maze width
WORLD_WIDTH = 9

# maze height
WORLD_HEIGHT = 6

# start state
START_STATE = [2, 0]

# goal state
GOAL_STATE = [0, 8]

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# all obstacles
obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]

# take @action in @state
# @return: [new state, reward]
def takeAction(state, action):
    x, y = state
    if action == ACTION_UP:
        x = max(x - 1, 0)
    elif action == ACTION_DOWN:
        x = min(x + 1, WORLD_HEIGHT - 1)
    elif action == ACTION_LEFT:
        y = max(y - 1, 0)
    elif action == ACTION_RIGHT:
        y = min(y + 1, WORLD_WIDTH - 1)
    if [x, y] in obstacles:
        x, y = state
    if [x, y] == GOAL_STATE:
        reward = 1.0
    else:
        reward = 0.0
    return [x, y], reward

# initial state action pair values
stateActionValues = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(actions)))

# discount
GAMMA = 0.95

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.1

# choose an action based on epsilon-greedy algorithm
def chooseAction(state, stateActionValues):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(actions)
    else:
        return argmax(stateActionValues[state[0], state[1], :])

# Trivial model for planning in Dyna-Q
class TrivialModel:

    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, currentState, action, newState, reward):
        if tuple(currentState) not in self.model.keys():
            self.model[tuple(currentState)] = dict()
        self.model[tuple(currentState)][action] = [newState, reward]

    # randomly sampling from previous experience
    def sample(self):
        stateIndex = self.rand.choice(range(0, len(self.model.keys())))
        state = self.model.keys()[stateIndex]
        actionIndex = self.rand.choice(range(0, len(self.model[state].keys())))
        action = self.model[state].keys()[actionIndex]
        newState, reward = self.model[state][action]
        return list(state), action, newState, reward

# play for an episode for Dyna-Q algorithm
# @stateActionValues: state action pair values, will be updated
# @model: model instance for planning
# @planningSteps: steps for planning
def dynaQ(stateActionValues, model, planningSteps):
    currentState = START_STATE
    steps = 0
    while currentState != GOAL_STATE:
        # track the steps
        steps += 1

        # get action
        action = chooseAction(currentState, stateActionValues)

        # take action
        newState, reward = takeAction(currentState, action)

        # Q-Learning update
        stateActionValues[currentState[0], currentState[1], action] += \
            ALPHA * (reward + GAMMA * np.max(stateActionValues[newState[0], newState[1], :]) -
            stateActionValues[currentState[0], currentState[1], action])

        # feed the model with experience
        model.feed(currentState, action, newState, reward)

        # sample experience from the model
        for t in range(0, planningSteps):
            stateSample, actionSample, newStateSample, rewardSample = model.sample()
            stateActionValues[stateSample[0], stateSample[1], actionSample] += \
                ALPHA * (rewardSample + GAMMA * np.max(stateActionValues[newStateSample[0], newStateSample[1], :]) -
                stateActionValues[stateSample[0], stateSample[1], actionSample])

        currentState = newState
    return steps

# Figure 8.3, use 10 runs instead of 30 runs
def figure8_3():
    runs = 10
    episodes = 50
    planningSteps = [0, 5, 50]
    steps = np.zeros((len(planningSteps), episodes))

    # this random seed is for sampling from model
    # we do need this separate random seed to make sure the first episodes for all planning steps are the same
    rand = np.random.RandomState(0)

    for run in range(0, runs):
        for index, planningStep in zip(range(0, len(planningSteps)), planningSteps):
            # set same random seed for each planning step
            np.random.seed(run)

            currentStateActionValues = np.copy(stateActionValues)

            # generate an instance of Dyna-Q model
            model = TrivialModel(rand)
            for ep in range(0, episodes):
                print 'run:', run, 'planning step:', planningStep, 'episode:', ep
                steps[index, ep] += dynaQ(currentStateActionValues, model, planningStep)

    # averaging over runs
    steps /= runs

    plt.figure(0)
    for i in range(0, len(planningSteps)):
        plt.plot(range(0, episodes), steps[i, :], label=str(planningSteps[i]) + ' planning steps')
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()

figure8_3()
plt.show()