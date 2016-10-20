#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from utils import *
import matplotlib.pyplot as plt

# A wrapper class for a maze, containing all the information about the maze.
# Basically it's initialized to DynaMaze by default, however it can be easily adapted
# to other maze
class Maze:

    def __init__(self):
        # maze width
        self.WORLD_WIDTH = 9

        # maze height
        self.WORLD_HEIGHT = 6

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state
        self.START_STATE = [2, 0]

        # goal state
        self.GOAL_STATE = [0, 8]

        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.oldObstacles = None
        self.newObstacles = None

        # time to change obstacles
        self.changingPoint = None

        # initial state action pair values
        self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # max steps
        self.maxSteps = float('inf')

    # take @action in @state
    # @return: [new state, reward]
    def takeAction(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] == self.GOAL_STATE:
            reward = 1.0
        else:
            reward = 0.0
        return [x, y], reward

# a wrapper class for parameters of dyna algorithms
class DynaParams:

    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.timeWeight = 0

        # n-step planning
        self.planningSteps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']


# choose an action based on epsilon-greedy algorithm
def chooseAction(state, stateActionValues, maze, dynaParams):
    if np.random.binomial(1, dynaParams.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        return argmax(stateActionValues[state[0], state[1], :])

# Trivial model for planning in Dyna-Q
class TrivialModel:

    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, currentState, action, newState, reward):
        if tuple(currentState) not in self.model.keys():
            self.model[tuple(currentState)] = dict()
        self.model[tuple(currentState)][action] = [list(newState), reward]

    # randomly sample from previous experience
    def sample(self):
        stateIndex = self.rand.choice(range(0, len(self.model.keys())))
        state = self.model.keys()[stateIndex]
        actionIndex = self.rand.choice(range(0, len(self.model[state].keys())))
        action = self.model[state].keys()[actionIndex]
        newState, reward = self.model[state][action]
        return list(state), action, list(newState), reward

# Time-based model for planning in Dyna-Q+
class TimeModel:

    # @maze: the maze instance. Indeed it's not very reasonable to give access to maze to the model.
    # @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, maze, timeWeight=1e-4, rand=np.random):
        self.rand = rand
        self.model = dict()

        # track the total time
        self.time = 0

        self.timeWeight = timeWeight
        self.maze = maze

    # feed the model with previous experience
    def feed(self, currentState, action, newState, reward):
        self.time += 1
        if tuple(currentState) not in self.model.keys():
            self.model[tuple(currentState)] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in self.maze.actions:
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[tuple(currentState)][action_] = [list(currentState), 0, 1]

        self.model[tuple(currentState)][action] = [list(newState), reward, self.time]

    # randomly sample from previous experience
    def sample(self):
        stateIndex = self.rand.choice(range(0, len(self.model.keys())))
        state = self.model.keys()[stateIndex]
        actionIndex = self.rand.choice(range(0, len(self.model[state].keys())))
        action = self.model[state].keys()[actionIndex]
        newState, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last vist
        reward += self.timeWeight * np.sqrt(self.time - time)

        return list(state), action, list(newState), reward

# play for an episode for Dyna-Q algorithm
# @stateActionValues: state action pair values, will be updated
# @model: model instance for planning
# @planningSteps: steps for planning
# @maze: a maze instance containing all information about the environment
# @dynaParams: several params for the algorithm
def dynaQ(stateActionValues, model, maze, dynaParams):
    currentState = maze.START_STATE
    steps = 0
    while currentState != maze.GOAL_STATE:
        # track the steps
        steps += 1

        # get action
        action = chooseAction(currentState, stateActionValues, maze, dynaParams)

        # take action
        newState, reward = maze.takeAction(currentState, action)

        # Q-Learning update
        stateActionValues[currentState[0], currentState[1], action] += \
            dynaParams.alpha * (reward + dynaParams.gamma * np.max(stateActionValues[newState[0], newState[1], :]) -
            stateActionValues[currentState[0], currentState[1], action])

        # feed the model with experience
        model.feed(currentState, action, newState, reward)

        # sample experience from the model
        for t in range(0, dynaParams.planningSteps):
            stateSample, actionSample, newStateSample, rewardSample = model.sample()
            stateActionValues[stateSample[0], stateSample[1], actionSample] += \
                dynaParams.alpha * (rewardSample + dynaParams.gamma * np.max(stateActionValues[newStateSample[0], newStateSample[1], :]) -
                stateActionValues[stateSample[0], stateSample[1], actionSample])

        currentState = newState

        # check whether it has exceeded the step limit
        if steps > maze.maxSteps:
            break

    return steps

# Figure 8.3, DynaMaze, use 10 runs instead of 30 runs
def figure8_3():

    # set up an instance for DynaMaze
    dynaMaze = Maze()
    dynaParams = DynaParams()

    runs = 10
    episodes = 50
    planningSteps = [0, 5, 50]
    steps = np.zeros((len(planningSteps), episodes))

    # this random seed is for sampling from model
    # we do need this separate random seed to make sure the first episodes for all planning steps are the same
    rand = np.random.RandomState(0)

    for run in range(0, runs):
        for index, planningStep in zip(range(0, len(planningSteps)), planningSteps):
            dynaParams.planningSteps = planningStep

            # set same random seed for each planning step
            np.random.seed(run)

            currentStateActionValues = np.copy(dynaMaze.stateActionValues)

            # generate an instance of Dyna-Q model
            model = TrivialModel(rand)
            for ep in range(0, episodes):
                print 'run:', run, 'planning step:', planningStep, 'episode:', ep
                steps[index, ep] += dynaQ(currentStateActionValues, model, dynaMaze, dynaParams)

    # averaging over runs
    steps /= runs

    plt.figure(0)
    for i in range(0, len(planningSteps)):
        plt.plot(range(0, episodes), steps[i, :], label=str(planningSteps[i]) + ' planning steps')
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()

# wrapper function for changing maze
# @maze: a maze instance
# @dynaParams: several parameters for dyna algorithms
def changingMaze(maze, dynaParams):

    # set up max steps
    maxSteps = maze.maxSteps

    # track the cumulative rewards
    rewards = np.zeros((2, maxSteps))

    for run in range(0, dynaParams.runs):
        # set up models
        models = [TrivialModel(), TimeModel(maze, timeWeight=dynaParams.timeWeight)]

        # track cumulative reward in current run
        rewards_ = np.zeros((2, maxSteps))

        # initialize state action values
        stateActionValues = [np.copy(maze.stateActionValues), np.copy(maze.stateActionValues)]

        for i in range(0, len(dynaParams.methods)):
            print 'run:', run, dynaParams.methods[i]

            # set old obstacles for the maze
            maze.obstacles = maze.oldObstacles

            steps = 0
            lastSteps = steps
            while steps < maxSteps:
                # play for an episode
                steps += dynaQ(stateActionValues[i], models[i], maze, dynaParams)

                # update cumulative rewards
                steps_ = min(steps, maxSteps - 1)
                rewards_[i, lastSteps: steps_] = rewards_[i, lastSteps]
                rewards_[i, steps_] = rewards_[i, lastSteps] + 1
                lastSteps = steps

                if steps > maze.changingPoint:
                    # change the obstacles
                    maze.obstacles = maze.newObstacles
        rewards += rewards_

    # averaging over runs
    rewards /= dynaParams.runs

    return rewards

# Figure 8.5, BlockingMaze
def figure8_5():
    # set up a blocking maze instance
    blockingMaze = Maze()
    blockingMaze.START_STATE = [5, 3]
    blockingMaze.GOAL_STATE = [0, 8]
    blockingMaze.oldObstacles = [[3, i] for i in range(0, 8)]

    # new obstalces will block the optimal path
    blockingMaze.newObstacles = [[3, i] for i in range(1, 9)]

    # step limit
    blockingMaze.maxSteps = 3000

    # obstacles will change after 1000 steps
    # the exact step for changing will be different
    # However given that 1000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    blockingMaze.changingPoint = 1000

    # set up parameters
    dynaParams = DynaParams()

    # it's a tricky alpha ...
    dynaParams.alpha = 0.7

    # 5-step planning
    dynaParams.planningSteps = 5

    # average over 20 runs
    dynaParams.runs = 20

    # kappa must be small, as the reward for getting the goal is only 1
    dynaParams.timeWeight = 1e-4

    # play
    rewards = changingMaze(blockingMaze, dynaParams)

    plt.figure(1)
    for i in range(0, len(dynaParams.methods)):
        plt.plot(range(0, blockingMaze.maxSteps), rewards[i, :], label=dynaParams.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

# Figure 8.6, ShortcutMaze
def figure8_6():
    # set up a shortcut maze instance
    shortcutMaze = Maze()
    shortcutMaze.START_STATE = [5, 3]
    shortcutMaze.GOAL_STATE = [0, 8]
    shortcutMaze.oldObstacles = [[3, i] for i in range(1, 9)]

    # new obstacles will have a shorter path
    shortcutMaze.newObstacles = [[3, i] for i in range(1, 8)]

    # step limit
    shortcutMaze.maxSteps = 6000

    # obstacles will change after 3000 steps
    # the exact step for changing will be different
    # However given that 3000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    shortcutMaze.changingPoint = 3000

    # set up parameters
    dynaParams = DynaParams()

    # 50-step planning
    dynaParams.planningSteps = 50

    # average over 5 independent runs
    dynaParams.runs = 5

    # weight for elapsed time sine last visit
    dynaParams.timeWeight = 1e-3

    # also a tricky alpha ...
    dynaParams.alpha = 0.7

    # play
    rewards = changingMaze(shortcutMaze, dynaParams)

    plt.figure(2)
    for i in range(0, len(dynaParams.methods)):
        plt.plot(range(0, shortcutMaze.maxSteps), rewards[i, :], label=dynaParams.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

figure8_3()
figure8_5()
figure8_6()
plt.show()