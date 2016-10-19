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

        # initial state action pair values
        self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # Following are some parameters for algorithms, maybe they shouldn't be enclosed in this class
        # However I'm lazy and don't want to create a separate class and pass it as parameter
        # to every function again

        # discount
        self.GAMMA = 0.95

        # probability for exploration
        self.EPSILON = 0.1

        # step size
        self.ALPHA = 0.1

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

# choose an action based on epsilon-greedy algorithm
def chooseAction(state, stateActionValues, maze):
    if np.random.binomial(1, maze.EPSILON) == 1:
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
def dynaQ(stateActionValues, model, planningSteps, maze):
    currentState = maze.START_STATE
    steps = 0
    while currentState != maze.GOAL_STATE:
        # track the steps
        steps += 1

        # get action
        action = chooseAction(currentState, stateActionValues, maze)

        # take action
        newState, reward = maze.takeAction(currentState, action)

        # Q-Learning update
        stateActionValues[currentState[0], currentState[1], action] += \
            maze.ALPHA * (reward + maze.GAMMA * np.max(stateActionValues[newState[0], newState[1], :]) -
            stateActionValues[currentState[0], currentState[1], action])

        # feed the model with experience
        model.feed(currentState, action, newState, reward)

        # sample experience from the model
        for t in range(0, planningSteps):
            stateSample, actionSample, newStateSample, rewardSample = model.sample()
            stateActionValues[stateSample[0], stateSample[1], actionSample] += \
                maze.ALPHA * (rewardSample + maze.GAMMA * np.max(stateActionValues[newStateSample[0], newStateSample[1], :]) -
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

            currentStateActionValues = np.copy(dynaMaze.stateActionValues)

            # generate an instance of Dyna-Q model
            model = TrivialModel(rand)
            for ep in range(0, episodes):
                print 'run:', run, 'planning step:', planningStep, 'episode:', ep
                steps[index, ep] += dynaQ(currentStateActionValues, model, planningStep, dynaMaze)

    # averaging over runs
    steps /= runs

    plt.figure(0)
    for i in range(0, len(planningSteps)):
        plt.plot(range(0, episodes), steps[i, :], label=str(planningSteps[i]) + ' planning steps')
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()

# Figure 8.5, BlockingMaze
def figure8_5():

    # set up a blocking maze instance
    blockingMaze = Maze()
    blockingMaze.START_STATE = [5, 3]
    blockingMaze.GOAL_STATE = [0, 8]
    oldObstacles = [[3, i] for i in range(0, 8)]
    newObstacles = [[3, i] for i in range(1, 9)]

    # it's a tricky alpha...
    blockingMaze.ALPHA = 0.7

    # set up max steps
    maxSteps = 3000
    blockingMaze.maxSteps = maxSteps

    # obstacles will change after 1000 steps
    # the exact step for changing will be different
    # However given that 1000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    blockingPoints = 1000

    # planning step
    planningSteps = 5

    # average over 20 runs
    runs = 20

    # track the cumulative rewards
    rewards = np.zeros((2, maxSteps))

    for run in range(0, runs):
        methods = ['Dyna-Q', 'Dyna-Q+']

        # set up models
        models = [TrivialModel(), TimeModel(blockingMaze)]

        # track cumulative reward in current run
        rewards_ = np.zeros((2, maxSteps))

        # initialize state action values
        stateActionValues = [np.copy(blockingMaze.stateActionValues), np.copy(blockingMaze.stateActionValues)]

        for i in range(0, len(methods)):
            print 'run:', run, methods[i]

            # set old obstacles for the maze
            blockingMaze.obstacles = oldObstacles

            steps = 0
            lastSteps = steps
            while steps < maxSteps:
                # play for an episode
                steps += dynaQ(stateActionValues[i], models[i], planningSteps, blockingMaze)

                # update cumulative rewards
                steps_ = min(steps, maxSteps - 1)
                rewards_[i, lastSteps: steps_] = rewards_[i, lastSteps]
                rewards_[i, steps_] = rewards_[i, lastSteps] + 1
                lastSteps = steps

                if steps > blockingPoints:
                    # change the obstacles
                    blockingMaze.obstacles = newObstacles
        rewards += rewards_

    # averaging over runs
    rewards /= runs

    plt.figure(1)
    for i in range(0, len(methods)):
        plt.plot(range(0, maxSteps), rewards[i, :], label=methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

# figure8_3()
figure8_5()
plt.show()