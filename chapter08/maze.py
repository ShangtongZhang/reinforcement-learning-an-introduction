#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy

class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder

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
        self.GOAL_STATES = [[0, 8]]

        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.old_obstacles = None
        self.new_obstacles = None

        # time to change obstacles
        self.obstacle_switch_time = None

        # initial state action pair values
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

    # extend a state to a higher resolution maze
    # @state: state in lower resoultion maze
    # @factor: extension factor, one state will become factor^2 states after extension
    def extend_state(self, state, factor):
        new_state = [state[0] * factor, state[1] * factor]
        new_states = []
        for i in range(0, factor):
            for j in range(0, factor):
                new_states.append([new_state[0] + i, new_state[1] + j])
        return new_states

    # extend a state into higher resolution
    # one state in original maze will become @factor^2 states in @return new maze
    def extend_maze(self, factor):
        new_maze = Maze()
        new_maze.WORLD_WIDTH = self.WORLD_WIDTH * factor
        new_maze.WORLD_HEIGHT = self.WORLD_HEIGHT * factor
        new_maze.START_STATE = [self.START_STATE[0] * factor, self.START_STATE[1] * factor]
        new_maze.GOAL_STATES = self.extend_state(self.GOAL_STATES[0], factor)
        new_maze.obstacles = []
        for state in self.obstacles:
            new_maze.obstacles.extend(self.extend_state(state, factor))
        new_maze.q_size = (new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions))
        # new_maze.stateActionValues = np.zeros((new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions)))
        new_maze.resolution = factor
        return new_maze

    # take @action in @state
    # @return: [new state, reward]
    def step(self, state, action):
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
        if [x, y] in self.GOAL_STATES:
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
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        self.theta = 0


# choose an action based on epsilon-greedy algorithm
def choose_action(state, q_value, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

# Trivial model for planning in Dyna-Q
class TrivialModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward

# Time-based model for planning in Dyna-Q+
class TimeModel:
    # @maze: the maze instance. Indeed it's not very reasonable to give access to maze to the model.
    # @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, maze, time_weight=1e-4, rand=np.random):
        self.rand = rand
        self.model = dict()

        # track the total time
        self.time = 0

        self.time_weight = time_weight
        self.maze = maze

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        self.time += 1
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in self.maze.actions:
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[tuple(state)][action_] = [list(state), 0, 1]

        self.model[tuple(state)][action] = [list(next_state), reward, self.time]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last vist
        reward += self.time_weight * np.sqrt(self.time - time)

        state = deepcopy(state)
        next_state = deepcopy(next_state)

        return list(state), action, list(next_state), reward

# Model containing a priority queue for Prioritized Sweeping
class PriorityModel(TrivialModel):

    def __init__(self, rand=np.random):
        TrivialModel.__init__(self, rand)
        # maintain a priority queue
        self.priorityQueue = PriorityQueue()
        # track predecessors for every state
        self.predecessors = dict()

    # add a @state-@action pair into the priority queue with priority @priority
    def insert(self, priority, state, action):
        # note the priority queue is a minimum heap, so we use -priority
        self.priorityQueue.add_item((tuple(state), action), -priority)

    # @return: whether the priority queue is empty
    def empty(self):
        return self.priorityQueue.empty()

    # get the first item in the priority queue
    def sample(self):
        (state, action), priority = self.priorityQueue.pop_item()
        newState, reward = self.model[state][action]
        return -priority, list(state), action, list(newState), reward

    # feed the model with previous experience
    def feed(self, currentState, action, newState, reward):
        TrivialModel.feed(self, currentState, action, newState, reward)
        if tuple(newState) not in self.predecessors.keys():
            self.predecessors[tuple(newState)] = set()
        self.predecessors[tuple(newState)].add((tuple(currentState), action))

    # get all seen predecessors of a state @state
    def predecessor(self, state):
        if tuple(state) not in self.predecessors.keys():
            return []
        predecessors = []
        for statePre, actionPre in list(self.predecessors[tuple(state)]):
            predecessors.append([list(statePre), actionPre, self.model[statePre][actionPre][1]])
        return predecessors


# play for an episode for Dyna-Q algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
def dyna_q(q_value, model, maze, dyna_params):
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        # track the steps
        steps += 1

        # get action
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        next_state, reward = maze.step(state, action)

        # Q-Learning update
        q_value[state[0], state[1], action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                                 q_value[state[0], state[1], action])

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) -
                                     q_value[state_[0], state_[1], action_])

        state = next_state

        # check whether it has exceeded the step limit
        if steps > maze.max_steps:
            break

    return steps

# play for an episode for prioritized sweeping algorithm
# @stateActionValues: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dynaParams: several params for the algorithm
# @return: # of backups happened in planning phase in this episode
def prioritizedSweeping(stateActionValues, model, maze, dynaParams):
    currentState = maze.START_STATE

    # track the steps in this episode
    steps = 0

    # track the backups in planning phase
    backups = 0

    while currentState not in maze.GOAL_STATES:
        steps += 1

        # get action
        action = choose_action(currentState, stateActionValues, maze, dynaParams)

        # take action
        newState, reward = maze.step(currentState, action)

        # feed the model with experience
        model.feed(currentState, action, newState, reward)

        # get the priority for current state action pair
        priority = np.abs(reward + dynaParams.gamma * np.max(stateActionValues[newState[0], newState[1], :]) -
                          stateActionValues[currentState[0], currentState[1], action])

        if priority > dynaParams.theta:
            model.insert(priority, currentState, action)

        # start planning
        planningStep = 0

        # planning for several steps,
        # although keep planning until the priority queue becomes empty will converge much faster
        while planningStep < dynaParams.planningSteps and not model.empty():
            # get a sample with highest priority from the model
            priority, sampleState, sampleAction, sampleNewState, sampleReward = model.sample()

            # update the state action value for the sample
            delta = sampleReward + dynaParams.gamma * np.max(stateActionValues[sampleNewState[0], sampleNewState[1], :]) - \
                    stateActionValues[sampleState[0], sampleState[1], sampleAction]
            stateActionValues[sampleState[0], sampleState[1], sampleAction] += dynaParams.alpha * delta

            # deal with all the predecessors of the sample state
            for statePre, actionPre, rewardPre in model.predecessor(sampleState):
                priority = np.abs(rewardPre + dynaParams.gamma * np.max(stateActionValues[sampleState[0], sampleState[1], :]) -
                                  stateActionValues[statePre[0], statePre[1], actionPre])
                if priority > dynaParams.theta:
                    model.insert(priority, statePre, actionPre)
            planningStep += 1

        currentState = newState

        # update the # of backups
        backups += planningStep

    return backups

# Figure 8.2, DynaMaze, use 10 runs instead of 30 runs
def figure_8_2():
    # set up an instance for DynaMaze
    dyna_maze = Maze()
    dyna_params = DynaParams()

    runs = 10
    episodes = 50
    planning_steps = [0, 5, 50]
    steps = np.zeros((len(planning_steps), episodes))

    for run in tqdm(range(runs)):
        for index, planning_step in zip(range(len(planning_steps)), planning_steps):
            dyna_params.planning_steps = planning_step
            q_value = np.zeros(dyna_maze.q_size)

            # generate an instance of Dyna-Q model
            model = TrivialModel()
            for ep in range(episodes):
                # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                steps[index, ep] += dyna_q(q_value, model, dyna_maze, dyna_params)

    # averaging over runs
    steps /= runs

    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], label='%d planning steps' % (planning_steps[i]))
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()

    plt.savefig('../images/figure_8_2.png')
    plt.close()

# wrapper function for changing maze
# @maze: a maze instance
# @dynaParams: several parameters for dyna algorithms
def changing_maze(maze, dyna_params):

    # set up max steps
    max_steps = maze.max_steps

    # track the cumulative rewards
    rewards = np.zeros((dyna_params.runs, 2, max_steps))

    for run in tqdm(range(dyna_params.runs)):
        # set up models
        models = [TrivialModel(), TimeModel(maze, time_weight=dyna_params.time_weight)]

        # initialize state action values
        q_values = [np.zeros(maze.q_size), np.zeros(maze.q_size)]

        for i in range(len(dyna_params.methods)):
            # print('run:', run, dyna_params.methods[i])

            # set old obstacles for the maze
            maze.obstacles = maze.old_obstacles

            steps = 0
            last_steps = steps
            while steps < max_steps:
                # play for an episode
                steps += dyna_q(q_values[i], models[i], maze, dyna_params)

                # update cumulative rewards
                rewards[run, i, last_steps: steps] = rewards[run, i, last_steps]
                rewards[run, i, min(steps, max_steps - 1)] = rewards[run, i, last_steps] + 1
                last_steps = steps

                if steps > maze.obstacle_switch_time:
                    # change the obstacles
                    maze.obstacles = maze.new_obstacles

    # averaging over runs
    rewards = rewards.mean(axis=0)

    return rewards

# Figure 8.4, BlockingMaze
def figure_8_4():
    # set up a blocking maze instance
    blocking_maze = Maze()
    blocking_maze.START_STATE = [5, 3]
    blocking_maze.GOAL_STATES = [[0, 8]]
    blocking_maze.old_obstacles = [[3, i] for i in range(0, 8)]

    # new obstalces will block the optimal path
    blocking_maze.new_obstacles = [[3, i] for i in range(1, 9)]

    # step limit
    blocking_maze.max_steps = 3000

    # obstacles will change after 1000 steps
    # the exact step for changing will be different
    # However given that 1000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    blocking_maze.obstacle_switch_time = 1000

    # set up parameters
    dyna_params = DynaParams()
    dyna_params.alpha = 1.0
    dyna_params.planning_steps = 10
    dyna_params.runs = 20

    # kappa must be small, as the reward for getting the goal is only 1
    dyna_params.time_weight = 1e-4

    # play
    rewards = changing_maze(blocking_maze, dyna_params)

    for i in range(len(dyna_params.methods)):
        plt.plot(rewards[i, :], label=dyna_params.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.savefig('../images/figure_8_4.png')
    plt.close()


# Figure 8.6, ShortcutMaze
def figure8_6():
    # set up a shortcut maze instance
    shortcutMaze = Maze()
    shortcutMaze.START_STATE = [5, 3]
    shortcutMaze.GOAL_STATES = [[0, 8]]
    shortcutMaze.old_obstacles = [[3, i] for i in range(1, 9)]

    # new obstacles will have a shorter path
    shortcutMaze.new_obstacles = [[3, i] for i in range(1, 8)]

    # step limit
    shortcutMaze.max_steps = 6000

    # obstacles will change after 3000 steps
    # the exact step for changing will be different
    # However given that 3000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    shortcutMaze.obstacle_switch_time = 3000

    # set up parameters
    dynaParams = DynaParams()

    # 50-step planning
    dynaParams.planning_steps = 50

    # average over 5 independent runs
    dynaParams.runs = 5

    # weight for elapsed time sine last visit
    dynaParams.time_weight = 1e-3

    # also a tricky alpha ...
    dynaParams.alpha = 0.7

    # play
    rewards = changing_maze(shortcutMaze, dynaParams)

    plt.figure(2)
    for i in range(0, len(dynaParams.methods)):
        plt.plot(range(0, shortcutMaze.max_steps), rewards[i, :], label=dynaParams.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

# Helper function to display best actions, just for debug
def printActions(stateActionValues, maze):
    bestActions = []
    for i in range(0, maze.WORLD_HEIGHT):
        bestActions.append([])
        for j in range(0, maze.WORLD_WIDTH):
            if [i, j] in maze.GOAL_STATES:
                bestActions[-1].append('G')
                continue
            if [i, j] in maze.obstacles:
                bestActions[-1].append('X')
                continue
            bestAction = np.argmax(stateActionValues[i, j, :])
            if bestAction == maze.ACTION_UP:
                bestActions[-1].append('U')
            if bestAction == maze.ACTION_DOWN:
                bestActions[-1].append('D')
            if bestAction == maze.ACTION_LEFT:
                bestActions[-1].append('L')
            if bestAction == maze.ACTION_RIGHT:
                bestActions[-1].append('R')
    for row in bestActions:
        print(row)
    print('')

# Check whether state-action values are already optimal
def checkPath(stateActionValues, maze):
    # get the length of optimal path
    # 14 is the length of optimal path of the original maze
    # 1.2 means it's a relaxed optifmal path
    maxSteps = 14 * maze.resolution * 1.2
    currentState = maze.START_STATE
    steps = 0
    while currentState not in maze.GOAL_STATES:
        bestAction = np.argmax(stateActionValues[currentState[0], currentState[1], :])
        currentState, _ = maze.step(currentState, bestAction)
        steps += 1
        if steps > maxSteps:
            return False
    return True

# Figure 8.7, mazes with different resolution
def figure8_7():
    # get the original 6 * 9 maze
    originalMaze = Maze()

    # set up the parameters for each algorithm
    paramsDyna = DynaParams()
    paramsDyna.planning_steps = 5
    paramsDyna.alpha = 0.5
    paramsDyna.gamma = 0.95

    paramsPrioritized = DynaParams()
    paramsPrioritized.theta = 0.0001
    paramsPrioritized.planning_steps = 5
    paramsPrioritized.alpha = 0.5
    paramsPrioritized.gamma = 0.95

    params = [paramsPrioritized, paramsDyna]

    # set up models for planning
    models = [PriorityModel, TrivialModel]
    methodNames = ['Prioritized Sweeping', 'Dyna-Q']

    # due to limitation of my machine, I can only perform experiments for 5 mazes
    # say 1st maze has w * h states, then k-th maze has w * h * k * k states
    numOfMazes = 5

    # build all the mazes
    mazes = [originalMaze.extend_maze(i) for i in range(1, numOfMazes + 1)]
    methods = [prioritizedSweeping, dyna_q]

    # track the # of backups
    backups = np.zeros((2, numOfMazes))

    # My machine cannot afford too many runs...
    runs = 5
    for run in range(0, runs):
        for i in range(0, len(methodNames)):
            for mazeIndex, maze in zip(range(0, len(mazes)), mazes):
                print('run:', run, methodNames[i], 'maze size:', maze.WORLD_HEIGHT * maze.WORLD_WIDTH)

                # initialize the state action values
                currentStateActionValues = np.copy(maze.stateActionValues)

                # track steps / backups for each episode
                steps = []

                # generate the model
                model = models[i]()

                # play for an episode
                while True:
                    steps.append(methods[i](currentStateActionValues, model, maze, params[i]))

                    # print best actions w.r.t. current state-action values
                    # printActions(currentStateActionValues, maze)

                    # check whether the (relaxed) optimal path is found
                    if checkPath(currentStateActionValues, maze):
                        break

                # update the total steps / backups for this maze
                backups[i][mazeIndex] += np.sum(steps)

    # Dyna-Q performs several backups per step
    backups[1, :] *= paramsDyna.planning_steps

    # average over independent runs
    backups /= float(runs)

    plt.figure(3)
    for i in range(0, len(methodNames)):
        plt.plot(np.arange(1, numOfMazes + 1), backups[i, :], label=methodNames[i])
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()

if __name__ == '__main__':
    # figure_8_2()
    figure_8_4()

