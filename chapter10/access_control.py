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
from mpl_toolkits.mplot3d.axes3d import Axes3D
from math import floor
import seaborn as sns

#######################################################################
# Following are some utilities for tile coding from Rich.
# To make each file self-contained, I copied them from
# http://incompleteideas.net/tiles/tiles3.py-remove
# with some naming convention changes
#
# Tile coding starts
class IHT:
    "Structure to handle collisions"
    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count

def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT): return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int): return hash(tuple(coordinates)) % m
    if m is None: return coordinates

def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles
# Tile coding ends
#######################################################################

# possible priorities
PRIORITIES = np.arange(0, 4)
# reward for each priority
REWARDS = np.power(2, np.arange(0, 4))

# possible actions
REJECT = 0
ACCEPT = 1
ACTIONS = [REJECT, ACCEPT]

# total number of servers
NUM_OF_SERVERS = 10

# at each time step, a busy server will be free w.p. 0.06
PROBABILITY_FREE = 0.06

# step size for learning state-action value
ALPHA = 0.01

# step size for learning average reward
BETA = 0.01

# probability for exploration
EPSILON = 0.1

# a wrapper class for differential semi-gradient Sarsa state-action function
class ValueFunction:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @alpha: step size for learning state-action value
    # @beta: step size for learning average reward
    def __init__(self, num_of_tilings, alpha=ALPHA, beta=BETA):
        self.num_of_tilings = num_of_tilings
        self.max_size = 2048
        self.hash_table = IHT(self.max_size)
        self.weights = np.zeros(self.max_size)

        # state features needs scaling to satisfy the tile software
        self.server_scale = self.num_of_tilings / float(NUM_OF_SERVERS)
        self.priority_scale = self.num_of_tilings / float(len(PRIORITIES) - 1)

        self.average_reward = 0.0

        # divide step size equally to each tiling
        self.alpha = alpha / self.num_of_tilings

        self.beta = beta

    # get indices of active tiles for given state and action
    def get_active_tiles(self, free_servers, priority, action):
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                            [self.server_scale * free_servers, self.priority_scale * priority],
                            [action])
        return active_tiles

    # estimate the value of given state and action without subtracting average
    def value(self, free_servers, priority, action):
        active_tiles = self.get_active_tiles(free_servers, priority, action)
        return np.sum(self.weights[active_tiles])

    # estimate the value of given state without subtracting average
    def state_value(self, free_servers, priority):
        values = [self.value(free_servers, priority, action) for action in ACTIONS]
        # if no free server, can't accept
        if free_servers == 0:
            return values[REJECT]
        return np.max(values)

    # learn with given sequence
    def learn(self, free_servers, priority, action, new_free_servers, new_priority, new_action, reward):
        active_tiles = self.get_active_tiles(free_servers, priority, action)
        estimation = np.sum(self.weights[active_tiles])
        delta = reward - self.average_reward + self.value(new_free_servers, new_priority, new_action) - estimation
        # update average reward
        self.average_reward += self.beta * delta
        delta *= self.alpha
        for active_tile in active_tiles:
            self.weights[active_tile] += delta

# get action based on epsilon greedy policy and @valueFunction
def get_action(free_servers, priority, value_function):
    # if no free server, can't accept
    if free_servers == 0:
        return REJECT
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = [value_function.value(free_servers, priority, action) for action in ACTIONS]
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

# take an action
def take_action(free_servers, priority, action):
    if free_servers > 0 and action == ACCEPT:
        free_servers -= 1
    reward = REWARDS[priority] * action
    # some busy servers may become free
    busy_servers = NUM_OF_SERVERS - free_servers
    free_servers += np.random.binomial(busy_servers, PROBABILITY_FREE)
    return free_servers, np.random.choice(PRIORITIES), reward

# differential semi-gradient Sarsa
# @valueFunction: state value function to learn
# @maxSteps: step limit in the continuing task
def differential_semi_gradient_sarsa(value_function, max_steps):
    current_free_servers = NUM_OF_SERVERS
    current_priority = np.random.choice(PRIORITIES)
    current_action = get_action(current_free_servers, current_priority, value_function)
    # track the hit for each number of free servers
    freq = np.zeros(NUM_OF_SERVERS + 1)

    for _ in tqdm(range(max_steps)):
        freq[current_free_servers] += 1
        new_free_servers, new_priority, reward = take_action(current_free_servers, current_priority, current_action)
        new_action = get_action(new_free_servers, new_priority, value_function)
        value_function.learn(current_free_servers, current_priority, current_action,
                             new_free_servers, new_priority, new_action, reward)
        current_free_servers = new_free_servers
        current_priority = new_priority
        current_action = new_action
    print('Frequency of number of free servers:')
    print(freq / max_steps)

# Figure 10.5, Differential semi-gradient Sarsa on the access-control queuing task
def figure_10_5():
    max_steps = int(1e6)
    # use tile coding with 8 tilings
    num_of_tilings = 8
    value_function = ValueFunction(num_of_tilings)
    differential_semi_gradient_sarsa(value_function, max_steps)
    values = np.zeros((len(PRIORITIES), NUM_OF_SERVERS + 1))
    for priority in PRIORITIES:
        for free_servers in range(NUM_OF_SERVERS + 1):
            values[priority, free_servers] = value_function.state_value(free_servers, priority)

    fig = plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for priority in PRIORITIES:
        plt.plot(range(NUM_OF_SERVERS + 1), values[priority, :], label='priority %d' % (REWARDS[priority]))
    plt.xlabel('Number of free servers')
    plt.ylabel('Differential value of best action')
    plt.legend()

    ax = fig.add_subplot(2, 1, 2)
    policy = np.zeros((len(PRIORITIES), NUM_OF_SERVERS + 1))
    for priority in PRIORITIES:
        for free_servers in range(NUM_OF_SERVERS + 1):
            values = [value_function.value(free_servers, priority, action) for action in ACTIONS]
            if free_servers == 0:
                policy[priority, free_servers] = REJECT
            else:
                policy[priority, free_servers] = np.argmax(values)

    fig = sns.heatmap(policy, cmap="YlGnBu", ax=ax, xticklabels=range(NUM_OF_SERVERS + 1), yticklabels=PRIORITIES)
    fig.set_title('Policy (0 Reject, 1 Accept)')
    fig.set_xlabel('Number of free servers')
    fig.set_ylabel('Priority')

    plt.savefig('../images/figure_10_5.png')
    plt.close()

if __name__ == '__main__':
    figure_10_5()
