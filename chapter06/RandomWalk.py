#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# 0 is the left terminal state
# 6 is the right terminal state
# 1 ... 5 represents A ... E
states = np.zeros(7)
states[1:6] = 0.5
# For convenience, we assume all rewards are 0
# and the left terminal state has value 0, the right terminal state has value 1
# This trick has been used in Gambler's Problem
states[6] = 1

# set up true state values
trueValue = np.zeros(7)
trueValue[1:6] = np.arange(1, 6) / 6.0
trueValue[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1

# @states: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @states
def temporalDifference(states, alpha=0.1, batch=False):
    state = 3
    trajectory = [state]
    rewards = [0]
    while True:
        oldState = state
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        # Assume all rewards are 0
        reward = 0
        trajectory.append(state)
        # TD update
        if not batch:
            states[oldState] += alpha * (reward + states[state] - states[oldState])
        if state == 6 or state == 0:
            break
        rewards.append(reward)
    return trajectory, rewards

# @states: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @states
def monteCarlo(states, alpha=0.1, batch=False):
    state = 3
    trajectory = [3]
    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    returns = 0
    while True:
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break
    if not batch:
        for state_ in trajectory[:-1]:
            # MC update
            states[state_] += alpha * (returns - states[state_])
    return trajectory, [returns] * (len(trajectory) - 1)

# Figure 6.2 left
def stateValue():
    episodes = [0, 1, 10, 100]
    currentStates = np.copy(states)
    plt.figure(1)
    axisX = np.arange(0, 7)
    for i in range(0, episodes[-1] + 1):
        if i in episodes:
            plt.plot(axisX, currentStates, label=str(i) + ' episodes')
        temporalDifference(currentStates)
    plt.plot(axisX, trueValue, label='true values')
    plt.xlabel('state')
    plt.legend()

# Figure 6.2 right
def RMSError():
    # I'm lazy here, so do not let same alpha value appear in both arrays
    # For example, if in TD you want to use alpha = 0.2, then in MC you can use alpha = 0.201
    TDAlpha = [0.15, 0.1, 0.05]
    MCAlpha = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    plt.figure(2)
    axisX = np.arange(0, episodes)
    for alpha in TDAlpha + MCAlpha:
        totalErrors = np.zeros(episodes)
        if alpha in TDAlpha:
            method = 'TD'
        else:
            method = 'MC'
        for run in range(0, runs):
            errors = []
            currentStates = np.copy(states)
            for i in range(0, episodes):
                errors.append(np.sqrt(np.sum(np.power(trueValue - currentStates, 2)) / 5.0))
                if method == 'TD':
                    temporalDifference(currentStates, alpha=alpha)
                else:
                    monteCarlo(currentStates, alpha=alpha)
            totalErrors += np.asarray(errors)
        totalErrors /= runs
        plt.plot(axisX, totalErrors, label=method + ', alpha=' + str(alpha))
    plt.xlabel('episodes')
    plt.legend()

# Figure 6.3
# @method: 'TD' or 'MC'
def batchUpdating(method, episodes, alpha=0.001):
    # perform 100 independent runs
    runs = 100
    totalErrors = np.zeros(episodes - 1)
    for run in range(0, runs):
        currentStates = np.copy(states)
        errors = []
        # track shown trajectories and reward/return sequences
        trajectories = []
        rewards = []
        for ep in range(1, episodes):
            print('Run:', run, 'Episode:', ep)
            if method == 'TD':
                trajectory_, rewards_ = temporalDifference(currentStates, batch=True)
            else:
                trajectory_, rewards_ = monteCarlo(currentStates, batch=True)
            trajectories.append(trajectory_)
            rewards.append(rewards_)
            while True:
                # keep feeding our algorithm with trajectories seen so far until state value function converges
                updates = np.zeros(7)
                for trajectory_, rewards_ in zip(trajectories, rewards):
                    for i in range(0, len(trajectory_) - 1):
                        if method == 'TD':
                            updates[trajectory_[i]] += rewards_[i] + currentStates[trajectory_[i + 1]] - currentStates[trajectory_[i]]
                        else:
                            updates[trajectory_[i]] += rewards_[i] - currentStates[trajectory_[i]]
                updates *= alpha
                if np.sum(np.abs(updates)) < 1e-3:
                    break
                # perform batch updating
                currentStates += updates
            # calculate rms error
            errors.append(np.sqrt(np.sum(np.power(currentStates - trueValue, 2)) / 5.0))
        totalErrors += np.asarray(errors)
    totalErrors /= episodes - 1
    return totalErrors

def figure6_2():
    stateValue()
    RMSError()

def figure6_3():
    episodes = 100 + 1
    TDErrors = batchUpdating('TD', episodes)
    MCErrors = batchUpdating('MC', episodes)
    axisX = np.arange(1, episodes)
    plt.figure(3)
    plt.plot(axisX, TDErrors, label='TD')
    plt.plot(axisX, MCErrors, label='MC')
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.legend()

figure6_2()

# Figure 6.3 may take a while to calculate
# figure6_3()

plt.show()