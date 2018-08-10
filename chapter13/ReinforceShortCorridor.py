#######################################################################
# Copyright (C)                                                       #
# 2018 Sergii Bondariev (sergeybondarev@gmail.com)                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#
# This is a reproduction of the plot shown in Figure 13.1
# in Chapter 13, "Policy Gradient Methods". Book draft May 27, 2018.
#

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import multiprocessing


class Env:
    """
    Short corridor environment, see Example 13.1
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.s = 0

    def step(self, go_right):
        """
        Args:
            go_right (bool): chosen action
        Returns:
            tuple of (reward, episode terminated?)
        """
        if self.s == 0 or self.s == 2:
            if go_right:
                self.s += 1
            else:
                self.s = max(0, self.s - 1)
        else:
            # self.s == 1
            if go_right:
                self.s -= 1
            else:
                self.s += 1

        if self.s == 3:
            # terminal state
            return 0, True
        else:
            return -1, False


def softmax(x):
    t = np.exp(x - np.max(x))
    return t / np.sum(t)


class Agent:
    """
    Agent that follows algorithm
    'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)'
    """
    def __init__(self, alpha, gamma):
        # set values such that initial conditions correspond to left-epsilon greedy
        self.theta = np.array([-1.47, 1.47])
        self.alpha = alpha
        self.gamma = gamma
        # first column - left, second - right
        self.x = np.array([[0, 1], 
                           [1, 0]])
        self.rewards = []
        self.actions = []

    def get_pi(self):
        h = np.dot(self.theta, self.x)
        t = np.exp(h - np.max(h))
        pmf = t / np.sum(t)
        # never become deterministic, 
        # guarantees episode finish
        imin = np.argmin(pmf)
        epsilon = 0.05

        if pmf[imin] < epsilon:
            pmf[:] = 1 - epsilon
            pmf[imin] = epsilon

        return pmf 

    def get_p_right(self):
        return self.get_pi()[1]

    def choose_action(self, reward):
        if reward is not None:
            self.rewards.append(reward)

        pmf = self.get_pi()
        go_right = np.random.uniform() <= pmf[1]
        self.actions.append(go_right)

        return go_right

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            if self.actions[i]:
                j = 1
            else:
                j = 0

            pmf = self.get_pi()
            grad_lnpi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * G[i] * grad_lnpi
            self.theta += update 
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []


def trial(num_episodes, alpha, gamma):
    env = Env()
    agent = Agent(alpha=alpha, gamma=gamma)

    g1 = np.zeros(num_episodes)
    p_right = np.zeros(num_episodes)

    for episode_idx in range(num_episodes):
        # print("Episode {}".format(episode_idx))
        rewards_sum = 0
        reward = None
        env.reset()

        while True:
            go_right = agent.choose_action(reward)
            reward, episode_end = env.step(go_right)
            rewards_sum += reward

            if episode_end:
                agent.episode_end(reward)
                #print('rewards_sum: {}'.format(rewards_sum))
                # decay alpha with time
                #agent.alpha *= 0.995
                break

        g1[episode_idx] = rewards_sum
        p_right[episode_idx] = agent.get_p_right()


    return (g1, p_right)


def run():
    num_trials = 1000
    num_episodes = 1000
    alpha = 2e-4
    gamma = 1

    g1 = np.zeros((num_trials, num_episodes))
    p_right = np.zeros((num_trials, num_episodes))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    res = [pool.apply_async(trial, (num_episodes, alpha, gamma)) for trial_idx in range(num_trials)]

    for trial_idx, r in enumerate(res):
        print("Trial {}".format(trial_idx))
        out = r.get()
        g1[trial_idx, :] = out[0]
        p_right[trial_idx, :] = out[1]

    avg_rewards_sum = np.mean(g1, axis=0)
    avg_p_right = np.mean(p_right, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(np.arange(num_episodes) + 1, avg_rewards_sum, color="blue")
    ax1.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls="dashed", color="red", label="-11.6")
    ax1.set_ylabel("Value of the first state")
    ax1.set_xlabel("Episode number")
    ax1.set_title("REINFORNCE Monte-Carlo Policy-Gradient Control (episodic) \n"
                 "on a short corridor with switched actions.")
    ax1.legend(loc="lower right")
    # ax1.set_yticks(np.sort(np.append(ax2.get_yticks(), -11.6)))

    ax2.plot(np.arange(num_episodes) + 1, avg_p_right, color="blue")
    ax2.plot(np.arange(num_episodes) + 1, 0.58 * np.ones(num_episodes), ls="dashed", color="red", label="0.58")
    ax2.set_ylabel("Agent's probability of going right")
    ax2.set_xlabel("Episode number\n\n alpha={}. Averaged over {} trials".format(alpha, num_trials))
    ax2.legend(loc="lower right")
    # ax2.set_yticks(np.append(ax2.get_yticks(), 0.58))

    fig.tight_layout()
    plt.show()
    # plt.savefig("out.png")

if __name__ == "__main__":
    run()
