# -*- coding: utf-8 -*-
#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
"""
Created on Wed Aug 21 13:05:26 2024

@author: Kevin Wang
"""
import numpy as np
import itertools
from scipy.stats import poisson


class Car_Rental:
    '''The order of events is in method cal_new_value as follows.
    Given a state, having an action from policy, we move the cars. In day,
    customers request cars. After that, customer return the cars.
    '''

    def __init__(self):
        self.min_n_car = 0
        self.max_n_car = 20
        self.moving_cost = 2
        self.rent_profit = 10
        self.actions = np.arange(-5, 5 + 1)
        self.n_car_list = np.arange(self.max_n_car + 1)
        self.value = np.zeros((self.max_n_car + 1, self.max_n_car + 1))
        self.policy = np.zeros(self.value.shape, dtype=int)
        self.gamma = 0.9
        # Calculate the model represented by transition matrix for later use
        self.eprofit, self.trans_1, self.trans_2 = self._cal_model()

    def move_car(self, n1: int, n2: int, n_move: int) -> tuple[int]:
        '''Get the numbers of cars in each location after moving the car(s)
        Note that we move as much cars as possible regardless the number of
        empty parking lot (cars more than 20 would disappear).
        '''
        if n_move > 0:  # move cars from L1 to L2
            can_move = min(n1, n_move)
        elif n_move < 0:  # from L2 to L1
            can_move = - min(n2, abs(n_move))
        else:
            return n1, n2  # do nothing when n_move is zero
        n1_after, n2_after = [min(n_car, self.max_n_car)
                              for n_car in (n1 - can_move, n2 + can_move)]
        return n1_after, n2_after

    def cal_new_value(self, n_1, n_2, n_move):
        '''Calculate the new state value given an action'''
        # 1. Moving car
        # moving cost
        ereward = -self.moving_cost * abs(n_move)  # expected reward
        n_1, n_2 = self.move_car(n_1, n_2, n_move)
        # 2. Car request
        # 3. Car return
        # update expected reward by adding expected profit
        ereward += self.eprofit[n_1, n_2]
        # probability of next state
        state_p = (self.trans_1[n_1, :].reshape(-1, 1) @
                   self.trans_2[n_2, :].reshape(1, -1))
        return ereward + self.gamma * (state_p * self.value).sum()

    def update_value(self):
        '''Update value function iteratively'''
        states = itertools.product(self.n_car_list, self.n_car_list)
        for n_1, n_2 in states:
            n_move = self.policy[n_1, n_2]
            self.value[n_1, n_2] = self.cal_new_value(n_1, n_2, n_move)

    def policy_eval(self, eps=1e-5):
        '''Policy evaluation. update value function given policy'''
        while True:
            old_value = self.value.copy()
            self.update_value()
            if abs(old_value - self.value).max() < eps:
                break

    def update_policy(self) -> bool:
        '''Find better policy for each state given the value function'''
        states = itertools.product(self.n_car_list, self.n_car_list)
        is_policy_change = False
        for n_1, n_2 in states:
            values = np.empty(self.actions.shape)
            for i, n_move in enumerate(self.actions):
                values[i] = self.cal_new_value(n_1, n_2, n_move)
            best_action = self.actions[values.argmax()]
            if self.policy[n_1, n_2] != best_action:
                is_policy_change = True
                self.policy[n_1, n_2] = best_action
        return is_policy_change

    def policy_improve(self):
        '''update policy function (of each state)'''
        self.policy_eval()
        while True:
            policy_change = self.update_policy()
            if policy_change:
                self.policy_eval()
            else:
                break

    def _cal_model(self):
        '''Calculate the environment model represented by transition
        probability and expected profit.
        The final transition probability is resulted from two transition
        matrices by matrix multiplication. We can see the detail of each
        transition matrix in methods.
        '''
        eprofit_1, req_trans_1 = self._request_transition(mu=3)
        eprofit_2, req_trans_2 = self._request_transition(mu=4)
        eprofit = eprofit_1.reshape(-1, 1) + eprofit_2.reshape(1, -1)
        ret_trans_1 = self._return_transition(mu=3)
        ret_trans_2 = self._return_transition(mu=2)
        trans_1 = req_trans_1 @ ret_trans_1
        trans_2 = req_trans_2 @ ret_trans_2
        return eprofit, trans_1, trans_2

    def _request_transition(self, mu):
        '''The transition matrix, the probability from state s to s'.
        row: current state (n cars in location)
        col: next state (after the event of car requestion)
        Note: The best way to understand this function is by printing out.
        '''
        # probability table
        rv = poisson(mu)
        poisson_pmf = rv.pmf(self.n_car_list)
        # probability of actual request for calculating expected profit
        req_p = np.empty(self.value.shape)
        # transition matrix (probability from s to s')
        trans = np.empty(self.value.shape)
        for s in self.n_car_list:  # for each state
            p = poisson_pmf[: s + 1].copy()
            p[-1] = 1 - p[: -1].sum()
            # pad to the right
            req_p[s, :] = np.pad(p, (0, self.max_n_car - s), constant_values=0)
            # transition matrix (is the reverse order of request number)
            _ = np.pad(np.flip(p), (0, self.max_n_car - s), constant_values=0)
            trans[s, :] = _
        # expected profit given state
        profit = np.arange(req_p.shape[1]) * self.rent_profit
        eprofit = (req_p * profit).sum(axis=1)
        return eprofit, trans

    def _return_transition(self, mu):
        '''The transition matrix for car return'''
        rv = poisson(mu)
        poisson_pmf = rv.pmf(self.n_car_list)
        trans = np.empty(self.value.shape)  # transition matrix
        for s in self.n_car_list:
            p = poisson_pmf[: (self.max_n_car + 1) - s].copy()
            p[-1] = 1 - p[: -1].sum()
            p = np.pad(p, (s, 0), constant_values=0)  # pad to the left
            trans[s, :] = p
        return trans


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    def heatmap(ax, data):
        im = ax.imshow(data, origin='lower')
        ax.figure.colorbar(im, ax=ax)

    env = Car_Rental()
    # env.policy_eval()
    # env.value.max()
    # env.value.min()

    env.policy_improve()

    fig, ax = plt.subplots()
    heatmap(ax, env.policy)
    fig, ax = plt.subplots()
    heatmap(ax, env.value)
