# -*- coding: utf-8 -*-
"""
# @Time    : 2018/1/29 上午9:22
# @Author  : zhanzecheng
# @File    : BL_brain.py
# @Software: PyCharm
"""
import math
import random
import pickle
import os
import sys

class QLearningTable:

    def __init__(self, learning=False, epsilon=1.0, alpha=0.5, penalty=-0.05):

        self.valid_actions = [i / 10 for i in range(4, 12)]
        self.learning = learning
        if os.path.exists('./state.txt'):
            with open('./state.txt', 'r') as f:
                line = f.readlines()
                self.Q = eval(line[0].strip())
        else:
            self.Q = dict()
        self.epsilon = epsilon
        self.alpha = alpha
        self.penalty = penalty
        self.t = 1
        # valid actions
        print('Valid actions:', self.valid_actions)

    def save_Q(self):
        with open('./state.txt', 'w') as f:
            f.write(str(self.Q))

    def reset(self, testing=False):
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
            self.epsilon = math.exp(-0.005 * self.t)
            self.t += 1
        return

    def add_t(self):
        self.t += 1

    # 这里使用了距离作为状态值
    def build_state(self, piece_x, piece_y, board_x, board_y):
        state = math.sqrt(math.fabs(piece_x - board_x) ** 2 + math.fabs(piece_y - board_y) ** 2)
        state = int(state)
        return state

    def learn(self, state, action, reward):
        if self.learning:
            self.Q[state][action] = reward * self.alpha + self.Q[state][action] * (1 - self.alpha)
        print('self.Q:', self.Q)


    def update(self, action, state, bias):
        # state = self.build_state()          # Get current state
        # Create 'state' in Q-table
        if state != 0:
            self.createQ(state)
            # Receive a reward
        reward = float("{0:.2f}".format(bias * self.penalty + 0.2))
        print("bias: ", bias)
        print("state: ", state)
        print('action: ', action)
        print('reward: ', reward)
        # Q-learn
        if state != 0:
            self.learn(state, action, reward)


    def choose_action(self, state):
        self.state = int(state)
        if not self.learning or random.random() <= self.epsilon:
            action = random.choice(self.valid_actions)
        else:
            maxQ = self.get_maxQ(state)
            maxQ_actions = [act for act, val in self.Q[state].items() if val == maxQ]
            print("maxQ_actions:", maxQ_actions)
            action = random.choice(maxQ_actions)
        return action

    def get_maxQ(self, state):
        if state in self.Q:
            maxQ = max(self.Q[state].values())
        else:
            self.createQ(state)
            maxQ = max(self.Q[state].values())
        return maxQ


    def createQ(self, state):
        if self.learning:
            if not state in self.Q:
                self.Q.setdefault(state, {action: 0.0 for action in self.valid_actions})

