from random import random
from random import choice
import numpy as np
import os
import math

import gym
from gym import spaces

def load_data( filename ):
    f = open( filename, 'r' )
    data = []
    for l in f:
        l = l.strip()
        if l == "":
            continue
        d, o, h, c, l, v = l.strip().split(',')
        h = float(h)
        l = float(l)
        c = float(c)
        v = int(v)

        # no normalization

        data.append( ( d, o, h, c, l, v ) )
    return data

class MarketEnv(gym.Env):
    def __init__(self, dataset, init_budget=10000, scope=60, max_cnt=90 ):
        self.scope = scope
        self.init_budget = init_budget
        self.info = { 'date': None, 'status': 'INIT', 'balance': [0,0,0], 'position':[], 'ratio': 100 }
        self.max_cnt = max_cnt

        self.targetCodes = []
        self.train_data = {}

        for root, dir, files in os.walk(dataset):
            if root == dataset:
                for f in files:
                    self.train_data[f] = load_data('/'.join([dataset,f]))
                    
        self.actions = [
            "BUY",
            "SELL",
            "HOLD"
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box( np.ones( scope ), np.ones( scope ) )

        self.reset()
        self._seed()

    def _step(self, action):
        if self.done:
            return self.state, self.reward, self.done, self.info 

        self.reward = 0
        self.index += 1
        self.cnt += 1
        self.info['date'] = self.train_data[self.pick][self.index][0]

        price = self.train_data[self.pick][self.index][3]
        if self.actions[action] == "BUY":
            self.balance -= price 
            if self.boughts != [] and self.boughts[0] < 0:
                self.boughts.pop(0)
            else:
                self.boughts.append( price )
        elif self.actions[action] == "SELL":
            self.balance += price 
            if self.boughts != [] and self.boughts[0] > 0:
                self.boughts.pop(0)
            else:
                self.boughts.append( -price )

        self.reward = sum( [ 1 if x > 0 else -1 for x in self.boughts] ) * price - sum( self.boughts ) 

        if self.balance < 0:
            self.reward = - self.init_budget
            self.info['status'] = 'BROKEN'
            self.done = True
        elif self.cnt == self.max_cnt:
            self.info['status'] = 'FINISH'
            self.done = True
        else:
            self.info['status'] = 'STEP'

        self.update()

        return self.state, self.reward, self.done, self.info 

    def _reset(self):
        start = -1
        while start < self.scope:
            self.pick = choice(self.train_data.keys())
            start = int( random() * ( len(self.train_data[self.pick]) - self.scope - 90) )
        start_hist = start - self.scope
        print self.pick, start_hist, start

        self.index = start
        self.boughts = []
        self.balance = self.init_budget
        self.done = False
        self.reward = 0
        self.cnt = 0
        self.info['status'] = 'RESET'

        self.update()

        return self.state

    def _render(self, mode='human', close=False):
        if close:
            return
        return self.state

    '''
    def _close(self):
        pass

    def _configure(self):
        pass
    '''

    def _seed(self):
        return int(random() * 100)

    def update(self):
        k_c = [ [x[3]] for x in self.train_data[self.pick][self.index-59:self.index+1] ]
        k_v = [ [x[5]] for x in self.train_data[self.pick][self.index-59:self.index+1] ]

        balance = self.balance
        position = sum( [ 1 if x > 0 else -1 for x in self.boughts] ) * self.train_data[self.pick][self.index][3] 
        size = len(self.boughts)
        self.state = [ np.array([[ balance, size, position ]]), np.array([[ k_c, k_v ]]) ]
        self.info['balance'] = ( balance, position, balance+position )
        self.info['close'] = self.train_data[self.pick][self.index][3]
        self.info['position'] = self.boughts
        self.info['ratio'] = int( (balance + position) * 100 / self.init_budget )

