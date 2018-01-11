from random import random
from random import choice
import numpy as np
import os
import math

import gym
from gym import spaces
from skilog import log

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
        log.info("[GYM] {}".format(__file__))
        log.info("[GYM][PARAMETER] budget={}, scope={}, trade={}".format( init_budget, scope, max_cnt ) )
        self.scope = scope
        self.init_budget = init_budget
        self.info = { 'date': None, 'status': 'INIT', 'balance': [0,0,0], 'position':[], 'ratio': 100, 'action': None }
        self.max_cnt = max_cnt

        self.targetCodes = []
        self.train_data = {}

        for root, dir, files in os.walk(dataset):
            if root == dataset:
                for f in files:
                    self.train_data[f] = load_data('/'.join([dataset,f]))
                    
        self.actions = [
            "LONG",
            "SHORT",
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
        self.info['action'] = 'HOLD'
        if self.boughts == []:
            if self.actions[action] in ( "LONG", "SHORT" ):
                self.balance -= 5000 
            if self.actions[action] == 'LONG':
                self.info['action'] = 'OPEN_LONG'
                self.boughts.append( price )
            elif self.actions[action] == 'SHORT':
                self.info['action'] = 'OPEN_SHORT'
                self.boughts.append( -price )
        elif self.boughts[0] > 0 and self.actions[action] == 'SHORT':
            self.balance += 5000 + ( self.boughts[0] - price ) * 20
            self.boughts.pop(0)
            self.info['action'] = 'CLOSE_LONG'
        elif self.boughts[0] < 0 and self.actions[action] == 'LONG':
            self.balance += 5000 + ( self.boughts[0] + price ) * 20
            self.boughts.pop(0)
            self.info['action'] = 'CLOSE_SHORT'

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
        start_hist = -1
        while start_hist < 1:
            self.pick = choice(self.train_data.keys())
            start_hist = int( random() * ( len(self.train_data[self.pick]) - self.scope - self.max_cnt) )
        start = self.scope + start_hist 
        log.debug( " DATA:{} HIST:{} START:{}".format( self.pick, start_hist, start ) )

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
        norm_c = self.train_data[self.pick][self.index-58][3]
        norm_v = self.train_data[self.pick][self.index-58][5]
        k_c = [ [x[3]/norm_c] for x in self.train_data[self.pick][self.index-59:self.index+1] ]
        k_v = [ [x[5]/norm_v] for x in self.train_data[self.pick][self.index-59:self.index+1] ]

        price = self.train_data[self.pick][self.index][3]
        balance = self.balance
        position = sum( [ price - x if x > 0 else price + x for x in self.boughts ] )
        size = sum( [ 1 if x > 0 else -1 for x in self.boughts ] )
        self.state = [ np.array([[ balance, size, position ]]), np.array([[ k_c, k_v ]]) ]
        self.info['balance'] = ( balance, position, balance+position+len(self.boughts)*5000 )
        self.info['close'] = self.train_data[self.pick][self.index][3]
        self.info['position'] = self.boughts
        self.info['ratio'] = int( (balance + position + len(self.boughts)*5000) * 100 / self.init_budget )

