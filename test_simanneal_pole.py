#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:58:10 2018

@author: juho
"""

from simanneal import Annealer
import gym
import numpy as np


class PoleBalanceAnnealer(Annealer):

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, seed):
        self.seed = seed
        super(PoleBalanceAnnealer, self).__init__(state)  # important!

    def move(self):
        rndIdx1 = np.random.randint(len(self.state))
        self.state[rndIdx1] = self.state[rndIdx1]^1

    def energy(self):
        env.seed(self.seed)
        observation = env.reset()
        
        reward = 0
        for a in self.state:
            observation, r, done, info = env.step(a)
            reward += r
            
            if done:
                break
        
        return len(self.state) - reward
        



np.random.seed(0)
env = gym.make('CartPole-v0')


seed = 1
env.seed(seed)
observation = env.reset()
initActions = list(np.random.randint(0, high=2, size=50))
initReward = 0
for a in initActions:
    
    observation, r, done, info = env.step(a)
    initReward += r
    
    if done:
        break
        


poleAnneal = PoleBalanceAnnealer(initActions, seed)
poleAnneal.steps = 50
poleAnneal.Tmax = 60
poleAnneal.Tmin = 1
poleAnneal.copy_strategy = "slice"  # since our state is just a list, slice is the fastest way to copy
state, e = poleAnneal.anneal()

print(e)




















