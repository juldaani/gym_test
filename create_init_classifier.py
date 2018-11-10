#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:52:59 2018

@author: juho
"""

    
import gym
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


env = gym.make('CartPole-v0')
env.seed(0)
np.random.seed(0)

actions = []
states = []
for i_episode in range(4000):
    observation = env.reset()
    
    if(i_episode % 1000 == 0): print(i_episode)
    
    for t in range(100):
    #        env.render()
#        print(observation)
        action = env.action_space.sample()
#        action = actions[t]
    #    action = t % 2
    #        print(action)
        observation, reward, done, info = env.step(action)
        
        actions.append(action)
        states.append(observation)
        
    #        print(reward)
        
        if done:
#            print("Episode finished after {} timesteps".format(t+1))
            break
        
        

actions = np.row_stack(actions).flatten()
states = np.row_stack(states)


clf = ExtraTreesClassifier(n_estimators=200, random_state=0, verbose=2)
clf.fit(states, actions)


















