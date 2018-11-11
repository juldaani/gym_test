#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:58:31 2018

@author: juho
"""


import gym
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier



clf.verbose = 0
clf.n_jobs = 1

rewards = []
acts = []
for i_episode in range(100):
    observation = env.reset()
    
#    print(i_episode)
    
    r = 0
    for t in range(10000):
        env.render()
        
        scaledObserv = scaler.transform([observation])
        action = int(clf.predict(scaledObserv))
        observation, reward, done, info = env.step(action)
        
        acts.append(action)
        
        r += reward
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        
    rewards.append(r)

rewards = np.array(rewards)


np.mean(rewards)
np.max(rewards)
np.min(rewards)
