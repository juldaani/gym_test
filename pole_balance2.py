#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:49:11 2018

@author: juho
"""


    
import gym
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


env = gym.make('CartPole-v0')



clf.verbose = 0
np.random.seed(0)


observsTraining = []
actionsTraining = []
for k in range(100):
    print(k)
    
    seed = np.random.randint(0,high=123454)
    env.seed(seed)
    observation = env.reset()
    actions = []
    initReward = 0
    
    # Run single test to get actions
    for i in range(10000):
        
        a = int(clf.predict([observation]))
#        a = env.action_space.sample()
        
        actions.append(a)
        
        observation, r, done, info = env.step(a)
        initReward += r
        
        if done:
            break
        
        
    # Permutate the actions
    depth = 10
    nBranches = 100
    permActions = []
#    for actionIdx in reversed(range(1,len(actions))):
    for actionIdx in range(len(actions)-1,max(len(actions)-depth,-1),-1):
        
        for n in range(nBranches):
            branch = np.random.randint(0, high=2, size=depth)
            permActions.append(actions[:actionIdx] + list(branch))


    # Test the permuted actions
    permActRewards = []
    permActObservs = []
    execPermActions = []
    for i,acts in enumerate(permActions):
#        if(i%100 == 0): print(str(i) + ' / ' + str(len(permActions)))
        
        env.seed(seed)
        observation = env.reset()
        curReward = 0
        curObservs = []
        tmpActions = []
        for a in acts:
            
#            print(acts)
            
    #        env.render()
            tmpActions.append(a)
            curObservs.append(observation)
            
            observation, r, done, info = env.step(a)
            curReward += r
            
            if done:
                break
        
        execPermActions.append(tmpActions)
        permActObservs.append(curObservs)
        permActRewards.append(curReward)
        
    
    # Filter out bad actions
    filtObservs = []
    filtActions = []
    for i in range(len(permActRewards)):
        r = permActRewards[i]
        
        if(r > initReward):
            filtObservs.append(np.row_stack(permActObservs[i]))
            filtActions.append(execPermActions[i])
        
        
    filtActions = np.concatenate(filtActions)
    filtObservs = np.row_stack(filtObservs)
    
    observsTraining.append(filtObservs)
    actionsTraining.append(filtActions)


# %%

x = np.row_stack(observsTraining)
y = np.concatenate(actionsTraining) 

len(x)

clf = ExtraTreesClassifier(n_estimators=100, random_state=0, verbose=2, n_jobs=-1, 
                           min_samples_split=5, min_samples_leaf=5)
clf.fit(x,y)

pred = clf.predict(x)
np.sum(pred == y) / len(y)


# 0.7052555058374266
# 0.936899833817079



#0.8365937610636875
#0.870550850908854
#0.8840891597856441










    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
