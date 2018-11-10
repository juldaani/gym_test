#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:52:59 2018

@author: juho
"""

    
import gym
import numpy as np


env = gym.make('CartPole-v0')




# %%
    
    

clf.verbose = 0

env.seed(0)
np.random.seed(0)



initActionChain = []
#states = []
initReward = 0
observation = env.reset()
for t in range(10000):
#        env.render()
    action = int(clf.predict([observation]))
    
    initActionChain.append(action)
#    states.append(observation)
    
    observation, r, done, info = env.step(action)

    initReward += r
    
    if done:
#            print("Episode finished after {} timesteps".format(t+1))
        break

# %%

depth = 10
nBranches = 500

newActionChains = []

for actionIdx in range(1,len(initActionChain)):
    print(actionIdx)
    
    for n in range(nBranches):
        branch = np.random.randint(0, high=2, size=depth)
        newActionChains.append(initActionChain[:actionIdx] + list(branch))


# %%



rewardsForNewActions = []
execNewActions = []
statesNewActions = []
for i,actionChain in enumerate(newActionChains):
    if(i%500 == 0): print(str(i) + ' / ' + str(len(newActionChains)))
    
    env.seed(0)
    np.random.seed(0)
    
    reward = 0
    state = env.reset()
    executedActions = []
    states = []
    for action in actionChain:
    #    env.render()

        executedActions.append(action)
        states.append(state)

        state, r, done, info = env.step(action)
        reward += r
        
        if done:
    #            print("Episode finished after {} timesteps".format(t+1))
            break
    
    rewardsForNewActions.append(reward)
    execNewActions.append(np.array(executedActions))
    statesNewActions.append(states)
    

# %%

statesForTraining = []
actionsForTraining = []
tmp = []
for i in range(len(rewardsForNewActions)):
    r = rewardsForNewActions[i]
    
    if(r > initReward):
        statesForTraining.append(np.row_stack(statesNewActions[i]))
        actionsForTraining.append(execNewActions[i])
        
        tmp.append(r)
    
actionsForTraining = np.concatenate(actionsForTraining)
statesForTraining = np.row_stack(statesForTraining)
    
len(actionsForTraining)
len(statesForTraining)

# %%



clf = ExtraTreesClassifier(n_estimators=500, random_state=0, verbose=2)
clf.fit(statesForTraining, actionsForTraining)

#pred = clf.predict(statesForTraining)
#np.sum(pred == actionsForTraining) / len(actionsForTraining)


# %%

clf.verbose = 0

rewards = []
for i_episode in range(100):
    observation = env.reset()
    
    print(i_episode)
    
    r = 0
    for t in range(10000):
        env.render()
        
        action = int(clf.predict([observation]))
        observation, reward, done, info = env.step(action)
        
        r += reward
        
        if done:
#            print("Episode finished after {} timesteps".format(t+1))
            break
        
    rewards.append(r)

rewards = np.array(rewards)



np.mean(rewards)
np.max(rewards)
np.min(rewards)












