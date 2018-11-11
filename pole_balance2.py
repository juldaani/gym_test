#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:49:11 2018

@author: juho
"""


    
import gym
import numpy as np
from simanneal import Annealer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class PoleBalanceAnnealer(Annealer):

    def __init__(self, state, seed):
        self.seed = seed
        super(PoleBalanceAnnealer, self).__init__(state)  # important!

    def move(self):
        rndIdx1 = np.random.randint(len(self.state))
        self.state[rndIdx1] = self.state[rndIdx1]^1

    def energy(self):
        reward, _, _ = RunActions(self.state, self.seed)

        """
        env.seed(self.seed)
        observation = env.reset()
        
        reward = 0
#        observs = []
        for a in self.state:
#            observs.append(observation)
            
            observation, r, done, info = env.step(a)
            reward += r
            
            if done:
                break
        
#        self.reward = reward
#        self.observations = observs
        """
        
        return len(self.state) - reward
        

def RunActions(actions, seed):
    env.seed(seed)
    observation = env.reset()
    
    reward = 0
    observs = []
    execActions = []
    for a in actions:
        observs.append(observation)
        
        observation, r, done, info = env.step(a)
        reward += r
        execActions.append(a)
        
        if done:
            break
        
    return reward, observs, execActions



env = gym.make('CartPole-v0')
clf.verbose = 0
clf.n_jobs = 1
#np.random.seed(0)

observsTraining = []
actionsTraining = []
rewardsTraining = []
initRewards = []
nIters = 1
for k in range(nIters):
    if(k%1 == 0): print(str(k) + ' / ' + str(nIters))
    
    seed = np.random.randint(0,high=123454)
    env.seed(seed)
    observation = env.reset()
    actions = []
    initReward = 0
    
    # Get actions for current seed
    for i in range(10000):
        
        x = scaler.transform([observation])
        a = clf.predict(x)[0]
#        a = env.action_space.sample()
        
        actions.append(a)
        
        observation, r, done, info = env.step(a)
        initReward += r
        
        if done:
            break
        
    
    if(initReward >= 200):
        reward, observs, execActions = RunActions(actions, seed)
        observsTraining.append(observs)
        rewardsTraining.append(reward)
        actionsTraining.append(execActions)
        initRewards.append(initReward)
        continue
        
    # Optimize the actions
    for i in range(3):
        actions2 = actions + list(np.random.randint(0, high=2, size=int(200-len(actions))))
        poleAnneal = PoleBalanceAnnealer(actions2, seed)
#            poleAnneal.steps = 15
        poleAnneal.steps = 3000
        poleAnneal.Tmax = 100
        poleAnneal.Tmin = 1
        poleAnneal.updates = 0
        poleAnneal.copy_strategy = "slice"  # since our state is just a list, slice is the fastest way to copy
        actionsOpt, e = poleAnneal.anneal()
        
        # Run optimized actions to get observations
        reward, observs, execActions = RunActions(actionsOpt, seed)
        
        print(' ')
#        print(len(actions2))
        print('opt rewards: ' + str(np.mean(reward)))
        print('init rewards: ' + str(np.mean(initReward)))
        
        observsTraining.append(observs)
        rewardsTraining.append(reward)
        actionsTraining.append(execActions)
        initRewards.append(initReward)
    


#print('opt rewards: ' + str(np.mean(rewardsTraining)))
#print('init rewards: ' + str(np.mean(initRewards)))


# Filter out bad actions
filtObservs = []
filtActions = []
for i in range(len(rewardsTraining)):
    r = rewardsTraining[i]
    initReward = initRewards[i]
    
    if((r > initReward) or (r >= 200)):
        filtObservs.append(np.row_stack(observsTraining[i]))
        filtActions.append(actionsTraining[i])
    
filtObservs = np.row_stack(filtObservs)
filtActions = np.concatenate(filtActions)

print('num filtered: ' + str(len(filtActions)) + ' / ' + str(len(np.concatenate(actionsTraining))) )

# %%

x = filtObservs.copy()
y = filtActions

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

#clf = MLPClassifier(hidden_layer_sizes=(50,30), solver='adam', random_state=2, verbose=1, tol=1e-4)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0, verbose=2, n_jobs=-1, 
                           min_samples_split=5, min_samples_leaf=5)

clf.fit(x,y)

pred = clf.predict(x)
np.sum(pred == y) / len(y)




# %%    
    


    """
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
    """
    
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








    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
