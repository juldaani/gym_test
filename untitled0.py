#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 18:10:45 2018

@author: juho
"""

import gym


env = gym.make('Pong-ram-v0')
env.reset()
for _ in range(300):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    
    
env.reset()

env.observation_space.n

len(env.observation_space.low)

env.action_space

env.action_space.sample()
    
# %%


    
import gym


env = gym.make('CartPole-v0')

for i_episode in range(20):
    observation = env.reset()
    
    for t in range(100):
        env.render()
#        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        print(reward)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        


env.action_space.n

env.observation_space





























