import numpy as np
import sys
import pickle
from os import mkdir
from os.path import join, isdir
from numpy import linalg as LA
from math import sqrt, inf
from decimal import Decimal
import time
import gym
import envs
from gym import spaces, logger
from scenario_objects import Point, Cell, User, Environment
import plotting
from my_utils import *
import agent
from Astar import *

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'UAVEnv-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)



'''
print('States: %r' %env.observation_space)

print('Actions: %r' %env.action_space)
N_UAVS = 2

#o = env.reset()
for t in range(100):
    env.render()
    #time.sleep(0.1)
    for i in range(30):
	    actions = [env.action_space.sample() for uav in range(N_UAVS)]
	    for uav in range(N_UAVS):
	    	while (env.q_table_action_set[actions[uav]]==GO_TO_CS): # --> The action space decreases or increases for each UAV according to its buttery level, but the 'general' action space is invariate.
	    		actions[uav] = env.action_space.sample()
	    o, r, done, info = env.step(actions)
    print("| Episode: {ep:1d} - UAVs rewards: {}".format(r, ep=t+1))
    #env.render()
'''