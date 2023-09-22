from pommerman.agents.base_agent import BaseAgent
from pommerman import characters

from tensorflow.keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from tensorflow.keras.models import Sequential
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np


class DQNAgent(BaseAgent):
    """Agent using a Deep Q-Network.
    """

    def __init__(self, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)

        self.qvals_s = 0
        

    def act(self, obs, action_space): 
        #Action from Q table
        action = np.argmax(self.qvals_s)
    
        return action
    
    def episode_end(self, reward):
        return 

    def restructure_state(self, obs):

        board = np.array(obs['board'])
        bomb_blast_strength = np.array(obs['bomb_blast_strength'])
        bomb_life = np.array(obs['bomb_life'])
        bomb_moving_direction = np.array(obs['bomb_moving_direction'])
        flame_life = np.array(obs['flame_life'])
        ammo = np.zeros((11, 11))
        ammo[0, 0] = obs['ammo']

        array = np.stack([board, bomb_blast_strength, bomb_life, bomb_moving_direction, flame_life, ammo], axis=2)

        return array

    def make_network(self, action_space, layers):
        n_actions = action_space

        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape=(11, 11, layers))) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten()) 
        model.add(Dense(64))

        model.add(Dense(n_actions, activation = 'linear'))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.0005), metrics=['accuracy'])

        return model

    def set_qvals_s(self, new_qvals_s):
        self.qvals_s = new_qvals_s

