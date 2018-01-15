#!/usr/bin/python3
'''
Train.py
Authors: Rafael Zamora
Last Updated: 3/3/17

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from RLAgent import RLAgent
from DoomScenario import DoomScenario
from Models import DQNModel     #, HDQNModel, all_skills_HDQN, all_skills_shooting_HDQN
import keras.backend as K
import numpy as np

"""
This script is used to train DQN models and Hierarchical-DQN models.

"""

# Training Parameters
# scenario = 'all_skills_shooting.cfg'
scenario = 'rigid_turning.cfg'
model_weights = None
depth_radius = 1.0
depth_contrast = 0.5
learn_param = {
    'learn_algo' : 'dqlearn',
    'exp_policy' : 'e-greedy',
    'frame_skips' : 4,
    'nb_epoch' : 5,     #100
    'steps' : 10,         # 5000
    'batch_size' : 40,
    'memory_size' : 10000,
    'nb_frames' : 3,
    'alpha' : [1.0, 0.1],
    'alpha_rate' : 0.7,
    'alpha_wait' : 10,
    'gamma' : 0.9,
    'epsilon' : [1., 0.1],     #[1.0, 0.1]
    'epsilon_rate' : 0.35,
    'epislon_wait' : 10,   #10
    'nb_tests' : 1,     # 20
}
# training = 'HDQN'
training = 'DQN'
# training_arg = [4,'all_skills_shooting']
training_arg = [4, 'rigid_turning']


def train_model():
    '''
    Method trains primitive DQN-Model.

    '''
    # Initiates VizDoom Scenario
    doom = DoomScenario(scenario)
    # print(doom.get_processed_state(depth_radius, depth_contrast).shape[-2:])
    # Initiates Model
    model = DQNModel(resolution=doom.get_processed_state(depth_radius, depth_contrast).shape[-2:], 
                     nb_frames=learn_param['nb_frames'], actions=doom.actions, 
                     depth_radius=depth_radius, depth_contrast=depth_contrast)

    # print("number of actions: ", len(doom.actions))   # 16

    if model_weights: 
        print("with a pretrained weights-------by amber")
        model.load_weights(model_weights)
    agent = RLAgent(model, **learn_param)

    # Preform Reinforcement Learning on Scenario
    agent.train(doom)



# run the train process
train_model()

