#!/usr/bin/python3
'''
Models.py
Authors: Rafael Zamora
Last Updated: 3/4/17

'''

"""
Script defines the models used by the Doom Ai.
Models are built using the Keras high-level neural network library.
Keras uses TensorFlow and Theano as back-ends.

"""
import itertools as it
import numpy as np
np.set_printoptions(precision=3)
import keras.callbacks as KC
import keras.backend as K
K.set_image_data_format("channels_first")
from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop, SGD
from keras.utils import plot_model
from sklearn.preprocessing import normalize

class DQNModel:
    """
    DQNModel class is used to define DQN models for the
    Vizdoom environment.

    """

    def __init__(self, resolution=(120, 160), nb_frames=1, actions=[], depth_radius=1.0, depth_contrast=0.8, distilled=False):
        '''
        DQN models have the following network architecture:
        - Input : (# of previous frames, img_width, img_length)
        - ConvLayer: 32 filters, 8x8 filter size, 4x4 stride, rectifier activation
        - ConvLayer: 64 filters, 5x5 filter size, 4x4 stride, rectifier activation
        - FullyConnectedLayer : 4032 nodes with 0.5 dropout rate
        - Output: (# of available actions)

        The loss function is mean-squared error.
        The optimizer is RMSprop with a learning rate of 0.0001

        '''
        # Network Parameters
        self.resolution = resolution
        self.actions = actions
        self.nb_actions = len(actions)
        self.nb_frames = nb_frames
        self.depth_radius = depth_radius
        self.depth_contrast = depth_contrast       # the parameter C balancing between depth and RGB data
        self.loss_fun = 'mse'
        self.optimizer = RMSprop(lr=0.0001)

        # Input Layers
        self.x0 = Input(shape=(nb_frames, resolution[0], resolution[1]))

        # Convolutional Layer
        m = Conv2D(32, (8, 8), strides = (4,4), activation='relu', )(self.x0)
        m = Conv2D(64, (5, 5), strides = (4,4), activation='relu')(m)
        m = Flatten()(m)

        # Fully Connected Layer
        m = Dense(4032, activation='relu')(m)
        m = Dropout(0.5)(m)

        # Output Layer
        if distilled:
            self.y0 = Dense(self.nb_actions, activation='softmax')(m)
        else:
            self.y0 = Dense(self.nb_actions)(m)

        self.online_network = Model(inputs=self.x0, outputs=self.y0)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun)
        self.target_network = None
        self.state_predictor = None
        #self.online_network.summary()
        #plot_model(self.online_network, to_file='../doc/model.png', show_shapes=True, show_layer_names=False)
        #tbcall = KC.TensorBoard(log_dir="../doc/logs", histogram_freq=0, write_graph=True, write_images=True)
        #tbcall.set_model(self)

    def predict(self, game, q):
        '''
        Method selects predicted action from set of available actions using the
        max-arg q value.

        '''
        a = self.actions[q]
        return a

    def softmax_q_values(self, S, actions, q_=None):
        '''
        Method returns softmax of predicted q values indexed according to the
        desired list of actions.

        '''
        # Calculate Softmax of Q values
        q = self.online_network.predict(S)
        max_q = int(np.argmax(q[0]))

        # Index Q values according to inputed list of actions
        final_q = [0 for i in range(len(actions))]
        for j in range(len(model_actions)):
            for i in range(len(actions)):
                if model_actions[j] == actions[i]:
                    final_q[i] = q[0][j]

        # ASk dr. Pierce about sharpening data points.
        final_q = np.array(final_q)
        softmax_q = np.exp((final_q)/0.15)
        softmax_q =  softmax_q / softmax_q.sum(axis=0)

        return softmax_q, max_q

    def load_weights(self, filename):
        '''
        Method loads DQN model weights from file located in /data/model_weights/ folder.

        '''
        self.online_network.load_weights('../data/model_weights/' + filename)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun)

    def save_weights(self, filename):
        '''
        Method saves DQN model weights to file located in /data/model_weights/ folder.

        '''
        self.online_network.save_weights('../data/model_weights/' + filename, overwrite=True)




