#!/usr/bin/python3
'''
DoomScenario.py
Authors: Rafael Zamora
Last Updated: 3/26/17

'''

"""
This script defines the instance of Vizdoom used to train and test
Reinforcement Learning Models.

"""

from vizdoom import DoomGame, Mode, ScreenResolution
import itertools as it
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from tqdm import tqdm

def softmax(x, t):
    '''
    Method defines softmax function

    '''
    e_x = np.exp(x - np.max(x))/t
    return e_x / e_x.sum(axis=0)

class DoomScenario:
    """
    DoomScenario class runs instances of Vizdoom according to scenario
    configuration (.cfg) files.

    Scenario Configuration files for this project are located in
    the /src/configs/ folder.

    """

    def __init__(self, config_filename):
        '''
        Method initiates Vizdoom with desired configuration file.

        '''
        self.config_filename = config_filename
        self.game = DoomGame()
        self.game.load_config("configs/" + config_filename)
        self.game.set_window_visible(False)
        self.game.init()

        self.res = (self.game.get_screen_height(), self.game.get_screen_width())
        self.actions = [list(a) for a in it.product([0, 1], repeat=self.game.get_available_buttons_size())]

        self.pbar = None
        self.game.new_episode()

    def play(self, action, tics):
        '''
        Method advances state with desired action for a number of tics.

        '''
        self.game.set_action(action)
        self.game.advance_action(tics, True)
        if self.pbar: self.pbar.update(int(tics))

    def get_processed_state(self, depth_radius, depth_contrast):
        '''
        Method processes the Vizdoom RGB and depth buffer into
        a composite one channel image that can be used by the Models.

        depth_radius defines how far the depth buffer sees with 1.0 being
        as far as ViZDoom allows.

        depth_contrast defines how much of the depth buffer is in the final
        processed image as compared to the greyscaled RGB buffer.
        **processed = (1-depth_contrast)* grey_buffer + depth_contrast*depth_buffer

        '''
        state = self.game.get_state()
        if not self.game.is_episode_finished():
            img = state.screen_buffer           # screen pixels
        # print(img)
            screen_buffer = np.array(img).astype('float32')/255
            # print(screen_buffer.shape)    # (3, 120, 160)
        try:
            # Grey Scaling
            grey_buffer = np.dot(np.transpose(screen_buffer, (1, 2, 0)), [0.21, 0.72, 0.07])
            # print(grey_buffer.shape)     # (120, 160)

            # Depth Radius
            depth_buffer = np.array(state.depth_buffer).astype('float32')/255
            depth_buffer[(depth_buffer > depth_radius)] = depth_radius #Effects depth radius
            depth_buffer_filtered = (depth_buffer - np.amin(depth_buffer))/ (np.amax(depth_buffer) - np.amin(depth_buffer))

            # Depth Contrast
            processed_buffer = ((1 - depth_contrast) * grey_buffer) + (depth_contrast* (1- depth_buffer))
            processed_buffer = (processed_buffer - np.amin(processed_buffer))/ (np.amax(processed_buffer) - np.amin(processed_buffer))
            processed_buffer = np.round(processed_buffer, 6)
            processed_buffer = processed_buffer.reshape(self.res[-2:])
        except:
            processed_buffer = np.zeros(self.res[-2:])
        return processed_buffer       # balance the depth & RGB data

    def run(self, agent, save_replay='', verbose=False, return_data=False):
        '''
        Method runs a instance of DoomScenario.

        '''
        if return_data:
            data_S = []
            data_a = []
        if verbose:
            print("\nRunning Simulation:", self.config_filename)
            self.pbar = tqdm(total=self.game.get_episode_timeout())

        # Initiate New Instance
        self.game.close()
        self.game.set_window_visible(False)
        self.game.add_game_args("+vid_forcesurface 1 ")
        self.game.init()
        if save_replay != '': self.game.new_episode("../data/replay_data/" + save_replay)
        else: self.game.new_episode()

        # Run Simulation
        while not self.game.is_episode_finished():
            S = agent.get_state_data(self)
            q = agent.model.online_network.predict(S)
            if np.random.random() < 0.1:
                q = np.random.choice(len(q[0]), 1, p=softmax(q[0], 1))[0]
            else: q = int(np.argmax(q[0]))
            a = agent.model.predict(self, q)
            if return_data:
                delta = np.zeros((len(self.actions)))
                a_ = np.cast['int'](a)
                delta[a_] = 1
                data_S.append(S.reshape(S.shape[1], S.shape[2], S.shape[3]))
                data_a.append(delta)
            if not self.game.is_episode_finished():
                self.play(a, agent.frame_skips+1)
            if agent.model.__class__.__name__ == 'HDQNModel' and not self.game.is_episode_finished():
                if q >= len(agent.model.actions):
                    for i in range(agent.model.skill_frame_skip):
                        if not self.game.is_episode_finished():
                            a = agent.model.predict(self, q)
                            self.play(a, agent.frame_skips+1)
                        else: break

        # Reset Agent and Return Score
        agent.frames = None
        if agent.model.__class__.__name__ == 'HDQNModel': agent.model.sub_model_frames = None
        score = self.game.get_total_reward()
        if verbose:
            self.pbar.close()
            print("Total Score:", score)
        if return_data:
            data_S = np.array(data_S)
            data_a = np.array(data_a)
            return [data_S, data_a]
        return score

    def replay(self, filename, verbose=False, doom_like=False):
        '''
        Method runs a replay of the simulations at 800 x 600 resolution.

        '''
        print("\nRunning Replay:", filename)

        # Initiate Replay
        self.game.close()
        self.game.set_screen_resolution(ScreenResolution.RES_800X600)
        self.game.set_window_visible(True)
        self.game.add_game_args("+vid_forcesurface 1")
        if doom_like:
            self.game.set_render_hud(True)
            self.game.set_render_minimal_hud(False)
            self.game.set_render_crosshair(False)
            self.game.set_render_weapon(True)
            self.game.set_render_particles(True)
        self.game.init()
        self.game.replay_episode("../data/replay_data/" + filename)

        # Run Replay
        while not self.game.is_episode_finished():
            if verbose: print("Reward:", self.game.get_last_reward())
            self.game.advance_action()

        # Print Score
        score = self.game.get_total_reward()
        print("Total Score:", score)
        self.game.close()

    def apprentice_run(self, test=False):
        '''
        Method runs an apprentice data gathering.

        '''
        # Initiate New Instance
        self.game.close()
        self.game.set_mode(Mode.SPECTATOR)
        self.game.set_screen_resolution(ScreenResolution.RES_800X600)
        self.game.set_window_visible(True)
        self.game.set_ticrate(30)
        self.game.init()
        self.game.new_episode()

        # Run Simulation
        while not self.game.is_episode_finished():
            self.game.advance_action()
        self.game.close()
