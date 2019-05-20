from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv
from autolab_core import YamlConfig
import numpy as np

import gym
from gym import spaces

import os

CONFIG_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
#FLEX_BIN_PATH = '/home/a/workspace/FlexRobotics/bin'
FLEX_BIN_PATH = '/home/katadh/Documents/Research/flex/FlexRobotics/bin'

class FetchReach(gym.Env):
        def __init__(self, render=False):
                self.cfg = YamlConfig(CONFIG_DIRECTORY + '/fetch_cube.yaml')

                self.cfg['gym']['renderBackend'] = render
                self.numAgents = self.cfg['scene']['NumAgents'] = 1
                self.cfg['scene']['NumPerRow'] = np.sqrt(np.floor(self.numAgents))
                self.cfg['scene']['SampleInitStates'] = True
                self.cfg['scene']['InitialGrasp'] = False
                self.cfg['scene']['RelativeTarget'] = False
                self.cfg['scene']['DoDeltaPlanarControl'] = True
                self.cfg['scene']['DoGripperControl'] = True
                self.cfg['scene']['InitialGraspProbability'] = 1
                self.cfg['scene']['DoWristRollControl'] = False

                set_flex_bin_path(FLEX_BIN_PATH)
                self.env = FlexVecEnv(self.cfg)

                self.previous_observation = None
                self.action_space = self.env.action_space
                self.observation_space = spaces.Dict(
                        {"achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)})

        def reset(self):
                self.env.reset()
                for i in range(20):
                        obs, reward, done, info = self.env.step(np.array([0, 0, 0, -1]))
                        self.previous_observation = obs[0][:4].copy()

                return {'observation': np.append(obs[0][:4], np.zeros(4)), 'achieved_goal': obs[0][:3],
                        'desired_goal': obs[0][4:8]}

        def step(self, action):
                obs, reward, done, info = self.env.step(action)
                velocity = obs[0][:4] - self.previous_observation
                self.previous_observation = obs[0][:4].copy()
                return {'observation': np.append(obs[0][:4], velocity), 'achieved_goal': obs[0][:3],
                        'desired_goal': obs[0][4:8]}, reward[0], done[0], {'is_success': reward[0] + 1}

        def compute_reward(self, achieved_goal, desired_goal, info):
                return self.env.compute_reward(achieved_goal[None, :], desired_goal[None, :], info)


class FetchPush(gym.Env):
        def __init__(self, render=0):
                self.cfg = YamlConfig(CONFIG_DIRECTORY + '/fetch_push.yaml')

                self.numAgents = self.cfg['scene']['NumAgents'] = 1
                self.cfg['scene']['NumPerRow'] = np.sqrt(np.floor(self.numAgents))
                self.cfg['scene']['SampleInitStates'] = True
                self.cfg['scene']['InitialGrasp'] = False
                self.cfg['scene']['RelativeTarget'] = False
                self.cfg['scene']['DoDeltaPlanarControl'] = True
                self.cfg['scene']['DoGripperControl'] = True
                self.cfg['scene']['InitialGraspProbability'] = 1
                self.cfg['scene']['DoWristRollControl'] = False

                self.cfg['gym']['renderBackend'] = render

                set_flex_bin_path(FLEX_BIN_PATH)
                self.env = FlexVecEnv(self.cfg)

                self.previous_observation = None
                self.action_space = self.env.action_space
                self.observation_space = spaces.Dict(
                        {"achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)})

        def reset(self):
                self.env.reset()
                for i in range(20):
                        obs, reward, done, info = self.env.step(np.array([0, 0, 0, -1]))
                        self.previous_observation = np.append(obs[0][:4], obs[0][7:]).copy()

                return {'observation': np.append(np.append(obs[0][:4], obs[0][7:]), np.zeros(7)), 'achieved_goal': obs[0][7:],
                        'desired_goal': obs[0][4:7]}

        def step(self, action):
                obs, reward, done, info = self.env.step(action)
                velocity = np.append(obs[0][:4], obs[0][7:]) - self.previous_observation
                self.previous_observation = np.append(obs[0][:4], obs[0][7:]).copy()
                return {'observation': np.append(np.append(obs[0][:4], obs[0][7:]), velocity), 'achieved_goal': obs[0][7:],
                        'desired_goal': obs[0][4:7]}, reward[0], done[0], {'is_success': reward[0] + 1}

        def compute_reward(self, achieved_goal, desired_goal, info):
                return self.env.compute_reward(achieved_goal[None, :], desired_goal[None, :], info)


class FetchReachMultiRobot(gym.Env):
        def __init__(self):
                self.cfg = YamlConfig(CONFIG_DIRECTORY + '/fetch_reach.yaml')

                self.cfg['scene']['SampleInitStates'] = True
                self.cfg['scene']['InitialGrasp'] = False
                self.cfg['scene']['RelativeTarget'] = False
                self.cfg['scene']['DoDeltaPlanarControl'] = True
                self.cfg['scene']['DoGripperControl'] = True
                self.cfg['scene']['InitialGraspProbability'] = 1
                self.cfg['scene']['DoWristRollControl'] = False

                set_flex_bin_path(FLEX_BIN_PATH)
                self.env = FlexVecEnv(self.cfg)

                self.previous_observation = None
                self.action_space = self.env.action_space
                self.observation_space = spaces.Dict(
                        {"achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)})

        def reset(self):
                self.env.reset()
                for i in range(20):
                        obs, reward, done, info = self.env.step(np.zeros((50, 4)))
                self.previous_observation = obs[:, :4].copy()

                return {'observation': np.append(obs[:, :4], np.zeros((50, 4)), axis=1), 'achieved_goal': obs[:, :3],
                        'desired_goal': obs[:, 4:8]}

        def step(self, action):
                obs, reward, done, info = self.env.step(action)
                velocity = obs[:, :4] - self.previous_observation
                self.previous_observation = obs[:, :4].copy()
                return {'observation': np.append(obs[:, :4], velocity, axis=1), 'achieved_goal': obs[:, :3],
                        'desired_goal': obs[:, 4:8]}, reward, done, {'is_success': reward + 1}

        def compute_reward(self, achieved_goal, desired_goal, info):
                return self.env.compute_reward(achieved_goal[None, :], desired_goal[None, :], info)


class FetchPushMultiRobot(gym.Env):
        def __init__(self):
                self.cfg = YamlConfig(CONFIG_DIRECTORY + '/fetch_push.yaml')

                self.numAgents = self.cfg['scene']['NumAgents'] = 50
                self.cfg['scene']['NumPerRow'] = np.sqrt(np.floor(self.numAgents))
                self.cfg['scene']['SampleInitStates'] = True
                self.cfg['scene']['InitialGrasp'] = False
                self.cfg['scene']['RelativeTarget'] = False
                self.cfg['scene']['DoDeltaPlanarControl'] = True
                self.cfg['scene']['DoGripperControl'] = True
                self.cfg['scene']['InitialGraspProbability'] = 1
                self.cfg['scene']['DoWristRollControl'] = False

                set_flex_bin_path(FLEX_BIN_PATH)
                self.env = FlexVecEnv(self.cfg)

                self.previous_observation = None
                self.action_space = self.env.action_space
                self.observation_space = spaces.Dict(
                        {"achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)})

        def reset(self):
                self.env.reset()

                action = np.zeros((50, 4))
                action[:, -1] = -1
                for i in range(20):
                        obs, reward, done, info = self.env.step(action)
                        self.previous_observation = np.concatenate((obs[:, :4], obs[:, 7:]), axis=1).copy()

                return {'observation': np.concatenate((obs[:, :4], obs[:, 7:], np.zeros((50, 7))), axis=1), 'achieved_goal': obs[:, 7:],
                        'desired_goal': obs[:, 4:7]}

        def step(self, action):
                obs, reward, done, info = self.env.step(action)
                velocity = np.concatenate((obs[:, :4], obs[:, 7:]), axis=1) - self.previous_observation
                self.previous_observation = np.concatenate((obs[:, :4], obs[:, 7:]), axis=1).copy()
                return {'observation': np.concatenate((obs[:, :4], obs[:, 7:], velocity), axis=1), 'achieved_goal': obs[:, 7:],
                        'desired_goal': obs[:, 4:7]}, reward, done, {'is_success': reward + 1}

        def compute_reward(self, achieved_goal, desired_goal, info):
                return self.env.compute_reward(achieved_goal[None, :], desired_goal[None, :], info)
