from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv
from autolab_core import YamlConfig
import numpy as np
from gym import spaces
import gym
import os

CONFIG_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) + '/cfg/'
# FLEX_BIN_PATH = '/home/a/workspace/FlexRobotics/bin'
FLEX_BIN_PATH = '/home/katadh/Documents/Research/flex/FlexRobotics/bin'


class FetchReach(gym.Env):
        def __init__(self, render=0):
                self.cfg = YamlConfig(CONFIG_DIRECTORY + 'fetch_reach.yaml')

                self.cfg['gym']['renderBackend'] = render
                self.num_agentss = self.cfg['scene']['NumAgents'] = 1
                self.cfg['scene']['NumPerRow'] = 1
                self.cfg['scene']['SampleInitStates'] = True
                self.cfg['scene']['InitialGrasp'] = False
                self.cfg['scene']['RelativeTarget'] = False
                self.cfg['scene']['DoDeltaPlanarControl'] = True
                self.cfg['scene']['DoGripperControl'] = False
                self.cfg['scene']['InitialGraspProbability'] = 1
                self.cfg['scene']['DoWristRollControl'] = False

                set_flex_bin_path(FLEX_BIN_PATH)
                self.env = FlexVecEnv(self.cfg)

                self.previous_observation = None
                self.action_space = self.env.action_space
                self.observation_space = spaces.Dict(
                        {"achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)})

        def reset(self):
                self.env.reset()
                for i in range(20):
                        obs, reward, done, info = self.env.step(np.zeros(3))
                self.previous_observation = obs[0][:3].copy()

                return {'observation': np.append(obs[0][:3], np.zeros(3)),
                        'achieved_goal': obs[0][:3].copy(),
                        'desired_goal': obs[0][3:6].copy()}

        def step(self, action):
                obs, reward, done, info = self.env.step(action)
                velocity = obs[0][:3] - self.previous_observation
                self.previous_observation = obs[0][:3].copy()
                return {'observation': np.append(obs[0][:3], velocity),
                        'achieved_goal': obs[0][:3].copy(),
                        'desired_goal': obs[0][3:6].copy()}, reward[0], done[0], {'is_success': reward[0] + 1}

        def compute_reward(self, achieved_goal, desired_goal, info):
                return self.env.compute_reward(achieved_goal[None, :], desired_goal[None, :], info)[0]


class FetchPush(gym.Env):
        def __init__(self, render=0):
                self.cfg = YamlConfig(CONFIG_DIRECTORY + 'fetch_push.yaml')

                self.cfg['gym']['renderBackend'] = render
                self.num_agentss = self.cfg['scene']['NumAgents'] = 1
                self.cfg['scene']['NumPerRow'] = np.sqrt(np.floor(self.num_agentss))
                self.cfg['scene']['SampleInitStates'] = True
                self.cfg['scene']['InitialGrasp'] = False
                self.cfg['scene']['RelativeTarget'] = False
                self.cfg['scene']['DoDeltaPlanarControl'] = True
                self.cfg['scene']['DoGripperControl'] = False
                self.cfg['scene']['InitialGraspProbability'] = 1
                self.cfg['scene']['DoWristRollControl'] = False

                set_flex_bin_path(FLEX_BIN_PATH)
                self.env = FlexVecEnv(self.cfg)

                self.previous_observation = None
                self.action_space = self.env.action_space
                self.observation_space = spaces.Dict(
                        {"achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)})

        def reset(self):
                self.env.reset()
                for i in range(20):
                        obs, reward, done, info = self.env.step(np.zeros(3))
                        self.previous_observation = np.append(obs[0][:3], obs[0][6:]).copy()

                return {'observation': np.append(np.append(obs[0][:3], obs[0][6:]), np.zeros(6)),
                        'achieved_goal': obs[0][6:].copy(),
                        'desired_goal': obs[0][3:6].copy()}

        def step(self, action):
                obs, reward, done, info = self.env.step(action)
                velocity = np.append(obs[0][:3], obs[0][6:]) - self.previous_observation
                self.previous_observation = np.append(obs[0][:3], obs[0][6:]).copy()
                return {'observation': np.append(np.append(obs[0][:3], obs[0][6:]), velocity),
                        'achieved_goal': obs[0][6:].copy(),
                        'desired_goal': obs[0][3:6].copy()}, reward[0], done[0], {'is_success': reward[0] + 1}

        def compute_reward(self, achieved_goal, desired_goal, info):
                return self.env.compute_reward(achieved_goal[None, :], desired_goal[None, :], info)[0]


class FetchReachMultiRobot(gym.Env):
        def __init__(self, render=0):
                self.cfg = YamlConfig(CONFIG_DIRECTORY + 'fetch_reach.yaml')

                self.cfg['gym']['renderBackend'] = render

                self.num_agents = self.cfg['scene']['NumAgents'] = 50
                self.cfg['scene']['NumPerRow'] = np.sqrt(np.floor(self.num_agents))
                self.cfg['scene']['SampleInitStates'] = True
                self.cfg['scene']['InitialGrasp'] = False
                self.cfg['scene']['RelativeTarget'] = False
                self.cfg['scene']['DoDeltaPlanarControl'] = True
                self.cfg['scene']['DoGripperControl'] = False
                self.cfg['scene']['InitialGraspProbability'] = 1
                self.cfg['scene']['DoWristRollControl'] = False

                set_flex_bin_path(FLEX_BIN_PATH)
                self.env = FlexVecEnv(self.cfg)

                self.previous_observation = None
                self.action_space = self.env.action_space
                self.observation_space = spaces.Dict(
                        {"achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)})

        def reset(self):
                self.env.reset()
                for i in range(20):
                        obs, reward, done, info = self.env.step(np.zeros((self.num_agents, 3)))
                self.previous_observation = obs[:, :3].copy()

                return {'observation': np.append(obs[:, :3], np.zeros((self.num_agents, 3)), axis=1),
                        'achieved_goal': obs[:, :3].copy(),
                        'desired_goal': obs[:, 3:6].copy()}

        def step(self, action):
                obs, reward, done, info = self.env.step(action)
                velocity = obs[:, :3] - self.previous_observation
                self.previous_observation = obs[:, :3].copy()
                return {'observation': np.append(obs[:, :3], velocity, axis=1),
                        'achieved_goal': obs[:, :3].copy(),
                        'desired_goal': obs[:, 3:6].copy()}, reward, done, {'is_success': reward + 1}

        def compute_reward(self, achieved_goal, desired_goal, info):
                return self.env.compute_reward(achieved_goal[None, :], desired_goal[None, :], info)[0]


class FetchPushMultiRobot(gym.Env):
        def __init__(self, render=0):
                self.cfg = YamlConfig(CONFIG_DIRECTORY + 'fetch_push.yaml')

                self.cfg['gym']['renderBackend'] = render

                self.num_agents = self.cfg['scene']['NumAgents'] = 50
                self.cfg['scene']['NumPerRow'] = np.sqrt(np.floor(self.num_agents))
                self.cfg['scene']['SampleInitStates'] = True
                self.cfg['scene']['InitialGrasp'] = False
                self.cfg['scene']['RelativeTarget'] = False
                self.cfg['scene']['DoDeltaPlanarControl'] = True
                self.cfg['scene']['DoGripperControl'] = False
                self.cfg['scene']['InitialGraspProbability'] = 1
                self.cfg['scene']['DoWristRollControl'] = False

                set_flex_bin_path(FLEX_BIN_PATH)
                self.env = FlexVecEnv(self.cfg)

                self.previous_observation = None
                self.action_space = self.env.action_space
                self.observation_space = spaces.Dict(
                        {"achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                         "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)})

        def reset(self):
                self.env.reset()

                action = np.zeros((self.num_agents, 3))
                for i in range(20):
                        obs, reward, done, info = self.env.step(action)
                        self.previous_observation = np.concatenate((obs[:, :3], obs[:, 6:]), axis=1).copy()

                return {'observation': np.concatenate((obs[:, :3], obs[:, 6:], np.zeros((self.num_agents, 6))), axis=1),
                        'achieved_goal': obs[:, 6:].copy(),
                        'desired_goal': obs[:, 3:6].copy()}

        def step(self, action):
                obs, reward, done, info = self.env.step(action)
                velocity = np.concatenate((obs[:, :3], obs[:, 6:]), axis=1) - self.previous_observation
                self.previous_observation = np.concatenate((obs[:, :3], obs[:, 6:]), axis=1).copy()
                return {'observation': np.concatenate((obs[:, :3], obs[:, 6:], velocity), axis=1),
                        'achieved_goal': obs[:, 6:].copy(),
                        'desired_goal': obs[:, 3:6].copy()}, reward, done, {'is_success': reward + 1}

        def compute_reward(self, achieved_goal, desired_goal, info):
                return self.env.compute_reward(achieved_goal[None, :], desired_goal[None, :], info)[0]
