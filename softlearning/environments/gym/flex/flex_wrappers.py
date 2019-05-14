from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv
from autolab_core import YamlConfig
import numpy as np

import gym
from gym import spaces


class FetchReach(gym.Env):
        def __init__(self):
                self.cfg = YamlConfig('/home/katadh/Documents/Research/VMGE/softlearning/softlearning/environments/gym/flex/fetch_cube.yaml')
            # self.cfg = YamlConfig('/home/a/workspace/katerina/rlkit/examples/her/cfg/fetch_cube.yaml')

                self.numAgents = self.cfg['scene']['NumAgents'] = 1
                self.cfg['scene']['NumPerRow'] = np.sqrt(np.floor(self.numAgents))
                self.cfg['scene']['SampleInitStates'] = True
                self.cfg['scene']['InitialGrasp'] = False
                self.cfg['scene']['RelativeTarget'] = False
                self.cfg['scene']['DoDeltaPlanarControl'] = True
                self.cfg['scene']['DoGripperControl'] = True
                self.cfg['scene']['InitialGraspProbability'] = 1
                self.cfg['scene']['DoWristRollControl'] = False

                set_flex_bin_path('/home/katadh/Documents/Research/flex/FlexRobotics/bin')
                # set_flex_bin_path('/home/a/workspace/FlexRobotics/bin')
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
	def __init__(self):
		self.cfg = YamlConfig('/home/katadh/Documents/Research/VMGE/softlearning/softlearning/environments/gym/flex/fetch_push.yaml')
		# self.cfg = YamlConfig('/home/a/workspace/katerina/rlkit/examples/her/cfg/fetch_push.yaml')

		self.numAgents = self.cfg['scene']['NumAgents'] = 1
		self.cfg['scene']['NumPerRow'] = np.sqrt(np.floor(self.numAgents))
		self.cfg['scene']['SampleInitStates'] = True
		self.cfg['scene']['InitialGrasp'] = False
		self.cfg['scene']['RelativeTarget'] = False
		self.cfg['scene']['DoDeltaPlanarControl'] = True
		self.cfg['scene']['DoGripperControl'] = True
		self.cfg['scene']['InitialGraspProbability'] = 1
		self.cfg['scene']['DoWristRollControl'] = False

		set_flex_bin_path('/home/katadh/Documents/Research/flex/FlexRobotics/bin')
		# set_flex_bin_path('/home/a/workspace/FlexRobotics/bin')
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
