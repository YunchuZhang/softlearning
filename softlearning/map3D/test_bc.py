import tensorflow as tf
import gym
import multiworld
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras
from softlearning.policies.utils import get_policy_from_variant
from softlearning.environments.utils import get_environment_from_params,get_environment_from_params_custom
from examples.instrument import change_env_to_use_correct_mesh
import ipdb
import json
import os
import pickle
st = ipdb.set_trace
# st()
from softlearning.environments.adapters.gym_adapter import GymAdapter
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
from softlearning.policies.utils import get_policy
from softlearning.samplers.simple_sampler import SimpleSampler
import ipdb
st = ipdb.set_trace
import softlearning.map3D.constants as const
# from softlearning.map3D.nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain
from scipy.misc import imsave
# from softlearning.map3D.map3D_trainer import MappingTrainer
import numpy as np
import os
from softlearning.map3D import save_images
multiworld.register_all_envs()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mesh", help="input object")
args = parser.parse_args()

mesh = args.mesh
print(mesh)
change_env_to_use_correct_mesh(mesh)


exploration_steps = 50


gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(session)
session = tf.keras.backend.get_session()

env = gym.make('SawyerPushAndReachEnvEasy-v0',reward_type="puck_success")

env_n = ImageEnv(env,
			   imsize=84,
			   normalize=True,
			   init_camera=init_multiple_cameras,
			   num_cameras=4,
			   depth=True,
			   cam_angles=True,
			   flatten=False)


observation_keys = ["image_observation","depth_observation","cam_angles_observation","image_desired_goal","desired_goal_depth","goal_cam_angle","observation_with_orientation","state_desired_goal", 
"state_achieved_goal" ,"state_observation", "state_desired_goal", "state_achieved_goal", "proprio_observation",  "proprio_desired_goal", "proprio_achieved_goal"]
#observation_keys = ["image_observation","depth_observation","cam_angles_observation","state_observation", "image_desired_goal","desired_goal_depth","goal_cam_angle"]
#st()
env_n.reset()
env = GymAdapter(None,
				 None,
				 env=env_n,
				 observation_keys=observation_keys)


replay_pool = SimpleReplayPool(env, concat_observations=False, max_size=1e4)

policy = get_policy('UniformPolicy', env)


checkpoint_path = "/projects/katefgroup/yunchu/" + "mug2/model.ckpt"
print("--------------")
print(checkpoint_path)
print("--------------")




sampler = SimpleSampler(batch_size=40, max_path_length=60, min_pool_size=0,)
sampler.initialize(env, policy, replay_pool)
path_length = 40

number = 0
success = 0

while True:
	#print("sampling")
	next_observation, reward, terminal, info ,length= sampler.sample()
	#print(terminal)
	#print(length)



	if length !=0:
		number = number + 1
		if length == 1:
			success = success + 1

	if number >49: break

print(success/number)

#st()



