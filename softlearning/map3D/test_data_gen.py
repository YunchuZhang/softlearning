import tensorflow as tf
import gym
import multiworld
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras
from softlearning.policies.utils import get_policy_from_variant
from softlearning.environments.utils import get_environment_from_params,get_environment_from_params_custom
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

exploration_steps = 4


gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(session)
session = tf.keras.backend.get_session()

env = gym.make('SawyerPushAndReachEnvEasy-v0')

env_n = ImageEnv(env,
			   imsize=84,
			   normalize=True,
			   init_camera=init_multiple_cameras,
			   num_cameras=4,
			   depth=True,
			   cam_angles=True,
			   flatten=False)


observation_keys = ["image_observation","depth_observation","cam_angles_observation","state_observation","image_desired_goal","desired_goal_depth","goal_cam_angle"]


env_n.reset()
env = GymAdapter(None,
				 None,
				 env=env_n,
				 observation_keys=observation_keys)


replay_pool = SimpleReplayPool(env, concat_observations=False, max_size=1e4)
#policy = get_policy('UniformPolicy', env)
checkpoint_path = "/home/robertmu/bcyc/result/checkpoint_100"
experiment_path = os.path.dirname(checkpoint_path)

variant_path = os.path.join(experiment_path, 'params.json')
with open(variant_path, 'r') as f:
	variant = json.load(f)
with session.as_default():
	pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
	with open(pickle_path, 'rb') as f:
		picklable = pickle.load(f)

environment_params = (
		variant['environment_params']['evaluation']
		if 'evaluation' in variant['environment_params']
		else variant['environment_params']['training'])


	# environment_params["kwargs"]["observation_keys"] = ["observation","desired_goal","achieved_goal","state_observation","state_desired_goal","state_achieved_goal"\
	# "state_achieved_goal","proprio_observation","proprio_desired_goal","proprio_achieved_goal"]

	# evaluation_environment =  get_environment_from_params_custom(environment_params)

evaluation_environment =  get_environment_from_params(environment_params)

policy = (get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
	
policy.set_weights(picklable['policy_weights'])


sampler = SimpleSampler(batch_size=40, max_path_length=40, min_pool_size=0)
sampler.initialize(env, policy, replay_pool)

while replay_pool.size < exploration_steps:
	print("sampling")
	sampler.sample()
#st()

# imsave("check_03.png",replay_pool.fields["observations.desired_goal_depth"][0,0])
# observation = sampler.random_batch()

# save_images.save_some_samples(sampler)
# def save_replay_buffer(fields):
#   # key_val = fields.keys()
#   for i in range(400):
#     images = fields["observations.image_observation"][i]
#     depths = fields["observations.depth_observation"][i]
#     angles = fields["observations.cam_angles_observation"][i]
#     img_folder_name = "data/images/"+str(i)
#     depth_folder_name = img_folder_name.replace("images","depths")

#     os.makedirs(img_folder_name)
#     os.makedirs(depth_folder_name)

#     for view in range(54):
#       image_view  = images[view]
#       depth_view  = depths[view]
#       elevation,azimuth = angles[view]
#       file_name = "{}_{}.png".format(azimuth,elevation)
#       image_name = img_folder_name + "/" + file_name
#       depth_name = depth_folder_name + "/" + file_name
#       # st()
#       imsave(image_name,image_view)
#       np.save(depth_name,depth_view)




# print()

#const.set_experiment("0520_bulletpush3D_4_multicam_bn_mask_nview1_vp")
# bulletpush = BulletPush3DTensor4_cotrain()
#
# 3d_trainer = MappingTrainer(bulletpush)
#3d_trainer.train_epoch(0, sample_batch=sampler.random_batch, batches=100)
