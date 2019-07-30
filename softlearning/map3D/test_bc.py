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

def rollout_and_gather_data(max_rollouts, mesh, iteration):
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--mesh", help="input object")
	# args = parser.parse_args()

	# mesh = args.mesh
	print(mesh)
	change_env_to_use_correct_mesh(mesh)


	exploration_steps = 100


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

	#policy = get_policy('UniformPolicy', env)
	checkpoint_path = "/projects/katefgroup/robert/fixresult_" + str(mesh) + "/checkpoint_200"
	print("--------------")
	print(checkpoint_path)
	print("--------------")
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
	#import pdb; pdb.set_trace()
	evaluation_environment =  get_environment_from_params(environment_params)

	policy = (get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
		
	policy.set_weights(picklable['policy_weights'])





	sampler = SimpleSampler(batch_size=40, max_path_length=50, min_pool_size=0,mesh = mesh, iteration=iteration)
	sampler.initialize(env, policy, replay_pool)
	length = 40



	number_rollouts = 0
	success = 0
	totalnum = 0


	expert_data_path = "/projects/katefgroup/yunchu/dagger_" + str(mesh)
	if not os.path.exists(expert_data_path):
		os.mkdir(expert_data_path)
	expert_actions = []
	while True:
		#print("sampling")
		current_obs, next_observation, expert_action, reward, terminal, info ,length= sampler.sample()
		expert_actions.append(expert_action)
		print("current_obs", current_obs)
		#st()
		#print(terminal)
		#print(length)

		#save observation
		

		# expert_data = {'actions':expert_action,
		# 		   'observation_with_orientation':current_obs[:-2], 
		# 		   'state_desired_goal':current_obs[-2:]}



		if length !=0:
			number_rollouts += 1

			

				#print(expert_data_path)
			#expert_actions = []
			if length == 1:
				success = success + 1
			print(number_rollouts)

		if number_rollouts >max_rollouts: break


	onlyfiles = next(os.walk(expert_data_path))[2]
	totalnum = len(onlyfiles)
	print('---------')
	print("before",totalnum)

	for counter in range(len(expert_actions)):
		#save the dagger expert trajectories 
		expert_data = {'image_observation': np.array(replay_pool.fields["observations.image_observation"][counter]),
		   'depth_observation': np.array(replay_pool.fields["observations.depth_observation"][counter]),
		   'cam_angles_observation':np.array(replay_pool.fields["observations.cam_angles_observation"][counter]),
		   'actions':expert_actions[counter],
		   'rewards':np.array(replay_pool.fields["rewards"][counter]),
		   'observation_with_orientation':np.array(replay_pool.fields["observations.observation_with_orientation"][counter]),
		   'state_desired_goal':np.array(replay_pool.fields["observations.state_desired_goal"][counter][3:]),
		   'terminals':np.array(replay_pool.fields["terminals"][counter])}
		#print(expert_data)

		#print('saving'+'{:d}'.format(counter)+'.pkl')
		with open(os.path.join(expert_data_path, 'state' + "{:d}".format(totalnum+counter) + '.pkl'), 'wb') as f:
			pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
	#st()
	onlyfiles = next(os.walk(expert_data_path))[2]
	totalnum = len(onlyfiles)
	print('---------')
	print("after",totalnum)


	succes_rate = success/max_rollouts
	print(succes_rate)
	onlyfiles = next(os.walk(expert_data_path))[2] 
	print("Total_Sample_data",len(onlyfiles))

	return succes_rate

#st()
#rollout
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh", help="input object")
	args = parser.parse_args()

	mesh = args.mesh
	max_rollouts = 50
	rollout_and_gather_data(max_rollouts, mesh)

