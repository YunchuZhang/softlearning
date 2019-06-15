import tensorflow as tf
import gym
import multiworld
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras

from softlearning.environments.adapters.gym_adapter import GymAdapter
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
from softlearning.policies.utils import get_policy
from softlearning.samplers.simple_sampler import SimpleSampler

import softlearning.map3D.constants as const
from softlearning.map3D.nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain

from softlearning.map3D.map3D_trainer import MappingTrainer

multiworld.register_all_envs()

exploration_steps = 1000

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(session)
session = tf.keras.backend.get_session()

env = gym.make('SawyerPushAndReachEnvEasy-v0')
env = ImageEnv(env,
               imsize=84,
               normalize=True,
               init_camera=init_multiple_cameras,
               num_cameras=4,
               depth=True,
               cam_angles=True,
               flatten=False)

env = GymAdapter(None,
                 None,
                 env=env,
                 observation_keys=['image_observation',
                                   'depth_observation',
                                   'cam_angles_observation'])

replay_pool = SimpleReplayPool(env, concat_observations=False, max_size=1e4)
policy = get_policy('UniformPolicy', env)

sampler = SimpleSampler(batch_size=1, max_path_length=50, min_pool_size=0)
sampler.initialize(env, policy, replay_pool)

while replay_pool.size < exploration_steps:
    sampler.sample()

print(sampler.random_batch())

#const.set_experiment("0520_bulletpush3D_4_multicam_bn_mask_nview1_vp")
#bulletpush = BulletPush3DTensor4_cotrain()
#
#3d_trainer = MappingTrainer(bulletpush)
#3d_trainer.train_epoch(0, sample_batch=sampler.random_batch, batches=100)
