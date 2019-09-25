import os
import numpy as np

import tensorflow as tf

from softlearning.environments.adapters.gym_adapter import GymAdapter
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
from softlearning.policies.utils import get_policy
from softlearning.samplers.simple_sampler import SimpleSampler

import discovery.hyperparams as hyp
from discovery.model_mujoco_online import MUJOCO_ONLINE

exploration_steps = 500
train_iters = 1000
sample_steps_per_iter = 2

log_freq = hyp.log_freqs['train']

observation_keys = ["image_observation",
                    "depth_observation",
                    "cam_info_observation",
                    "state_observation",
                    "state_desired_goal",
                    "image_desired_goal",
                    "depth_desired_goal",
                    "cam_info_goal"]

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(session)
session = tf.keras.backend.get_session()

env = GymAdapter('SawyerMulticameraReach',
                 'v0',
                 observation_keys=observation_keys)


replay_pool = SimpleReplayPool(env,
                               concat_observations=False,
                               max_size=1e4)

policy = get_policy('UniformPolicy', env)

sampler = SimpleSampler(batch_size=hyp.B, max_path_length=50, min_pool_size=0)
sampler.initialize(env, policy, replay_pool)

for i in range(exploration_steps):
    #print("sampling")
    sampler.sample()

checkpoint_dir = os.path.join("checkpoints", hyp.name)
log_dir = os.path.join("logs_mujoco_online", hyp.name)

model = MUJOCO_ONLINE(graph=None,
                      sess=session,
                      checkpoint_dir=checkpoint_dir,
                      log_dir=log_dir
)

model.prepare_graph()

writer = model.all_writers['train']

for step in range(train_iters):

    print("Step:", step)

    for sample_step in range(sample_steps_per_iter):
        sampler.sample()

    log_this = np.mod(step, log_freq) == 0
    #log_this = False

    if log_this:
        print("Should log!")

    model.train_step(step, 'train', sampler.random_batch(), True, writer, log_this)
