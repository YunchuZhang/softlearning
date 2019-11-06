import os
import numpy as np
from collections import defaultdict

import tensorflow as tf

from softlearning.algorithms.utils import map3D_save, map3D_load
from softlearning.environments.adapters.gym_adapter import GymAdapter
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
from softlearning.policies.utils import get_policy
from softlearning.samplers.simple_sampler import SimpleSampler

import discovery.hyperparams as hyp
from discovery.model_mujoco_online import MUJOCO_ONLINE

exploration_steps = 500
train_iters = 10000
sample_steps_per_iter = 2
do_cropping = False
map3D_scope = "memory"

log_freq = hyp.log_freqs['train']

observation_keys = ["image_observation",
                    "depth_observation",
                    "cam_info_observation",
                    "state_observation",
                    "state_desired_goal",
                    "image_desired_goal",
                    "depth_desired_goal",
                    "cam_info_goal"]
if do_cropping:
    observation_keys.extend(["full_state_observation", "object_size"])

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(session)
session = tf.keras.backend.get_session()

env = GymAdapter('SawyerMulticameraPushRandomObjects',
                 'v0',
                 num_cameras=2,
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

with tf.compat.v1.variable_scope(map3D_scope):

    model = MUJOCO_ONLINE(graph=None,
                          sess=session,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          do_cropping = do_cropping
    )

    model.prepare_graph()

memory_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=map3D_scope)

key_vars = defaultdict(list)
for x in memory_vars:
    key_vars[x.name.split("/")[1]].append(x)

map3D_saver = {}
for k, v in key_vars.items():
    map3D_saver[k] = tf.train.Saver(var_list=v, max_to_keep=None, restore_sequentially=True)

writer = model.all_writers['train']

for step in range(train_iters):

    #print("Step:", step)

    for sample_step in range(sample_steps_per_iter):
        sampler.sample()

    log_this = np.mod(step, log_freq) == 0
    #log_this = False

    if log_this:
        map3D_save(session, "/home/ychandar/tempsave", map3D_saver, step // log_freq)

    model.train_step(step, 'train', sampler.random_batch(), True, writer, log_this, do_cropping=do_cropping)
