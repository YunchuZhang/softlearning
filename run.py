import ray
import tensorflow as tf
import ipdb
st = ipdb.set_trace

from examples.map3D.variants import *
import cProfile

from softlearning.environments.utils import get_environment_from_params_custom
#from softlearning.environments.gym.flex.flex_wrappers import FetchReachMultiRobot
from softlearning.environments.adapters.gym_adapter import GymAdapter

from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant

from softlearning.samplers.utils import get_sampler_from_variant
#from softlearning.samplers.multiagent_sampler import MultiAgentSampler
from softlearning.samplers.simple_sampler import SimpleSampler

from softlearning.value_functions.utils import get_Q_function_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables


gpu_options = tf.GPUOptions(allow_growth=False)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(session)
session = tf.keras.backend.get_session()

#ray.init()

variant = get_variant_spec_3D('gym', 'SawyerReachXYEnv', 'v1', 'gaussian', 'SAC', 'SimpleSampler', 'HerReplayPool')
print(variant)
train_env = get_environment_from_params_custom(variant['environment_params']['training'])
eval_env = get_environment_from_params_custom(variant['environment_params']['training'])
#env = GymAdapter(None, None, env=FetchReachMultiRobot())
replay_pool = get_replay_pool_from_variant(variant, train_env)
sampler = get_sampler_from_variant(variant)
Qs = get_Q_function_from_variant(variant, train_env)
policy = get_policy_from_variant(variant, train_env, Qs)
initial_exploration_policy = get_policy('UniformPolicy', train_env)

#sampler = MultiAgentSampler(batch_size=1, max_path_length=2, min_pool_size=0)
#sampler = SimpleSampler(batch_size=1, max_path_length=2, min_pool_size=0)
#sampler.initialize(env, policy, replay_pool)

batch_size = variant['sampler_params']['kwargs']['batch_size']
observation_keys = variant['environment_params']['training']["kwargs"]["observation_keys"]
bulletPush = variant["map3D"]

algorithm = get_algorithm_from_variant(
    map3D=bulletPush,
    batch_size=batch_size,
    observation_keys=observation_keys,
    variant=variant,
    training_environment=train_env,
    evaluation_environment=eval_env,
    policy=policy,
    initial_exploration_policy=initial_exploration_policy,
    Qs=Qs,
    pool=replay_pool,
    sampler=sampler,
    session=session)

initialize_tf_variables(session, only_uninitialized=True)

#for _ in range(11):
#    sampler.sample()
#
#print(replay_pool.size)
#print(sampler._total_samples)
# st()
train_gen = algorithm.train()
# print(train_gen)
# cProfile.run("next(train_gen)")
# for _ in range(10):
cProfile.runctx("next(train_gen)",locals(),globals(),filename="timings")