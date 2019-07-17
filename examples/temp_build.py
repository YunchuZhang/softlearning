import importlib
import sys
import ipdb
st = ipdb.set_trace
import copy
from softlearning.environments.utils import get_environment_from_params,get_environment_from_params_custom
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables
from examples.instrument import run_example_local

from softlearning.vae.conv_vae import ConvVAE
from softlearning.vae.vae_trainer import ConvVAETrainer
from softlearning.map3D import constants as const
from softlearning.map3D.nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain
import tensorflow as tf
import sys

sys.path.append("/Users/mihirprabhudesai/Documents/projects/rl/softlearning/softlearning/map3D")
# import utils
# sys.path.append("../")
# st()
# from examples import rig

def add_command_line_args_to_variant_spec(variant_spec, command_line_args):
    variant_spec['run_params'].update({
        'checkpoint_frequency': (
            command_line_args.checkpoint_frequency
            if command_line_args.checkpoint_frequency is not None
            else variant_spec['run_params'].get('checkpoint_frequency', 0)
        ),
        'checkpoint_at_end': (
            command_line_args.checkpoint_at_end
            if command_line_args.checkpoint_at_end is not None
            else variant_spec['run_params'].get('checkpoint_at_end', True)
        ),
    })

    variant_spec['restore'] = command_line_args.restore

    return variant_spec

op="map3d"

if op =="vae":
    example_module_name = "examples.rig"
    example_argv = ('--universe=vae', '--checkpoint-frequency=0', '--domain=SawyerReachXYEnv', '--task=v1', '--trial-gpus=1', '--exp-name=vae-test', '--replay_pool=VAEReplayPool', '--algorithm=SAC_VAE')

elif op == "map3d":
    example_module_name = "examples.map3D"
    example_argv = ('--universe=gym', '--checkpoint-frequency=0', '--domain=SawyerPushAndReachEnvEasy', '--task=v0', '--trial-gpus=1', '--exp-name=test', '--replay_pool=SimpleReplayPoolTemp', '--algorithm=SAC')




example_module = importlib.import_module(example_module_name)

example_args = example_module.get_parser().parse_args(example_argv)
# print("example_args",example_args)
variant_spec = example_module.get_variant_spec(example_args)
# st()
# print(variant_spec)
trainable_class = example_module.get_trainable_class(example_args)

variant_spec = add_command_line_args_to_variant_spec(variant_spec, example_args)
# # st()
_variant = variant_spec
variant = copy.deepcopy(_variant)\

variant["Q_params"]["kwargs"]["preprocessor_params"] = {}
variant["Q_params"]['input_shape'] = [(32,32,32,16)]

variant["policy_params"]["input_shape"] = [(32,32,32,16)]


environment_params = variant['environment_params']
env_train_params = environment_params['training']
env_train_params["kwargs"] = {}
env_train_params["kwargs"]["observation_keys"] = ["image_observation","depth_observation","cam_angles_observation","image_desired_goal","desired_goal_depth","goal_cam_angle","achieved_goal"]

batch_size = variant['sampler_params']['kwargs']['batch_size']

bulledtPush = BulletPush3DTensor4_cotrain()

training_environment = training_environment = (get_environment_from_params_custom(env_train_params))

replay_pool = replay_pool = (
    get_replay_pool_from_variant(variant, training_environment))

sampler = sampler = get_sampler_from_variant(variant)
# st()
Qs = Qs = get_Q_function_from_variant(variant, training_environment)
# st()
policy = policy = get_policy_from_variant(
    variant, training_environment, Qs)
# st()
initial_exploration_policy = initial_exploration_policy = (
    get_policy('UniformPolicy', training_environment))
# st()
_session = tf.Session

algorithm = get_algorithm_from_variant(
    variant=_variant,
    map3D =bulledtPush,
    training_environment=training_environment,
    evaluation_environment=training_environment,
    policy=policy,
    batch_size = batch_size,
    initial_exploration_policy=initial_exploration_policy,
    Qs=Qs,
    pool=replay_pool,
    observation_keys = env_train_params["kwargs"]["observation_keys"] ,
    sampler=sampler,
    session=_session)
# initialize_tf_variables(_session, only_uninitialized=True)

# _built = True
