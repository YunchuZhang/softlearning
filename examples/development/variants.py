from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'Pendulum': 200,
    'SawyerPushAndReachEnvEasy': 200,
    'SawyerPushAndReachEnvMedium': 200,
    'SawyerPushAndReachEnvHard': 200,
    'SawyerPickupEnvYZEasy': 200,
    'SawyerReachXYEnv': 50,
    'FetchReach': 50,
    'FlexFetchReach': 100,
    'FlexFetchPush': 200,
    'FlexFetchReachMultiRobot': 100,
    'FlexFetchPushMultiRobot': 250,
    'SawyerPushRandomObjects': 200
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 10000,
        'train_every_n_steps': 1,
        'n_train_repeat': 3,
        'avg_weights_every_n_steps': 3,
        'eval_render_mode': None, 
        'eval_n_episodes': 10,
        'eval_deterministic': True,
        'num_agents': 4,
        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'RemoteSAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 5e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    }
}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e2),
    'Hopper': int(1e3),
    'HalfCheetah': int(3e3),
    'Walker2d': int(3e3),
    'Ant': int(3e3),
    'Humanoid': int(1e4),
    'Pusher2d': int(2e3),
    'HandManipulatePen': int(1e4),
    'HandManipulateEgg': int(1e4),
    'HandManipulateBlock': int(1e4),
    'HandReach': int(1e4),
    'Point2DEnv': int(200),
    'Reacher': int(200),
    'Pendulum': 10,
    'FetchPush': 1000,
    'FetchPickAndPlace': 1000,
    'SawyerReachXYEnv': 1000,
    'SawyerPushAndReachEnvEasy': 2000,
    'SawyerPushAndReachEnvMedium': 2000,
    'SawyerPushAndReachEnvHard': 2000,
    'SawyerPickupEnvYZEasy': 2000,
    'FlexFetchReach': 1000,
    'FlexFetchPush': 1000,
    'FlexFetchReachMultiRobot': 1000,
    'FlexFetchPushMultiRobot': 10000,
    'SawyerPushRandomObjects': 1000,
}

DEFAULT_ALGORITHM_DOMAIN_PARAMS = {
    'kwargs': {
        'n_epochs': DEFAULT_NUM_EPOCHS,
        'n_initial_exploration_steps': DEFAULT_MAX_PATH_LENGTH * 10
    }
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}

ENVIRONMENT_PARAMS = {
    'Swimmer': {  # 2 DoF
    },
    'Hopper': {  # 3 DoF
    },
    'HalfCheetah': {  # 6 DoF
    },
    'Walker2d': {  # 6 DoF
    },
    'Ant': {  # 8 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Humanoid': {  # 17 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Pusher2d': {  # 3 DoF
        'Default-v3': {
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 1.0,
            'goal': (0, -1),
        },
        'DefaultReach-v0': {
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'ImageDefault-v0': {
            'image_shape': (32, 32, 3),
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 3.0,
        },
        'ImageReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'BlindReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        }
    },
    'Point2DEnv': {
        'Default-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
        'Wall-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
    },
    'SawyerReachXYEnv': {
        'v1': {
            'reward_type': 'hand_success',
            'observation_keys': ('state_desired_goal', 'state_achieved_goal')
        }
    },
    'SawyerPushAndReachEnvEasy': {
        'v0': {
            #'reward_type': 'hand_success'
            #'reward_type': 'hand_distance'
            'reward_type': 'puck_success',
            #'fix_goal': True
        }
    },
    'SawyerPushAndReachEnvMedium': {
        'v0': {
            #'reward_type': 'hand_success'
            #'reward_type': 'hand_distance'
            'reward_type': 'puck_success'
        }
    },
    'SawyerPushAndReachEnvHard': {
        'v0': {
            #'reward_type': 'hand_success'
            #'reward_type': 'hand_distance'
            'reward_type': 'puck_success'
        }
    },
    'FlexFetchReachMultiRobot': {
        'v0': {
            'render': 1
        }
    },
    'FlexFetchPushMultiRobot': {
        'v0': {
            'render': 0
        }
    },
    'SawyerPushRandomObjects': {
        'v0': {
            'observation_keys': ('full_state_observation', 'state_desired_goal', 'state_achieved_goal'),
            #'xml_paths': ['sawyer_xyz/sawyer_push_mug1.xml']
            'xml_paths': ['sawyer_xyz/sawyer_push_mug1.xml',
                          'sawyer_xyz/sawyer_push_headphones.xml',
                          'sawyer_xyz/sawyer_push_car2.xml',
                          'sawyer_xyz/sawyer_push_mouse.xml']
            'clamp_puck_on_step': False,
            'reward_type': 'puck_success'
        }
    },
    #'SawyerPickupEnvYZEasy': {
    #    'v0': {
    #        'reward_type': 'hand_and_obj_success'
    #    }
    #},
    #'FetchReach': {
    #    'v1': {
    #        'reward_type': 'dense'
    #    }
    #}
}


NUM_CHECKPOINTS = 10


SIMPLE_SAMPLER_PARAMS = {
    'type': 'SimpleSampler',
    'kwargs': {
        'batch_size': 256,
    }
}

MULTIAGENT_SAMPLER_PARAMS = {
    'type': 'MultiAgentSampler',
    'kwargs': {
        'batch_size': 16,
        'num_agents': 8,
    }
}


SAMPLER_PARAMS_BASE = {
    'SimpleSampler': SIMPLE_SAMPLER_PARAMS,
    'MultiAgentSampler': MULTIAGENT_SAMPLER_PARAMS,
}


DEFAULT_SAMPLER_DOMAIN_PARAMS = {
    'kwargs': {
        'max_path_length': DEFAULT_MAX_PATH_LENGTH,
        'min_pool_size': DEFAULT_MAX_PATH_LENGTH
    }
}


SAMPLER_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
            }
        } for domain in MAX_PATH_LENGTH_PER_DOMAIN
    }
}


SIMPLE_REPLAY_POOL_PARAMS = {
    'type': 'SimpleReplayPool',
    'kwargs': {
        'max_size': tune.sample_from(lambda spec: (
            {
                'SimpleReplayPool': int(1e6),
                'TrajectoryReplayPool': int(1e4),
            }.get(
                spec.get('config', spec)
                ['replay_pool_params']
                ['type'],
                int(1e6))
        )),
    }
}


HER_REPLAY_POOL_PARAMS = {
    'type': 'HerReplayPool',
    'kwargs': {
        'max_size': 5e4,
        'desired_goal_key': 'desired_goal',
        'achieved_goal_key': 'achieved_goal',
        'reward_key': 'rewards',
        'terminal_key': 'terminals'
    }
}


REPLAY_POOL_PARAMS_BASE = {
    'SimpleReplayPool': SIMPLE_REPLAY_POOL_PARAMS,
    'HerReplayPool': HER_REPLAY_POOL_PARAMS
}


def get_variant_spec_base(universe, domain, task, policy, algorithm, sampler, replay_pool):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, DEFAULT_ALGORITHM_DOMAIN_PARAMS)
    )
    algorithm_params = deep_update(
        algorithm_params,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    )
    variant_spec = {
        'git_sha': get_git_rev(),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': (
                    ENVIRONMENT_PARAMS.get(domain, {}).get(task, {})),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': deep_update(
            REPLAY_POOL_PARAMS_BASE[replay_pool]
        ),
        'sampler_params': deep_update(
            SAMPLER_PARAMS_BASE[sampler],
            SAMPLER_PARAMS_PER_DOMAIN.get(domain, DEFAULT_SAMPLER_DOMAIN_PARAMS)
        ),
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           sampler,
                           replay_pool,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, sampler, replay_pool, *args, **kwargs)

    if 'image' in task.lower() or 'image' in domain.lower():
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'image_shape': (
                    variant_spec
                    ['training']
                    ['environment_params']
                    ['image_shape']),
                'output_size': M,
                'conv_filters': (4, 4),
                'conv_kernel_sizes': ((3, 3), (3, 3)),
                'pool_type': 'MaxPool2D',
                'pool_sizes': ((2, 2), (2, 2)),
                'pool_strides': (2, 2),
                'dense_hidden_layer_sizes': (),
            },
        }
        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())

    return variant_spec


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    if ('image' in task.lower()
        or 'blind' in task.lower()
        or 'image' in domain.lower()):
        variant_spec = get_variant_spec_image(
            universe, domain, task, args.policy, args.algorithm, args.sampler, args.replay_pool)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, args.policy, args.algorithm, args.sampler, args.replay_pool)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
