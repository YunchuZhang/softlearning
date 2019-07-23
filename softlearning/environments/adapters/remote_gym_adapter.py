"""Implements a GymAdapter that converts Gym envs into SoftlearningEnv."""

import time
import numpy as np
import gym
from gym import spaces, wrappers

from .softlearning_env import SoftlearningEnv
from softlearning.environments.gym import register_environments
from softlearning.environments.gym.wrappers import RemoteGymEnv
from collections import defaultdict

import ray
from ray.rllib.utils.memory import ray_get_and_free


#def parse_domain_task(gym_id):
#    domain_task_parts = gym_id.split('-')
#    domain = '-'.join(domain_task_parts[:1])
#    task = '-'.join(domain_task_parts[1:])
#
#    return domain, task
#
#
#CUSTOM_GYM_ENVIRONMENT_IDS = register_environments()
#CUSTOM_GYM_ENVIRONMENTS = defaultdict(list)
#
#for gym_id in CUSTOM_GYM_ENVIRONMENT_IDS:
#    domain, task = parse_domain_task(gym_id)
#    CUSTOM_GYM_ENVIRONMENTS[domain].append(task)
#
#CUSTOM_GYM_ENVIRONMENTS = dict(CUSTOM_GYM_ENVIRONMENTS)
#
#GYM_ENVIRONMENT_IDS = tuple(gym.envs.registry.env_specs.keys())
#GYM_ENVIRONMENTS = defaultdict(list)
#
#
#for gym_id in GYM_ENVIRONMENT_IDS:
#    domain, task = parse_domain_task(gym_id)
#    GYM_ENVIRONMENTS[domain].append(task)
#
#GYM_ENVIRONMENTS = dict(GYM_ENVIRONMENTS)


class RemoteGymAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for Gym envs."""

    def __init__(self,
                 domain,
                 task,
                 *args,
                 num_agents=8,
                 normalize=True,
                 observation_keys=None,
                 unwrap_time_limit=True,
                 **kwargs):
        assert not args, (
            "Gym environments don't support args. Use kwargs instead.")

        self._num_agents = num_agents
        self.normalize = normalize
        self.unwrap_time_limit = unwrap_time_limit

        self._Serializable__initialize(locals())
        super(RemoteGymAdapter, self).__init__(domain, task, *args, **kwargs)

        assert (domain is not None and task is not None), (domain, task)
        env_id = f"{domain}-{task}"

        self._envs = []

        for _ in range(num_agents):
            self._envs.append(RemoteGymEnv.remote(env_id,
                                          normalize=normalize,
                                          env_params=kwargs))
            #time.sleep(1)

        #self._envs = [RemoteGymEnv.remote(env_id,
        #                                  normalize=normalize,
        #                                  env_params=kwargs)
        #              for _ in range(num_agents)]

        self._observation_space = ray_get_and_free(self._envs[0].observation_space.remote())
        self._action_space = ray_get_and_free(self._envs[0].action_space.remote())

        if isinstance(self._observation_space, spaces.Dict):
            self.observation_keys = (
                observation_keys or list(self._observation_space.spaces.keys()))


    @property
    def observation_space(self):
        return self._observation_space


    @property
    def active_observation_shape(self):
        #active_size = sum(
        #    np.prod(self._observation_space.spaces[key].shape)
        #    for key in self.observation_keys)

        #active_observation_shape = (active_size,)

        #return active_observation_shape

        active_observation_shape = [
            self._observation_space.spaces[key].shape
            for key in self.observation_keys
        ]

        return active_observation_shape


    def convert_to_active_observation(self, observations):

        active_observation = []
        for key in self.observation_keys:
            active_observation.append(np.concatenate([
                observation[key][None] for observation in observations
            ], axis=0))

        return active_observation


    @property
    def action_space(self, *args, **kwargs):
        return self._action_space


    def step(self, actions, *args, **kwargs):
        # TODO(hartikainen): refactor this to always return an OrderedDict,
        # such that the observations for all the envs is consistent. Right now
        # some of the gym envs return np.array whereas others return dict.
        #
        # Something like:
        # observation = OrderedDict()
        # observation['observation'] = env.step(action, *args, **kwargs)
        # return observation

        # Ray currently does not support args/kwargs
        results = ray_get_and_free([env.step.remote(action) for env, action in zip(self._envs, actions)])
        reshaped_res = list(zip(*results))

        return reshaped_res[0], np.array(reshaped_res[1]), reshaped_res[2], reshaped_res[3]

    def is_multiworld_env(self):
        return ray_get_and_free(self._envs[0].is_multiworld_env.remote())


    def compute_reward(self,
                       achieved_goal=None,
                       desired_goal=None,
                       info=None,
                       actions=None,
                       observations=None):
        # We assume that the reward computation is the same for all environments

        return ray_get_and_free(self._envs[0].compute_reward.remote(achieved_goal=achieved_goal,
                                                           desired_goal=desired_goal,
                                                           info=info,
                                                           actions=actions,
                                                           observations=observations))


    def reset(self, *args, **kwargs):
        return ray_get_and_free([env.reset.remote() for env in self._envs])


    def render(self, mode='rgb_array'):
        return ray_get_and_free(self._envs[0].render.remote(mode=mode))


    def close(self, *args, **kwargs):
        return ray_get_and_free([env.close.remote() for env in self._envs])


    def seed(self, *args, **kwargs):
        return ray_get_and_free([env.seed.remote() for env in self._envs])


    @property
    def unwrapped(self):
        raise NotImplementedError


    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError


    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError
