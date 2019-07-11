import tensorflow as tf
import math

from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant

from softlearning.samplers import rollouts

import copy

class RLAgent():

    def __init__(
            self,
            variant,
            n_initial_exploration_steps=0,
    ):

        self._env = get_environment_from_params(variant['environment_params']['training'])
        self._eval_env = get_environment_from_params(variant['environment_params']['evaluation'])

        self._sampler = get_sampler_from_variant(variant)
        self._pool = get_replay_pool_from_variant(variant, self._env)

        self._Qs = get_Q_function_from_variant(variant, self._env)
        self._policy = get_policy_from_variant(variant, self._env, self._Qs)

        self._initial_exploration_policy = get_policy('UniformPolicy', self._env)
        self._n_initial_exploration_steps = n_initial_exploration_steps


    def init_sampler(self):
        self._sampler.initialize(self._env, self._policy, self._pool)

    def sampler_diagnostics(self):
        return self._sampler.get_diagnostics()

    def terminate_sampler(self):
        self._sampler.terminate()

    def total_samples(self):
        #print("getting total samples")
        return self._sampler._total_samples

    def initial_exploration(self):
        if self._n_initial_exploration_steps < 1: return

        if not self._initial_exploration_policy:
            raise ValueError(
                "Initial exploration policy must be provided when"
                " n_initial_exploration_steps > 0.")

        self._sampler.initialize(self._env, self._initial_exploration_policy, self._pool)
        while self._pool.size < self._n_initial_exploration_steps:
            self._sampler.sample()

    def do_sampling(self, timestep, steps):
        for _ in range(steps):
            self._sampler.sample()

    def ready_to_train(self):
        return self._sampler.batch_ready()

    def training_batch(self, batch_size=None):
        return self._sampler.random_batch(batch_size)
    
    def training_paths(self, epoch_length):
        return self._sampler.get_last_n_paths(
            math.ceil(epoch_length / self._sampler._max_path_length))
    
    def evaluation_paths(self, num_episodes, eval_deterministic, render_mode):

        with self._policy.set_deterministic(eval_deterministic):
            paths = rollouts(
                num_episodes,
                self._eval_env,
                self._policy,
                self._sampler._max_path_length,
                sampler=copy.deepcopy(self._sampler),
                render_mode=render_mode)

            return paths


    def env_path_info(self, paths):
        return self._env.get_path_infos(paths)

    def eval_env_path_info(self, paths):
        return self._eval_env.get_path_infos(paths)

    def render_rollouts(self, paths):
        if hasattr(self._eval_env, 'render_rollouts'):
            # TODO(hartikainen): Make this consistent such that there's no
            # need for the hasattr check.
            self._eval_env.render_rollouts(paths)

    @property
    def tf_saveables(self):
        return {}

