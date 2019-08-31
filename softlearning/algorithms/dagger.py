from collections import OrderedDict
from itertools import count
from numbers import Number

import os
import json
import pickle
import gtimer as gt
import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from .rl_algorithm import RLAlgorithm
from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant


class ReplayBuffer:

    def __init__(self, env, max_size=100000):
        obs_dim = sum([env.observation_space.spaces[key].shape[0] for key in env.observation_space.spaces.keys()])
        self.buffer = {'observations': np.zeros([1, obs_dim]),
                       'actions': np.zeros([1, env.action_space.shape[0]]),
                       'next_observations': np.zeros([1, obs_dim]),
                       'terminals': np.zeros([1, 1])}
        self.max_size = max_size

    @property
    def _size(self):
        return self.buffer['observations'].shape[0]

    def add_path(self, paths):
        num_new_samples = paths['observations'].shape[0]
        num_samples_estimates = self._size + num_new_samples
        diff_samples = num_samples_estimates - self.max_size
        start_index = diff_samples if diff_samples > 0 else 0
        self.buffer = {'observations': np.concatenate([self.buffer['observations'][start_index:], paths['observations']]),
                       'next_observations': np.concatenate([self.buffer['next_observations'][start_index:], paths['next_observations']]),
                       'actions': np.concatenate([self.buffer['actions'][start_index:], paths['actions']]),
                       'terminals': np.concatenate([self.buffer['terminals'][start_index:], paths['terminals']])}

    def random_batch(self, batch_size):
        indices = np.random.randint(1, self._size, batch_size)
        field_names = ['observations', 'actions', 'next_observations', 'terminals']
        batch = { field_name: self.buffer[field_name][indices] for field_name in field_names}
        return batch

class DAGGER(RLAlgorithm):
    """DAGGER"""

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            expert_path,
            pool,
            plotter=None,
            tf_summaries=False,

            lr=1e-3,

            save_full_state=False,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
        """

        super(DAGGER, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy
        self._policy.set_deterministic(True)  # Dagger policy always deterministically executes
        self._n_train_repeat = 50
        self._n_epochs = 30
        self._replay_buffer = ReplayBuffer(evaluation_environment)
        
        ### RETRIEVING EXPERT POLICY
        checkpoint_path = expert_path.rstrip('/')
        experiment_path = os.path.dirname(checkpoint_path)
        variant_path = os.path.join(experiment_path, 'params.json')
        with open(variant_path, 'r') as f:
            variant = json.load(f)

        with self._session.as_default() as sess:
            pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)

        environment_params = (  variant['environment_params']['evaluation']
                                if 'evaluation' in variant['environment_params']
                                else variant['environment_params']['training'])

        self._expert_evaluation_environment = get_environment_from_params(environment_params)
        expert_policy = (
        get_policy_from_variant(variant, self._expert_evaluation_environment, Qs=[None]))
        expert_policy.set_weights(picklable['policy_weights'])
        self._expert_policy = expert_policy
        #####

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr

        self._save_full_state = save_full_state

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self._build()

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()

    def train(self, *args, **kwargs):
        """Initiate training of the SAC instance."""
        return self._train(*args, **kwargs)

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - action
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

    def _init_actor_update(self):
        """Create minimization operations for policy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        """

        actions = self._policy.actions([self._observations_ph])

        self._policy_loss = tf.losses.mean_squared_error(self._actions_ph, actions)

        assert self._policy_loss.shape.as_list() == []

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            self._policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._training_ops, feed_dict)

    def _do_training_repeats(self, timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        if timestep % self._train_every_n_steps > 0: return
        trained_enough = (
            self._train_steps_this_epoch
            > self._max_train_repeat_per_timestep * self._timestep)
        if trained_enough: return

        batch_size = 64
        for i in range(self._replay_buffer._size // batch_size):
            self._do_training(
                iteration=timestep,
                batch=self._replay_buffer.random_batch(batch_size))

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat


    def _train(self):
        """Return a generator that performs RL training.

        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy (`Policy`): Policy used for training
            pool (`PoolBase`): Sample pool to add samples to
        """
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        expert_evaluation_environment = self._expert_evaluation_environment
        policy = self._policy
        expert_policy = self._expert_policy
        pool = self._pool

        # Filling sampler's replay buffer with expert samples initially
        evaluation_paths = self._evaluation_paths(
                expert_policy, expert_evaluation_environment)
        clean_evaluation_paths = self._clean_paths(evaluation_paths)
        self._replay_buffer.add_path(clean_evaluation_paths)


        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

        self._training_before_hook()

        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            gt.stamp('timestep_before_hook')

            self._do_training_repeats(timestep=self._total_timestep)
            gt.stamp('train')

            gt.stamp('timestep_after_hook')

            evaluation_paths = self._evaluation_paths(
                policy, expert_evaluation_environment)
            gt.stamp('evaluation_paths')

            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, expert_evaluation_environment)
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}

            # ADD EVALUATING EXPERT ACTIONS HERE
            # 1. Get format of paths in evaluation rollouts.
            clean_evaluation_paths = self._clean_paths(evaluation_paths)
            # 2. Forward pass all samples through expert policy.
            clean_evaluation_paths = self._get_expert_actions(clean_evaluation_paths)
            # 3. Add samples to replay buffer
            self._replay_buffer.add_path(clean_evaluation_paths)

            gt.stamp('epoch_after_hook')

            diagnostics = {}
            time_diagnostics = gt.get_times().stamps.itrs

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ), 
                ('epoch', self._epoch),
                ('train-steps', self._num_train_steps),
            )))

            yield diagnostics

        yield {'done': True, **diagnostics}


    def _clean_paths(self, paths):
        """Cleaning up paths to only contain relevant information like
           observation, next_observation, action, reward, terminal.
        """

        clean_paths = {'observations': np.concatenate([path['observations'] for path in paths]),
                       'next_observations': np.concatenate([path['next_observations'] for path in paths]),
                       'actions': np.concatenate([path['actions'] for path in paths]),
                       'rewards': np.concatenate([path['rewards'] for path in paths]),
                       'terminals': np.concatenate([path['terminals'] for path in paths])}

        return clean_paths

    def _get_expert_actions(self, paths):
        """ Getting expert actions for sttaes visited by the agent."""

        actions = self._expert_policy.actions([self._observations_ph])
        feed_dict = self._get_feed_dict(None, paths)
        act = self._session.run(actions, feed_dict)
        assert act.shape == paths['actions'].shape
        paths['actions'] = act
        return paths

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
        }

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        policy_diagnostics = self._policy.get_diagnostics(
            batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
        }

        return saveables
