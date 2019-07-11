from numbers import Number

import sys
import os
import copy
import math
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

import ray
from ray.experimental.tf_utils import TensorFlowVariables

from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.samplers import rollouts
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.misc.utils import initialize_tf_variables

#from .rl_agent import RLAgent

from softlearning.map3D import constants as const
from softlearning.map3D.nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain

const.set_experiment("rl_new")


def td_target(reward, discount, next_value):
    return reward + discount * next_value

@ray.remote(num_gpus=1)
class SACAgent():
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            variant,
            tf_summaries=False,
            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,
            save_full_state=False,
            remote=False,
            n_initial_exploration_steps=0,
            batch_size=None,
            map3D=None,
            observation_keys=None
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        #print("sac agent kwargs", kwargs)

        #super(SACAgent, self).__init__(
        #    variant,
        #    n_initial_exploration_steps=n_initial_exploration_steps)

        print("starting sac agent initialization")
        sys.stdout.flush()

        self.variant = variant

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids()))

        self._training_environment = get_environment_from_params(variant['environment_params']['training'])
        self._evaluation_environment = get_environment_from_params(variant['environment_params']['evaluation'])

        self._sampler = get_sampler_from_variant(variant)
        self._pool = get_replay_pool_from_variant(variant, self._training_environment)

        self._Qs = get_Q_function_from_variant(variant, self._training_environment)
        self._policy = get_policy_from_variant(variant, self._training_environment, self._Qs)

        self._initial_exploration_policy = get_policy('UniformPolicy', self._training_environment)
        self._n_initial_exploration_steps = n_initial_exploration_steps

        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in self._Qs)

        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        self._save_full_state = save_full_state

        self._remote = remote

        self.map3D = BulletPush3DTensor4_cotrain()
        self.batch_size = batch_size

        self._observation_keys = (
            observation_keys
            or list(self._training_environment.observation_space.spaces.keys()))

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        #assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        #assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self._build()

        gpu_options = tf.GPUOptions(allow_growth=True)
        config_proto = tf.ConfigProto(gpu_options=gpu_options)
        #config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, device_count={'GPU': 0})
        #off = rewriter_config_pb2.RewriterConfig.OFF
        ##config_proto.graph_options.rewrite_options.arithmetic_optimization = off
        #config_proto.graph_options.rewrite_options.memory_optimization=4
        session = tf.Session(config=config_proto)
        tf.keras.backend.set_session(session)
        self._session = tf.keras.backend.get_session()

        train_op = tf.group(*list(self._training_ops.values()))

        if self._remote:
            self.variables = TensorFlowVariables(
                train_op,
                sess=self._session
            )
            #self.variables.set_session(self._session)
        else:
            self.variables = None

        initialize_tf_variables(self._session, only_uninitialized=True)

        print("finished initialization")
        sys.stdout.flush()


    def init_sampler(self):
        self._sampler.initialize(self._training_environment,
                                self._policy,
                                self._pool,
                                memory3D=self.memory,
                                obs_ph=self._observations_phs,
                                session=self._session)


    def sampler_diagnostics(self):
        return self._sampler.get_diagnostics()

    def terminate_sampler(self):
        self._sampler.terminate()

    def total_samples(self):
        #print("getting total samples")
        return self._sampler._total_samples

    def initial_exploration(self):
        print("starting initial exploration")
        sys.stdout.flush()

        if self._n_initial_exploration_steps < 1: return

        if not self._initial_exploration_policy:
            raise ValueError(
                "Initial exploration policy must be provided when"
                " n_initial_exploration_steps > 0.")

        self._sampler.initialize(self._training_environment, self._initial_exploration_policy, self._pool)
        while self._pool.size < self._n_initial_exploration_steps:
            self._sampler.sample()

        print("finished initial exploration")
        sys.stdout.flush()


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
                self._evaluation_environment,
                self._policy,
                self._sampler._max_path_length,
                sampler=get_sampler_from_variant(self.variant),
                memory3D=self.memory,
                obs_ph=self._observations_phs,
                session=self._session,
                render_mode=render_mode)

            return paths

    def env_path_info(self, paths):
        return self._training_environment.get_path_infos(paths)

    def eval_env_path_info(self, paths):
        return self._evaluation_environment.get_path_infos(paths)

    def render_rollouts(self, paths):
        if hasattr(self._evaluation_environment, 'render_rollouts'):
            # TODO(hartikainen): Make this consistent such that there's no
            # need for the hasattr check.
            self._evaluation_environment.render_rollouts(paths)

    @property
    def tf_saveables(self):
        return {}


    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_map3D()
        self._init_actor_update()
        self._init_critic_update()


    def _init_global_step(self):
        from tensorflow.python.training import training_util

        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        import tensorflow as tf

        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_phs = [tf.placeholder(
            tf.float32,
            shape=(self.batch_size, *shape),
            name='observations.{}'.format(key)
        ) for key, shape in zip(self._observation_keys,
                                self._observation_shape)]

        self._next_observations_phs = [tf.placeholder(
            tf.float32,
            shape=(self.batch_size, *shape),
            name='next_observations.{}'.format(key)
        ) for key, shape in zip(self._observation_keys,
                                self._observation_shape)]

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, *self._action_shape),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, *self._action_shape),
                name='raw_actions',
            )


    def _init_map3D(self):
        # self.map3D.set_batchSize(1)
        
        # sampler_obs_images, sampler_obs_zmap, sampler_obs_camAngle,sampler_obs_images_goal, sampler_obs_zmap_goal, sampler_obs_camAngle_goal = [tf.expand_dims(i,1) for i in self._sampler_observations_phs[:6]]

        # sampler_obs_zmap = tf.expand_dims(sampler_obs_zmap,-1)
        # sampler_obs_zmap_goal = tf.expand_dims(sampler_obs_zmap_goal,-1)

        # memory_sampler = self.map3D(sampler_obs_images,sampler_obs_camAngle,sampler_obs_zmap, is_training=None,reuse=False)
        # memory_sampler_goal = self.map3D(sampler_obs_images_goal,sampler_obs_camAngle_goal,sampler_obs_zmap_goal, is_training=None,reuse=True)
        # self.memory_sampler = [tf.concat([memory_sampler,memory_sampler_goal],-1)]
        # # self.memory_sampler = 1
        # self.map3D.set_batchSize(4)

        obs_images, obs_zmap, obs_camAngle, obs_images_goal, obs_zmap_goal, obs_camAngle_goal = [tf.expand_dims(i,1) for i in self._observations_phs[:6]]
        obs_zmap = tf.expand_dims(obs_zmap,-1)
        obs_zmap_goal = tf.expand_dims(obs_zmap_goal,-1)

        # st()
        memory = self.map3D(obs_images, obs_camAngle, obs_zmap, is_training=None, reuse=False)
        memory_goal = self.map3D(obs_images_goal, obs_camAngle_goal ,obs_zmap_goal, is_training=None, reuse=True)
        self.memory = [tf.concat([memory,memory_goal],-1)]

        next_obs_images, next_obs_zmap, next_obs_camAngle, next_obs_images_goal, next_obs_zmap_goal, next_obs_camAngle_goal = [tf.expand_dims(i,1) for i in self._next_observations_phs[:6]]

        next_obs_zmap = tf.expand_dims(next_obs_zmap,-1)
        next_obs_zmap_goal = tf.expand_dims(next_obs_zmap_goal,-1)

        memory_next = self.map3D(next_obs_images,next_obs_camAngle,next_obs_zmap, is_training=None,reuse=True)
        memory_next_goal = self.map3D(next_obs_images_goal,next_obs_camAngle_goal,next_obs_zmap_goal, is_training=None,reuse=True)
        self.memory_next = [tf.concat([memory_next,memory_next_goal],-1)]


    def _get_Q_target(self):
        import tensorflow as tf

        next_actions = self._policy.actions(self.memory_next)
        next_log_pis = self._policy.log_pis(
            self.memory_next, next_actions)

        next_Qs_values = tuple(
            Q([*self.memory_next, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        import tensorflow as tf

        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [self.batch_size, 1]

        Q_values = self._Q_values = tuple(
            Q([*self.memory_next, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))
        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                    "loss", "gradients", "gradient_norm", "global_gradient_norm"
                ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        import tensorflow as tf

        actions = self._policy.actions(self.memory)
        log_pis = self._policy.log_pis(self.memory, actions)

        assert log_pis.shape.as_list() == [self.batch_size, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([*self.memory, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [self.batch_size, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})

    def init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def do_training(self, iteration, steps=1, weights=None):
        """Runs the operations for updating training and target ops."""
        if self._remote:
            self.variables.set_weights(weights)

        for i in range(iteration, iteration + steps):

            if i % self._target_update_interval == 0:
                # Run target ops here.
                self._update_target()

            batch = self.training_batch()

            feed_dict = self._get_feed_dict(i, batch)

            self._session.run(self._training_ops, feed_dict)

        if self._remote:
            return self.variables.get_weights()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._actions_ph: batch['actions'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        feed_dict.update({
            self._observations_phs[i]: batch['observations.{}'.format(key)]
            for i, key in enumerate(self._observation_keys)
        })

        feed_dict.update({
            self._next_observations_phs[i]: batch['next_observations.{}'.format(key)]
            for i, key in enumerate(self._observation_keys)
        })

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict


    def get_diagnostics(self, iteration, batch):
        feed_dict = self._get_feed_dict(iteration, batch)
        (Q_values, Q_losses, alpha, global_step, memory) = self._session.run(
            (self._Q_values,
             self._Q_losses,
             self._alpha,
             self.global_step,
             self.memory),
            feed_dict)

        policy_diagnostics = self._policy.get_diagnostics(memory)
        return Q_values, Q_losses, alpha, global_step, policy_diagnostics


    def policy_diagnostics(self, batch):
        return self._policy.get_diagnostics(batch)


    def get_weights(self):
        return self.variables.get_weights()


    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
