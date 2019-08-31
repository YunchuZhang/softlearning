from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

import discovery.hyperparams as hyp

from .rl_algorithm import RLAlgorithm
import ipdb
st = ipdb.set_trace

def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(RLAlgorithm):
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
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            exp_name,
            plotter=None,
            tf_summaries=False,
            map3D = None,
            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,
            batch_size=None,
            save_full_state=False,
            observation_keys=None,
            **kwargs,
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

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = training_environment
        self._policy = policy

        self._Qs = Qs
        self.map3D = map3D
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr
        self.exp_name = exp_name

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
        self.batch_size = batch_size
        self._save_full_state = save_full_state

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

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_map3D()
        # st()
        # st()

        self._init_actor_update()
        self._init_critic_update()

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
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self.pix_T_cams_obs = tf.placeholder(tf.float32, [B, S, 4, 4], name='pix_T_cams')
        self.cam_T_velos_obs = tf.placeholder(tf.float32, [B, S, 4, 4], name='cam_T_velos')
        self.origin_T_camRs_obs = tf.placeholder(tf.float32, [B, S, 4, 4], name='origin_T_camRs')
        self.origin_T_camXs_obs = tf.placeholder(tf.float32, [B, S, 4, 4], name='origin_T_camXs')
        self.rgb_camX_obs = tf.placeholder(tf.float32, [B, S, H, W, 3], name='rgb_camXs')
        self.xyz_camXs_obs = tf.placeholder(tf.float32, [B, S, V, 3], name='xyz_camXs')

        self.pix_T_cams_goal = tf.placeholder(tf.float32, [B, S, 4, 4], name='pix_T_cams')
        self.cam_T_velos_goal = tf.placeholder(tf.float32, [B, S, 4, 4], name='cam_T_velos')
        self.origin_T_camRs_goal = tf.placeholder(tf.float32, [B, S, 4, 4], name='origin_T_camRs')
        self.origin_T_camXs_goal = tf.placeholder(tf.float32, [B, S, 4, 4], name='origin_T_camXs')
        self.rgb_camX_goal = tf.placeholder(tf.float32, [B, S, H, W, 3], name='rgb_camXs')
        self.xyz_camXs_goal = tf.placeholder(tf.float32, [B, S, V, 3], name='xyz_camXs')

        self.next_pix_T_cams_obs = tf.placeholder(tf.float32, [B, S, 4, 4], name='pix_T_cams')
        self.next_cam_T_velos_obs = tf.placeholder(tf.float32, [B, S, 4, 4], name='cam_T_velos')
        self.next_origin_T_camRs_obs = tf.placeholder(tf.float32, [B, S, 4, 4], name='origin_T_camRs')
        self.next_origin_T_camXs_obs = tf.placeholder(tf.float32, [B, S, 4, 4], name='origin_T_camXs')
        self.next_rgb_camX_obs = tf.placeholder(tf.float32, [B, S, H, W, 3], name='rgb_camXs')
        self.next_xyz_camXs_obs = tf.placeholder(tf.float32, [B, S, V, 3], name='xyz_camXs')

        self.next_pix_T_cams_goal = tf.placeholder(tf.float32, [B, S, 4, 4], name='pix_T_cams')
        self.next_cam_T_velos_goal = tf.placeholder(tf.float32, [B, S, 4, 4], name='cam_T_velos')
        self.next_origin_T_camRs_goal = tf.placeholder(tf.float32, [B, S, 4, 4], name='origin_T_camRs')
        self.next_origin_T_camXs_goal = tf.placeholder(tf.float32, [B, S, 4, 4], name='origin_T_camXs')
        self.next_rgb_camX_goal = tf.placeholder(tf.float32, [B, S, H, W, 3], name='rgb_camXs')
        self.next_xyz_camXs_goal = tf.placeholder(tf.float32, [B, S, V, 3], name='xyz_camXs')

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

    def _get_Q_target(self):
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

    def _init_map3D(self):
        with tf.compat.v1.variable_scope("memory", reuse=False):
            memory = self.map3D.infer_from_tensors(
                                                   tf.constant(0),
                                                   self.rgb_camXs_obs,
                                                   self.pix_T_cams_obs,
                                                   self.cam_T_velos_obs,
                                                   self.origin_T_camRs_obs,
                                                   self.origin_T_camXs_obs,
                                                   self.xyz_camXs_obs
                                                  )
        with tf.compat.v1.variable_scope("memory", reuse=True):
            memory_goal = self.map3D.infer_from_tensors(
                                                        tf.constant(0),
                                                        self.rgb_camXs_goal,
                                                        self.pix_T_cams_goal,
                                                        self.cam_T_velos_goal,
                                                        self.origin_T_camRs_goal,
                                                        self.origin_T_camXs_goal,
                                                        self.xyz_camXs_goal
                                                       )
        self.memory = [tf.concat([memory,memory_goal],-1)]

        with tf.compat.v1.variable_scope("memory", reuse=True):
            memory_next = self.map3D.infer_from_tensors(
                                                        tf.constant(0),
                                                        self.next_rgb_camXs_obs,
                                                        self.next_pix_T_cams_obs,
                                                        self.next_cam_T_velos_obs,
                                                        self.next_origin_T_camRs_obs,
                                                        self.next_origin_T_camXs_obs,
                                                        self.next_xyz_camXs_obs
                                                       )

        with tf.compat.v1.variable_scope("memory", reuse=True):
            memory_next_goal = self.map3D.infer_from_tensors(
                                                             tf.constant(0),
                                                             self.next_rgb_camXs_goal,
                                                             self.next_pix_T_cams_goal,
                                                             self.next_cam_T_velos_goal,
                                                             self.next_origin_T_camRs_goal,
                                                             self.next_origin_T_camXs_goal,
                                                             self.next_xyz_camXs_goal
                                                            )

        self.memory_next = [tf.concat([memory_next, memory_next_goal],-1)]



        # # # st()

        

        # st()

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [self.batch_size, 1]

        Q_values = self._Q_values = tuple(
            Q([*self.memory, self._actions_ph])
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

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            #print("param shapes")
            #for param in source_params:
            #    print(param.shape)
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        # st()
        # self.map3D.set_batch_size(self.batch_size)
        self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

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
        
        # feed_dict.update({
        #     self._sampler_observations_phs[i]: batch['next_observations.{}'.format(key)][:1]
        #     for i, key in enumerate(self._observation_keys)
        # })
        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

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

        (Q_values, Q_losses, alpha, global_step,memory) = self._session.run(
            (self._Q_values,
             self._Q_losses,
             self._alpha,
             self.global_step,self.memory),
            feed_dict)

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha,
        })
        # st()
        policy_diagnostics = self._policy.get_diagnostics(
            memory)
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
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
