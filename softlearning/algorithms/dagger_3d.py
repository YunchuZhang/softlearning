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


### 3D Mapping libraries
from softlearning.map3D import constants as const
from softlearning.map3D.nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain
from softlearning.map3D.fig import Config
from softlearning.map3D import utils_map as utils

#TODO(Rakesh):
#1. Create an environment wrapper to get both state information and image information. (DONE)
#2. Load and test the map3D model. Then modify corresponding function.
#3. Change the feed_dict for getting expert actions. (DONE)
#4. Modify the replay buffer to incorporate all information. (DONE)
#5. Include a placeholder for expert observations. (DONE)
#6. Change _clean_path for getting correct observations. (DONE)
#7. Problem while computing rollouts. Where to compute the memory?
#8. Create new policy class for 3D tensor policy (incorporate map3D into the new policy).
#9. Create a new rollout function for 3D mapping.

XY_OBSERVATION_KEYS = ["observation_with_orientation", "desired_goal", 
                       "achieved_goal", "state_observation", 
                       "state_desired_goal", "state_achieved_goal",
                       "proprio_observation", "proprio_desired_goal", "proprio_achieved_goal"]

class ReplayBuffer:

    def __init__(self, env, max_size=10000):
        image_obs_keys = [key for key in env.observation_space.spaces.keys() if key not in XY_OBSERVATION_KEYS]
        obs_dim = sum([env.observation_space.spaces[key].shape[0] for key in XY_OBSERVATION_KEYS])
        image_dim = [1] + list(env.observation_space.spaces['image_observation'].shape)
        self.buffer = {'observations': np.zeros([1, obs_dim]),
                       'image_observation': np.zeros(image_dim),
                       'image_desired_goal': np.zeros(image_dim),
                       'image_achieved_goal': np.zeros(image_dim),
                       'depth_observation': np.zeros(image_dim[:-1]),
                       'desired_goal_depth': np.zeros(image_dim[:-1]),
                       'cam_angles_observation': np.zeros([1, image_dim[1], 2]),
                       'goal_cam_angle': np.zeros([1, image_dim[1], 2]),
                       'actions': np.zeros([1, env.action_space.shape[0]])}
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
                       'image_observation': np.concatenate([self.buffer['image_observation'][start_index:], paths['image_observation']]),
                       'image_desired_goal': np.concatenate([self.buffer['image_desired_goal'][start_index:], paths['image_desired_goal']]),
                       'depth_observation': np.concatenate([self.buffer['depth_observation'][start_index:], paths['depth_observation']]),
                       'desired_goal_depth': np.concatenate([self.buffer['desired_goal_depth'][start_index:], paths['desired_goal_depth']]),
                       'cam_angles_observation': np.concatenate([self.buffer['cam_angles_observation'][start_index:], paths['cam_angles_observation']]),
                       'goal_cam_angle': np.concatenate([self.buffer['goal_cam_angle'][start_index:], paths['goal_cam_angle']]),
                       'actions': np.concatenate([self.buffer['actions'][start_index:], paths['actions']])}

    def random_batch(self, batch_size):
        indices = np.random.randint(1, self._size, batch_size)
        field_names = ['observations', 'actions', 'image_observation', 'image_desired_goal', 'depth_observation',
                       'desired_goal_depth', 'cam_angles_observation', 'goal_cam_angle']
        batch = { field_name: self.buffer[field_name][indices] for field_name in field_names}
        return batch





class DAGGER(RLAlgorithm):
    """DAGGER"""

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            sampler,
            expert_path,
            pool,
            plotter=None,
            tf_summaries=False,

            lr=1e-3,

            save_full_state=False,

            pretrained_map3D=True,
            stop_3D_grads=False,
            observation_keys=None,
            batch_size=None,
            session=None,
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

        super(DAGGER, self).__init__(sampler=sampler, **kwargs)

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

        self.map3D = BulletPush3DTensor4_cotrain()
        self._stop_3D_grads = stop_3D_grads

        self.batch_size = batch_size
        self._observation_keys = ( observation_keys or list(training_environment.observation_space.spaces.keys()))

        self._save_full_state = save_full_state

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape
        self._obs_dim = sum([self._expert_evaluation_environment.observation_space.spaces[key].shape[0] for key in XY_OBSERVATION_KEYS])
        self._image_keys = [key for key in training_environment.observation_space.spaces.keys() if key not in XY_OBSERVATION_KEYS]
        self.field_names = ['image_observation', 'depth_observation', 'cam_angles_observation',
                       'image_desired_goal', 'desired_goal_depth', 'goal_cam_angle']

        self._build()

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_map3D()
        self._init_actor_update()

    def train(self, *args, **kwargs):
        """Initiate training of the DAGGER instance."""
        return self._train(*args, **kwargs)

    #TODO: Check this function again after training map3D model
    def _map3D_load(self, sess, name="rl_new/1", map3D=None):
        config = Config(name)
        config.load()
        parts = map3D.weights
        for partname in config.dct:
            partscope, partpath = config.dct[partname]
            if partname not in parts:
                raise Exception("cannot load, part %s not in model" % partpath)
            partpath = "/home/adhaig/softlearning/softlearning/map3D/" + partpath
            ckpt = tf.train.get_checkpoint_state(partpath)
            if not ckpt:
                raise Exception("checkpoint not found? (1)")
            elif not ckpt.model_checkpoint_path:
                raise Exception("checkpoint not found? (2)")
            loadpath = ckpt.model_checkpoint_path

            scope, weights = parts[partname]

            if not weights: #nothing to do
                continue
            
            weights = {utils.utils.exchange_scope(weight.op.name, scope, partscope): weight
                       for weight in weights}

            saver = tf.train.Saver(weights)
            saver.restore(sess, loadpath)
            print(f"restore model from {loadpath}")
        return config.step

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

        self._expert_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._obs_dim),
            name='observation',
        ) 

        self._observations_phs = [tf.placeholder(
            tf.float32,
            shape=(None, *self._training_environment.observation_space.spaces[key].shape),
            name='{}'.format(key)
        ) for key in self.field_names]

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

        actions = self._policy.actions(self.memory)

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

    def _init_map3D(self):

        import ipdb; ipdb.set_trace()
        obs_images, obs_zmap, obs_camAngle, obs_images_goal, obs_zmap_goal, obs_camAngle_goal = [tf.expand_dims(i, 1) for i in self._observations_phs]
        obs_zmap = tf.expand_dims(obs_zmap, -1)
        obs_zmap_goal = tf.expand_dims(obs_zmap_goal, -1)

        memory = self.map3D(obs_images, obs_camAngle, obs_zmap, is_training=None, reuse=False)
        print("MEMORY SHAPE:", memory.get_shape())
        memory_goal = self.map3D(obs_images_goal, obs_camAngle_goal ,obs_zmap_goal, is_training=None, reuse=True)
        

        if self._stop_3D_grads:
            print("Stopping 3D gradients")
            memory = tf.stop_gradient(memory)
            memory_goal = tf.stop_gradient(memory_goal)

        self.memory = [tf.concat([memory,memory_goal],-1)]
        print("MEMORY + GOAL SHAPE:", self.memory[0].get_shape())

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

        for i in range(self._replay_buffer._size // self.batch_size):
            self._do_training(
                iteration=timestep,
                batch=self._replay_buffer.random_batch(self.batch_size))

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
                expert_policy, evaluation_environment)
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

            evaluation_paths = self._evaluation_3D_paths(
                policy, evaluation_environment)
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
        for path in paths:
            path['observations'] = np.concatenate([
                path['observations.{}'.format(key)]
                for key in XY_OBSERVATION_KEYS
            ], axis=-1)
        clean_paths = {'observations': np.concatenate([path['observations'] for path in paths]),
                       'image_observation': np.concatenate([path['observations.image_observation'] for path in paths]),
                       'image_desired_goal': np.concatenate([path['observations.image_desired_goal'] for path in paths]),
                       'image_achieved_goal': np.concatenate([path['observations.image_achieved_goal'] for path in paths]),
                       'depth_observation': np.concatenate([path['observations.depth_observation'] for path in paths]),
                       'desired_goal_depth': np.concatenate([path['observations.desired_goal_depth'] for path in paths]),
                       'cam_angles_observation': np.concatenate([path['observations.cam_angles_observation'] for path in paths]),
                       'goal_cam_angle': np.concatenate([path['observations.goal_cam_angle'] for path in paths]),
                       'actions': np.concatenate([path['actions'] for path in paths])}
        return clean_paths

    def _evaluation_3D_paths(self, policy, evaluation_env):
        if self._eval_n_episodes < 1: return ()

        with policy.set_deterministic(self._eval_deterministic):
            paths = [rollout_3D(
                evaluation_env,
                policy,
                self.sampler._max_path_length,
                render_mode=self._eval_render_mode) for _ in self._eval_n_episodes]

        return paths

    def rollout_3D(env,
            policy,
            path_length,
            callback=None,
            render_mode=None,
            break_on_terminal=True):

        pool = replay_pools.SimpleReplayPool(env, max_size=path_length)
        sampler = simple_3D_sampler.SimpleSampler(
            max_path_length=path_length,
            min_pool_size=None,
            batch_size=None,
            obs_ph=self.obs_ph,
            memory=self.memory)

        sampler.initialize(env,
            policy,
            self.memory,
            self.obs_ph,
            pool)

        images = []
        infos = []

        t = 0
        for t in range(path_length):
            observation, reward, terminal, info = sampler.sample()
            infos.append(info)

            if callback is not None:
                callback(observation)

            if render_mode is not None:
                if render_mode == 'rgb_array':
                    image = env.render(mode=render_mode)
                    images.append(image)
                else:
                    env.render()

            if terminal:
                policy.reset()
                if break_on_terminal: break

        assert pool._size == t + 1

        path = pool.batch_by_indices(
            np.arange(pool._size),
            observation_keys=getattr(env, 'observation_keys', None))
        path['infos'] = infos

        if render_mode == 'rgb_array':
            path['images'] = np.stack(images, axis=0)

        return path

    def _get_expert_actions(self, paths):
        """ Getting expert actions for states visited by the agent."""

        actions = self._expert_policy.actions([self._expert_observations_ph])
        feed_dict = {
             self._expert_observations_ph: paths['observations'],
             self._actions_ph: paths['actions'],
        }
        act = self._session.run(actions, feed_dict)
        assert act.shape == paths['actions'].shape
        paths['actions'] = act
        return paths

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._actions_ph: batch['actions'],
        }

        feed_dict.update({
            self._observations_ph[i]: batch['{}'.format(key)]
            for i, key in enumerate(self._image_keys)
        })

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
