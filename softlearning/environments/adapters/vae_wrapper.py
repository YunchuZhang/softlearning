import copy
import random
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

import gym
from gym.spaces import Box, Dict

from .softlearning_env import SoftlearningEnv
from softlearning.environments.gym.wrappers import NormalizeActionWrapper

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.image_env import ImageEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

import pdb

class VAEWrappedEnv(SoftlearningEnv, MultitaskEnv):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(
            self,
            domain,
            task,
            *args,
            wrapped_env=None,
            vae=None,
            observation_keys=None,
            vae_input_key_prefix='image',
            sample_from_true_prior=False,
            decode_goals=False,
            render_goals=False,
            render_rollouts=False,
            reward_params=None,
            goal_sampling_mode="vae_prior",
            imsize=84,
            obs_size=None,
            norm_order=2,
            epsilon=20,
            presampled_goals=None,
            normalize=True,
            init_camera=sawyer_init_camera_zoomed_in,
            **kwargs
    ):
        if reward_params is None:
            reward_params = dict()
        #super().__init__(wrapped_env)

        self.normalize = normalize

        if wrapped_env is None:
            assert (domain is not None and task is not None), (domain, task)
            env_id = f"{domain}-{task}"
            wrapped_env = gym.envs.make(env_id, **kwargs)
        else:
            assert domain is None and task is None, (domain, task)

        #if normalize:
        #    wrapped_env = NormalizeActionWrapper(wrapped_env)

        # Currently only works for multiworld environments
        if not isinstance(wrapped_env, ImageEnv):
            # Normalize here is different from normalize above
            wrapped_env = ImageEnv(wrapped_env,
                                   imsize=imsize,
                                   presampled_goals=presampled_goals,
                                   normalize=True,
                                   transpose=True,
                                   init_camera=init_camera
                                   )

        self._wrapped_env = wrapped_env
        
        self.vae = vae
        self.vae_input_ph = tf.placeholder(tf.float32, [None, vae.imlength])
        self.encoder_mu_notrain, self.encoder_var_notrain = self.vae.encode(self.vae_input_ph, training=False)

        self.recons_test, _, _ = self.vae(self.vae_input_ph, training=False)

        self.representation_size = self.vae.representation_size
        self.input_channels = self.vae.input_channels
        self.sample_from_true_prior = sample_from_true_prior
        self._decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.default_kwargs=dict(
            decode_goals=decode_goals,
            render_goals=render_goals,
            render_rollouts=render_rollouts,
        )
        if render_rollouts:
            self.env_plt = None
            self.env_recon_plt = None
            self.init_plt = None
            self.init_recon_plt = None

        self.imsize = imsize
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.norm_order = self.reward_params.get("norm_order", norm_order)
        self.epsilon = self.reward_params.get("epsilon", epsilon)
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        latent_space = Box(
            -10 * np.ones(obs_size or self.representation_size),
            10 * np.ones(obs_size or self.representation_size),
            dtype=np.float32,
        )
        spaces = self._wrapped_env.observation_space.spaces
        spaces['observation'] = latent_space
        spaces['desired_goal'] = latent_space
        spaces['achieved_goal'] = latent_space
        spaces['latent_observation'] = latent_space
        spaces['latent_desired_goal'] = latent_space
        spaces['latent_achieved_goal'] = latent_space
        self._observation_space = Dict(spaces)
        self._presampled_goals = presampled_goals
        if self._presampled_goals is None:
            self.num_goals_presampled = 0
        else:
            self.num_goals_presampled = presampled_goals[random.choice(list(presampled_goals))].shape[0]

        self.observation_keys = observation_keys
        self.vae_input_key_prefix = vae_input_key_prefix
        assert vae_input_key_prefix in {'image', 'image_proprio'}
        self.vae_input_observation_key = vae_input_key_prefix + '_observation'
        self.vae_input_achieved_goal_key = vae_input_key_prefix + '_achieved_goal'
        self.vae_input_desired_goal_key = vae_input_key_prefix + '_desired_goal'
        self._mode_map = {}
        self.desired_goal = {'latent_desired_goal': latent_space.sample()}
        self._initial_obs = None
        self._custom_goal_sampler = None
        self._goal_sampling_mode = goal_sampling_mode

        self._session = tf.keras.backend.get_session()

        self.past_goal_recon = None


    @property
    def observation_space(self):
        return self._observation_space


    @property
    def active_observation_shape(self):
        """Shape for the active observation based on observation_keys."""
        observation_keys = (
            self.observation_keys
            or list(self._observation_space.spaces.keys()))

        active_size = sum(
            np.prod(self._observation_space.spaces[key].shape)
            for key in observation_keys)

        active_observation_shape = (active_size, )

        return active_observation_shape


    def convert_to_active_observation(self, observation):

        #print("observation", observation)

        #print("observation keys", self.observation_keys)

        observation_keys = (
            self.observation_keys
            or list(self._observation_space.spaces.keys()))

        observation = np.concatenate([
            observation[key] for key in observation_keys
        ], axis=-1)

        return observation


    @property
    def action_space(self):
        action_space = self._wrapped_env.action_space
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Action space ({}) is not flat, make sure to check the"
                " implemenation.".format(action_space))
        return action_space


    def reset(self, *args, **kwargs):
        obs = self._wrapped_env.reset()
        goal = self.sample_goal()
        self.set_goal(goal)
        self._initial_obs = obs
        return self._update_obs(obs)

    def step(self, action):
        #if self.render_rollouts:
            #print("taking step")
        obs, reward, done, info = self._wrapped_env.step(action)
        #print("OBSERVATION:", obs)
        new_obs = self._update_obs(obs)
        self._update_info(info, new_obs)
        #print("NEW OBSERVATION:", new_obs)
        if not self.reward_type == 'wrapped_env':
            reward = self.compute_reward(
                action,
                {'latent_achieved_goal': new_obs['latent_achieved_goal'],
                'latent_desired_goal': new_obs['latent_desired_goal']}
            )
        self.try_render(new_obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        latent_obs = self._encode_one(obs[self.vae_input_observation_key].astype(np.float32))
        #print("graph length", len([n.name for n in tf.get_default_graph().as_graph_def().node]))
        obs['latent_observation'] = latent_obs
        obs['latent_achieved_goal'] = latent_obs
        obs['observation'] = latent_obs
        obs['achieved_goal'] = latent_obs
        obs = {**obs, **self.desired_goal}
        return obs

    def _update_info(self, info, obs):
        latent_obs, logvar = self._session.run([self.encoder_mu_notrain, self.encoder_var_notrain], feed_dict={self.vae_input_ph: obs[self.vae_input_observation_key].reshape(1,-1).astype(np.float32)})
        # assert (latent_obs == obs['latent_observation']).all()
        latent_goal = self.desired_goal['latent_desired_goal']
        dist = latent_goal - latent_obs
        var = np.exp(logvar.flatten())
        var = np.maximum(var, self.reward_min_variance)
        err = dist * dist / 2 / var
        mdist = np.sum(err)  # mahalanobis distance
        info["vae_mdist"] = mdist
        info["vae_success"] = 1 if mdist < self.epsilon else 0
        info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)

    """
    Multitask functions
    """
    def sample_goals(self, batch_size):
        # TODO: make mode a parameter you pass in
        if self._goal_sampling_mode == 'custom_goal_sampler':
            return self.custom_goal_sampler(batch_size)
        elif self._goal_sampling_mode == 'presampled':
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            # ensures goals are encoded using latest vae
            if 'image_desired_goal' in sampled_goals:
                sampled_goals['latent_desired_goal'] = self._encode(sampled_goals['image_desired_goal'].astype(np.float32))
            return sampled_goals
        elif self._goal_sampling_mode == 'env':
            goals = self._wrapped_env.sample_goals(batch_size)
            latent_goals = self._encode(goals[self.vae_input_desired_goal_key].astype(np.float32))
        elif self._goal_sampling_mode == 'reset_of_env':
            assert batch_size == 1
            goal = self._wrapped_env.get_goal()
            goals = {k: v[None] for k, v in goal.items()}
            latent_goals = self._encode(
                goals[self.vae_input_desired_goal_key]
            )
        elif self._goal_sampling_mode == 'vae_prior':
            goals = {}
            latent_goals = self._sample_vae_prior(batch_size)
        else:
            raise RuntimeError("Invalid: {}".format(self._goal_sampling_mode))

        if self._decode_goals:
            decoded_goals = self._decode(latent_goals)
        else:
            decoded_goals = None
        image_goals, proprio_goals = self._image_and_proprio_from_decoded(
            decoded_goals
        )

        goals['desired_goal'] = latent_goals
        goals['latent_desired_goal'] = latent_goals
        if proprio_goals is not None:
            goals['proprio_desired_goal'] = proprio_goals
        if image_goals is not None:
            goals['image_desired_goal'] = image_goals
        if decoded_goals is not None:
            goals[self.vae_input_desired_goal_key] = decoded_goals
        return goals

    def get_goal(self):
        return self.desired_goal

    @property
    def is_multiworld_env(self):
        return True

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs):
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            return -dist
        elif self.reward_type == 'vectorized_latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'latent_sparse':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            reward = 0 if dist < self.epsilon else -1
            return reward
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            return - np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
        elif self.reward_type == 'wrapped_env':
            return self._wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    @property
    def goal_dim(self):
        return self.representation_size

    def set_goal(self, goal):
        """
        Assume goal contains both image_desired_goal and any goals required for wrapped envs

        :param goal:
        :return:
        """
        self.desired_goal = goal
        # TODO: fix this hack / document this
        if self._goal_sampling_mode in {'presampled', 'env'}:
            self._wrapped_env.set_goal(goal)


    def get_diagnostics(self, paths, **kwargs):
        statistics = self._wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["vae_mdist", "vae_success", "vae_dist"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics


    """
    Other functions
    """
    @property
    def goal_sampling_mode(self):
        return self._goal_sampling_mode

    @goal_sampling_mode.setter
    def goal_sampling_mode(self, mode):
        assert mode in [
            'custom_goal_sampler',
            'presampled',
            'vae_prior',
            'env',
            'reset_of_env'
        ], "Invalid env mode"
        self._goal_sampling_mode = mode
        if mode == 'custom_goal_sampler':
            test_goals = self.custom_goal_sampler(1)
            if test_goals is None:
                self._goal_sampling_mode = 'vae_prior'
                warnings.warn(
                    "self.goal_sampler returned None. " + \
                    "Defaulting to vae_prior goal sampling mode"
                )

    @property
    def custom_goal_sampler(self):
        return self._custom_goal_sampler

    @custom_goal_sampler.setter
    def custom_goal_sampler(self, new_custom_goal_sampler):
        assert self.custom_goal_sampler is None, (
            "Cannot override custom goal setter"
        )
        self._custom_goal_sampler = new_custom_goal_sampler

    @property
    def decode_goals(self):
        return self._decode_goals

    @decode_goals.setter
    def decode_goals(self, _decode_goals):
        self._decode_goals = _decode_goals

    def get_env_update(self):
        """
        For online-parallel. Gets updates to the environment since the last time
        the env was serialized.

        subprocess_env.update_env(**env.get_env_update())
        """
        return dict(
            mode_map=self._mode_map,
            #gpu_info=dict(
            #    use_gpu=ptu._use_gpu,
            #    gpu_id=ptu._gpu_id,
            #),
            vae_state=self.vae.__getstate__(),
        )

    def update_env(self, mode_map, vae_state): #, gpu_info):
        self._mode_map = mode_map
        self.vae.__setstate__(vae_state)
        #gpu_id = gpu_info['gpu_id']
        #use_gpu = gpu_info['use_gpu']
        #ptu.device = torch.device("cuda:" + str(gpu_id) if use_gpu else "cpu")
        #self.vae.to(ptu.device)

    def enable_render(self):
        self._decode_goals = True
        self.render_goals = True
        self.render_rollouts = True

    def disable_render(self):
        self._decode_goals = False
        self.render_goals = False
        self.render_rollouts = False

    def try_render(self, obs):
        if self.render_rollouts:
            img = obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('env', img)
            cv2.waitKey(1)
            reconstruction = self._reconstruct_img(
                obs['image_observation'].astype(np.float32)
            ).transpose()
            #print(reconstruction)
            cv2.imshow('env_reconstruction', reconstruction)
            cv2.waitKey(1)
            init_img = self._initial_obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('initial_state', init_img)
            cv2.waitKey(1)
            init_reconstruction = self._reconstruct_img(
                self._initial_obs['image_observation'].astype(np.float32)
            ).transpose()
            cv2.imshow('init_reconstruction', init_reconstruction)
            cv2.waitKey(1)

            #print("finished rollout render")

        if self.render_goals:
            goal = obs['image_desired_goal'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('goal', goal)
            cv2.waitKey(1)
            goal_reconstruction = self._reconstruct_img(
                obs['image_desired_goal'].astype(np.float32)
            ).transpose()
            #if not (goal_reconstruction == self.past_goal_recon).all():
            #    plt.figure()
            #    plt.imshow(goal_reconstruction)
            #    plt.show()

            #self.past_goal_recon = goal_reconstruction
            cv2.imshow('goal_reconstruction', goal_reconstruction)
            cv2.waitKey(1)


    def _sample_vae_prior(self, batch_size):
        if self.sample_from_true_prior:
            mu, sigma = 0, 1  # sample from prior
        else:
            mu, sigma = self.vae.dist_mu, self.vae.dist_std
        n = np.random.randn(batch_size, self.representation_size)
        return sigma * n + mu


    def _decode(self, latents):
        reconstructions, _ = self.vae.decode(latents)
        decoded = reconstructions.eval(session=self._session)
        return decoded


    def _encode_one(self, img):
        return self._encode(img[None])[0]


    def _encode(self, imgs):
        mu = self._session.run(self.encoder_mu_notrain, feed_dict={self.vae_input_ph: imgs})
        return mu


    def _reconstruct_img(self, flat_img):
        #latent_distribution_params = self.vae.encode(flat_img.reshape(1,-1))
        #reconstructions, _ = self.vae.decode(latent_distribution_params[0])
        #imgs = reconstructions.eval(session=self._session)
        #latent_obs = self._session.run(self.encoder_mu_notrain, feed_dict={self.vae_input_ph: flat_img.reshape(1,-1).astype(np.float32)})
        #print(latent_obs)
        imgs = self._session.run(self.recons_test, feed_dict={self.vae_input_ph: flat_img.reshape(1, -1)})
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]


    def _image_and_proprio_from_decoded(self, decoded):
        if decoded is None:
            return None, None
        if self.vae_input_key_prefix == 'image_proprio':
            images = decoded[:, :self.image_length]
            proprio = decoded[:, self.image_length:]
            return images, proprio
        elif self.vae_input_key_prefix == 'image':
            return decoded, None
        else:
            raise AssertionError("Bad prefix for the vae input key.")

    def __getstate__(self):
        state = super().__getstate__()
        state = copy.copy(state)
        state['_custom_goal_sampler'] = None
        warnings.warn('VAEWrapperEnv.custom_goal_sampler is not saved.')
        return state

    def __setstate__(self, state):
        warnings.warn('VAEWrapperEnv.custom_goal_sampler was not loaded.')
        super().__setstate__(state)

    @property
    def unwrapped(self):
        return self._wrapped_env.unwrapped

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._wrapped_env.close(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._wrapped_env.seed(*args, **kwargs)

    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError


def temporary_mode(env, mode, func, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    cur_mode = env.cur_mode
    env.mode(env._mode_map[mode])
    return_val = func(*args, **kwargs)
    env.mode(cur_mode)
    return return_val
