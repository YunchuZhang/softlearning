from collections import deque, OrderedDict
from itertools import islice

from discovery.backend.mujoco_online_inputs import get_inputs
import sys


class BaseSampler(object):
    def __init__(self,
                 max_path_length,
                 min_pool_size,
                 batch_size,
                 store_last_n_paths=10,
                 filter_keys=None):

        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size
        self._store_last_n_paths = store_last_n_paths
        self._last_n_paths = deque(maxlen=store_last_n_paths)
        self.filter_keys = filter_keys
        self.initialized =False
        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool, memory3D=None, obs_ph=None, session=None):
        self.env = env
        self.policy = policy
        self.pool = pool
        self.initialized =True
        self.memory3D_sampler = memory3D
        self.obs_ph = obs_ph
        self.session = session

    def set_policy(self, policy):
        self.policy = policy

    def clear_last_n_paths(self):
        self._last_n_paths.clear()

    def get_last_n_paths(self, n=None):
        if n is None:
            n = self._store_last_n_paths

        last_n_paths = tuple(islice(self._last_n_paths, None, n))

        return last_n_paths

    def sample(self, do_cropping):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        return self.pool.random_batch(batch_size, **kwargs)

    def terminate(self):
        self.env.close()

    def get_diagnostics(self):
        diagnostics = OrderedDict({'pool-size': self.pool.size})
        return diagnostics

    def __getstate__(self):
        state = {
            key: value for key, value in self.__dict__.items()
            if key not in ('env', 'policy', 'pool')
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.env = None
        self.policy = None
        self.pool = None


    def add_discovery_fields(self, obs_dict, do_cropping):

        if do_cropping:
            obs_fields = get_inputs(obs_dict['image_observation'],
                                    obs_dict['depth_observation'],
                                    obs_dict['cam_info_observation'],
                                    obs_dict['full_state_observation'])
        else:
            obs_fields = get_inputs(obs_dict['image_observation'],
                                    obs_dict['depth_observation'],
                                    obs_dict['cam_info_observation'])

        obs_dict.update(obs_fields)


    def forward_3D(self, obs_fields, do_cropping):

        if do_cropping:

            memory = self.session.run(
                    self.memory3D_sampler,
                    feed_dict={
                               self.obs_ph['pix_T_cams_obs']: obs_fields['pix_T_cams'],
                               self.obs_ph['origin_T_camRs_obs']: obs_fields['origin_T_camRs'],
                               self.obs_ph['origin_T_camXs_obs']: obs_fields['origin_T_camXs'],
                               self.obs_ph['rgb_camXs_obs']: obs_fields['rgb_camXs'],
                               self.obs_ph['xyz_camXs_obs']: obs_fields['xyz_camXs'],

                               self.obs_ph['state_centroid']: obs_fields['full_state_observation'],
                               #self.obs_ph['pix_T_cams_goal']: goal_fields['pix_T_cams'],
                               #self.obs_ph['origin_T_camRs_goal']: goal_fields['origin_T_camRs'],
                               #self.obs_ph['origin_T_camXs_goal']: goal_fields['origin_T_camXs'],
                               #self.obs_ph['rgb_camXs_goal']: goal_fields['rgb_camXs'],
                               #self.obs_ph['xyz_camXs_goal']: goal_fields['xyz_camXs']
                               self.obs_ph['centroid_goal']: obs_fields['state_desired_goal'],
                               self.obs_ph['puck_xyz_camRs']: obs_fields['crop_center_xyz_camRs'],
                               self.obs_ph['camRs_T_puck']: obs_fields['camRs_T_crop'],
                               self.obs_ph['obj_size']: obs_fields['object_size'],
                              })
        else:

            memory = self.session.run(
                    self.memory3D_sampler,
                    feed_dict={
                               self.obs_ph['pix_T_cams_obs']: obs_fields['pix_T_cams'],
                               self.obs_ph['origin_T_camRs_obs']: obs_fields['origin_T_camRs'],
                               self.obs_ph['origin_T_camXs_obs']: obs_fields['origin_T_camXs'],
                               self.obs_ph['rgb_camXs_obs']: obs_fields['rgb_camXs'],
                               self.obs_ph['xyz_camXs_obs']: obs_fields['xyz_camXs'],

                               self.obs_ph['state_centroid']: obs_fields['full_state_observation'],
                               #self.obs_ph['pix_T_cams_goal']: goal_fields['pix_T_cams'],
                               #self.obs_ph['origin_T_camRs_goal']: goal_fields['origin_T_camRs'],
                               #self.obs_ph['origin_T_camXs_goal']: goal_fields['origin_T_camXs'],
                               #self.obs_ph['rgb_camXs_goal']: goal_fields['rgb_camXs'],
                               #self.obs_ph['xyz_camXs_goal']: goal_fields['xyz_camXs']
                               self.obs_ph['centroid_goal']: obs_fields['state_desired_goal'],
                              })

        return memory
