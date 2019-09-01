from collections import deque, OrderedDict
from itertools import islice


class BaseSampler(object):
    def __init__(self,
                 max_path_length,
                 min_pool_size,
                 batch_size,
                 store_last_n_paths=10,filter_keys=None):
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

    def sample(self):
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

    def forward_3D(self, active_obs):

        obs_fields = get_inputs(active_obs['image_observation'],
                                active_obs['depth_observation'],
                                active_obs['cam_angles_observation'],
                                active_obs['cam_dist_observation'])

        goal_fields = get_inputs(active_obs['image_desired_goal'],
                                 active_obs['desired_goal_depth'],
                                 active_obs['goal_cam_angles'],
                                 active_obs['goal_cam_dist'])

        memory = self.session.run(
                    self.memory3D_sampler,
                    feed_dict={
                               self.obs_ph['pix_T_cams_obs']: obs_fields['pix_T_cams'],
                               self.obs_ph['cam_T_velos_obs']: obs_fields['cam_T_velos'],
                               self.obs_ph['origin_T_camRs_obs']: obs_fields['origin_T_camRs'],
                               self.obs_ph['origin_T_camXs_obs']: obs_fields['origin_T_camXs'],
                               self.obs_ph['rgb_camXs_obs']: obs_fields['rgb_camXs'],
                               self.obs_ph['xyz_camXs_obs']: obs_fields['xyz_camXs'],
                               self.obs_ph['pix_T_cams_goal']: goal_fields['pix_T_cams'],
                               self.obs_ph['cam_T_velos_goal']: goal_fields['cam_T_velos'],
                               self.obs_ph['origin_T_camRs_goal']: goal_fields['origin_T_camRs'],
                               self.obs_ph['origin_T_camXs_goal']: goal_fields['origin_T_camXs'],
                               self.obs_ph['rgb_camXs_goal']: goal_fields['rgb_camXs'],
                               self.obs_ph['xyz_camXs_goal']: goal_fields['xyz_camXs']
                              })
        return memory
