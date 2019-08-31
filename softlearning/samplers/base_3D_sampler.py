from collections import deque, OrderedDict
from itertools import islice


XY_OBSERVATION_KEYS = ["observation_with_orientation", "desired_goal",
                       "achieved_goal", "state_observation",
                       "state_desired_goal", "state_achieved_goal",
                       "proprio_observation", "proprio_desired_goal", "proprio_achieved_goal"]

class BaseSampler(object):
    def __init__(self,
                 max_path_length,
                 min_pool_size,
                 batch_size,
                 store_last_n_paths=10):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size
        self._store_last_n_paths = store_last_n_paths
        self._last_n_paths = deque(maxlen=store_last_n_paths)

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, memory, obs_ph, pool, expert=False):
        self.env = env
        self.policy = policy
        self.pool = pool
        self.memory = memory
        self.obs_ph = obs_ph
        self.obs_keys = [key for key in self.env.observation_space.spaces.keys() if key not in XY_OBSERVATION_KEYS]

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
