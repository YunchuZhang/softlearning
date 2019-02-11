import numpy as np

from softlearning.replay_pools.rollout_builder import RolloutBuilder
from .simple_sampler import SimpleSampler


class HerSampler(SimpleSampler):
    def __init__(self, **kwargs):
        super(HerSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._infos = []
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self._rollout_builder = RolloutBuilder()

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action = self.policy.actions_np([
            self.env.convert_to_active_observation(
                self._current_observation)[None]
        ])[0]

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._infos.append(info)
        self._total_samples += 1

        self._rollout_builder.add_all(
            observations=self._current_observation,
            actions=action,
            rewards=reward,
            terminals=terminal,
            next_observations=next_observation
        )

        if terminal or self._path_length >= self._max_path_length:

            rollout = self._rollout_builder.get_all_stacked()
            self.pool.add_samples(
                len(rollout),
                observations=rollout['observations'],
                actions=rollout['actions'],
                rewards=rollout['rewards'],
                terminals=rollout['terminals'],
                next_observations=rollout['next_observations']
            )

            self._rollout_builder = RolloutBuilder()

            # TODO: This might be more efficient using the rollout
            last_path = self.pool.last_n_batch(
                self._path_length,
                observation_keys=getattr(self.env, 'observation_keys', None))
            last_path.update({'infos': self._infos})
            self._last_n_paths.appendleft(last_path)

            self.policy.reset()
            self._current_observation = self.env.reset()

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_length = 0
            self._path_return = 0
            self._infos = []

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return self._current_observation, reward, terminal, info
