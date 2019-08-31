from collections import defaultdict

import numpy as np

from .base_3D_sampler import BaseSampler


class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def _process_observations(self,
                              observation,
                              action,
                              reward,
                              terminal,
                              next_observation,
                              info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        obs_req = {field_name: self.env.observation_space.spaces[field_name]
                   for field_name in self.obs_keys}
        

        if self.memory3D:
            active_observation = []
            for key in self.obs_req:
                active_observation.append(np.concatenate([
                    observation[key][None] for observation in self._current_observation
                ], axis=0))
            active_obs =[np.repeat(i, 4, 0)  for i in active_observation]
        
            active_obs = self.session.run(self.memory3D,feed_dict={self.obs_ph[0]:active_obs[0], \
                                          self.obs_ph[1]:active_obs[1],self.obs_ph[2]:active_obs[2],\
                                          self.obs_ph[3]:active_obs[3],self.obs_ph[4]:active_obs[4],\
                                          self.obs_ph[5]:active_obs[5]})
            active_obs = active_obs[:1]
        
        action = self.policy.actions_np(active_obs)[0]

        next_observation, reward, terminal, info = self.env.step(action)

        reward = reward[0]
        terminal = terminal[0]

        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            self.pool.add_path(last_path)
            self.trajectory = last_path['observations.state_observation']
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
            'current_trajectory': self.trajectory
        })

        return diagnostics
