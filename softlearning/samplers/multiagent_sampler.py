from collections import defaultdict

import copy
import numpy as np

from .base_sampler import BaseSampler

#import pdb

class MultiAgentSampler(BaseSampler):
    def __init__(self,
                 num_agents=1,
                 **kwargs):

        super(MultiAgentSampler, self).__init__(**kwargs)

        self.num_agents = num_agents
        self._path_length = 0
        self._path_returns = None
        self._current_paths = [defaultdict(list) for _ in range(self.num_agents)]
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observations = None
        self._total_samples = 0

    def _process_observations(self,
                              index,
                              observations,
                              actions,
                              rewards,
                              terminals,
                              next_observations,
                              infos):

        # Handles dictionary of lists (i.e. multirobot Flex wrapper)
        if isinstance(observations, dict):
            observation = {field_name: values[index]
                        for field_name, values in observations.items()}
            next_observation = {field_name: values[index]
                                for field_name, values in next_observations.items()}
            info = {field_name: values[index]
                    for field_name, values in infos.items()}

        # Handles list of dictionaries (i.e. RemoteGymAdapter)
        elif isinstance(observations, (tuple, list)):
            observation = observations[index]
            next_observation = next_observations[index]
            info = infos[index]

        processed_observation = {
            'observations': observation,
            'actions': actions[index],
            'rewards': [rewards[index]],
            'terminals': [terminals[index]],
            'next_observations': next_observation,
            'infos': info,
        }

        return copy.deepcopy(processed_observation)

    def sample(self):
        #pdb.set_trace()
        if self._current_observations is None:
            self._current_observations = self.env.reset()

        #actions = self.policy.actions_np([
        #    self.env.convert_to_active_observation(
        #        self._current_observations)
        #])

        active_obs = self.env.convert_to_active_observation(self._current_observations)


        if self.initialized and self.memory3D_sampler:
            active_obs = [np.vstack([field] * int(self._batch_size / self.num_agents)) for field in active_obs]
            active_obs = self.session.run(self.memory3D_sampler,feed_dict={self.obs_ph[0]:active_obs[0],self.obs_ph[1]:active_obs[1],self.obs_ph[2]:active_obs[2],\
                self.obs_ph[3]:active_obs[3]})
            active_obs = active_obs[:self.num_agents]

        actions = self.policy.actions_np(active_obs)

        next_observations, rewards, terminals, infos = self.env.step(actions)
        self._path_length += 1
        self._total_samples += self.num_agents

        if self._path_returns is None:
            self._path_returns = np.zeros(self.num_agents)
        self._path_returns += rewards

        for i in range(self.num_agents):
            processed_sample = self._process_observations(
                i,
                observations=self._current_observations,
                actions=actions,
                rewards=rewards,
                terminals=terminals,
                next_observations=next_observations,
                infos=infos,
            )

            for key, value in processed_sample.items():
                self._current_paths[i][key].append(value)

        if np.count_nonzero(terminals) or self._path_length >= self._max_path_length:
            for i in range(self.num_agents):
                path = {
                    field_name: np.array(values)
                    for field_name, values in self._current_paths[i].items()
                }

                self.pool.add_path(path)
                self._last_n_paths.appendleft(path)

            self._max_path_return = max(self._max_path_return,
                                        np.max(self._path_returns))
            self._last_path_return = self._path_returns[-1]

            self.reset()

            self._n_episodes += 1
        else:
            self._current_observations = next_observations

        return next_observations, rewards, terminals, infos


    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(MultiAgentSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics

    def reset(self):
        self.policy.reset()
        self._current_observations = None
        self._path_length = 0
        self._path_returns = np.zeros(self.num_agents)
        self._current_paths = [defaultdict(list) for _ in range(self.num_agents)]
