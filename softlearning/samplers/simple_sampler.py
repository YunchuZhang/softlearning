from collections import defaultdict

import tensorflow as tf

import numpy as np
import ipdb
st = ipdb.set_trace
#from scipy.misc import imsave

from .base_sampler import BaseSampler

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
        # st()
        if self._current_observation is None:
            self._current_observation = self.env.reset()
        active_obs = self.env.convert_to_active_observation(self._current_observation)
        # st()
        active_obs = [np.repeat(i, 4, 0)  for i in active_obs]

        if self._train_writer is None and self.session is not None:
            self._train_writer = tf.summary.FileWriter('train_metadata', self.session.graph)

        if self.policy_graph is not None:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            action = self.session.run(self.policy_graph,
                                      feed_dict={ph : v
                                                 for ph, v in
                                                 #zip(self.obs_ph, active_obs)})[0]
                                                 zip(self.obs_ph, active_obs)},
                                      options=run_options,
                                      run_metadata=run_metadata)[0]
            self._train_writer.add_run_metadata(run_metadata, 'step%03d' % self._total_samples)
            #self._train_writer.add_summary(summary, self._total_samples)
            
        else:
            action = self.policy.actions_np(active_obs)[0]
            # active_obs =  np.ones([1, 32, 32, 32, 16])
            # st()

        next_observation, reward, terminal, info = self.env.step(action)
        #imsave("check_02.png",next_observation["desired_goal_depth"][0])
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
        
        # processed_sample_filter = {}
        # observation_keys = []

        # if self.filter_keys:
        #     for i in self.filter_keys:
        #         i.split(".")
        # processed_sample_filter["observations"] = {}
        # processed_sample_filter["next_observations"] = {}

        # for i in self.observation_keys:
        #     processed_sample_filter["observations"][i] = processed_sample["observations"][i]

        # processed_sample = processed_sample_filter
        # st()

        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            # st()
            self.pool.add_path(last_path)
            # st()
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            if self.policy is not None:
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
        })

        return diagnostics
