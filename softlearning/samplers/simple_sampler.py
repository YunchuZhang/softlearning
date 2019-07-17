from collections import defaultdict
import ipdb
st = ipdb.set_trace
import time
import numpy as np
# <<<<<<< Updated upstream
#from scipy.misc import imsave
import time
# =======
# from scipy.misc import imsave
# import time
# >>>>>>> Stashed changes
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

    def forward(self,active_obs):
        # st()
        active_obs = self.session.run(self.memory3D_sampler,feed_dict={self.obs_ph[0]:active_obs[0],self.obs_ph[1]:active_obs[1],self.obs_ph[2]:active_obs[2],\
            self.obs_ph[4]:active_obs[4],self.obs_ph[5]:active_obs[5],self.obs_ph[6]:active_obs[6]})        
        return active_obs
    
    def sample(self):
        import ipdb
        #ipdb.set_trace()
        if self._current_observation is None:
            self._current_observation = self.env.reset()
        active_obs = self.env.convert_to_active_observation(self._current_observation)
        # st()
        # if preprocess:
        active_obs =[np.repeat(i,4,0)  for i in active_obs]


        # if len(active_obs[0].shape) == 5 and len(active_obs[4].shape) == 5:
        #     # for case when we train 3d code
        #     active_obs[0] = np.concatenate([active_obs[0],np.ones_like(active_obs[0][...,:1])],-1)
        #     active_obs[4] = np.concatenate([active_obs[4],np.ones_like(active_obs[4][...,:1])],-1)

        if self.initialized and self.memory3D_sampler:
            a = time.time()
            active_obs = self.forward(active_obs)
            print(time.time()-a,"sampler sess")
            active_obs = active_obs[:1]
        else:
            active_obs  = [i[:1] for i in active_obs]
            # active_obs =  np.ones([1, 32, 32, 32, 16])
            # st()
        # st()

        action = self.policy.actions_np(active_obs)[0]
        

        #st()
        next_observation, reward, terminal, info = self.env.step(action)

        reward=reward[0]
        terminal =terminal[0]
        # st()


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
            #st()
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            #st()
            self.pool.add_path(last_path)
            # st()
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
        })

        return diagnostics
