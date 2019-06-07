import numpy as np

from .simple_replay_pool import SimpleReplayPool

# Observation space should be a dict
class HerReplayPool(SimpleReplayPool):

    def __init__(self,
                 env,
                 desired_goal_key,
                 achieved_goal_key,
                 reward_key,
                 terminal_key,
                 *args,
                 **kwargs):

        self._desired_goal_key = desired_goal_key
        self._achieved_goal_key = achieved_goal_key
        self._reward_key = reward_key
        self._terminal_key = terminal_key

        self.env = env

        # Fraction of samples we should resample goals for using the 'future' stragtegy
        self._fraction_future_goals = 0.8

        # Need to include both 'state' and non-'state' for image_env vs base environment
        self._reward_keys = ('state_achieved_goal', 'state_desired_goal', 'achieved_goal', 'desired_goal')

        super(HerReplayPool, self).__init__(*args, env, **kwargs)

        # For each sample keep track of the index of the last sample
        # in that episode
        self._episode_boundaries = np.zeros(self._max_size)


    def add_samples(self, samples):

        field_names = list(samples.keys())
        num_samples = samples[field_names[0]].shape[0]

        indices = np.arange(self._pointer, self._pointer + num_samples) % self._max_size
        self._episode_boundaries[indices] = indices[-1]

        return super(HerReplayPool, self).add_samples(samples)


    def batch_by_indices(self, indices, field_name_filter=None, observation_keys=None):

        batch_size = len(indices)
        num_resamples = int(batch_size * self._fraction_future_goals)

        if observation_keys is None:
            observation_keys = tuple(self._observation_space.spaces.keys())

        batch = {
            field_name: self.fields[field_name][indices]
            for field_name in self.field_names
        }

        if num_resamples > 0:
            achieved_goals = batch['next_observations.' + self._achieved_goal_key]
            rewards = batch[self._reward_key]
            terminals = batch[self._terminal_key]
            actions = batch['actions']

            # batch_idx is index of sample in batch
            # pool_idx is index of sample in replay pool
            for batch_idx in range(num_resamples):
                pool_idx = indices[batch_idx]

                episode_boundary = self._episode_boundaries[pool_idx] + 1

                # Episode crosses the end of the buffer
                # and wraps back to the beginning
                if episode_boundary < pool_idx:
                    future_sample_offset = np.random.randint(0, self._max_size - pool_idx + episode_boundary)
                else:
                    future_sample_offset = np.random.randint(0, episode_boundary - pool_idx)

                future_sample_idx = (pool_idx + future_sample_offset) % self._max_size

                future_achieved_goal = self.fields['next_observations.' + self._achieved_goal_key][future_sample_idx]

                batch['observations.' + self._desired_goal_key][batch_idx] = future_achieved_goal
                batch['next_observations.' + self._desired_goal_key][batch_idx] = future_achieved_goal

                if self.env.is_multiworld_env:
                    observation = {key: np.array([batch['next_observations.{}'.format(key)][batch_idx]])
                                   for key in self._reward_keys}
                    #print(observation)
                    rewards[batch_idx] = self.env.compute_rewards(actions=np.array([actions[batch_idx]]),
                                                                 obs=observation)
                else:
                    rewards[batch_idx] = self.env.compute_reward(achieved_goal=achieved_goals[batch_idx],
                                                                  desired_goal=future_achieved_goal,
                                                                  info=None)
                if future_sample_offset == 0:
                    terminals[batch_idx] = True

            batch[self._reward_key] = rewards
            batch[self._terminal_key] = terminals

        observations = np.concatenate([
            batch['observations.{}'.format(key)]
            for key in observation_keys
        ], axis=-1)

        next_observations = np.concatenate([
            batch['next_observations.{}'.format(key)]
            for key in observation_keys
        ], axis=-1)

        batch['observations'] = observations
        batch['next_observations'] = next_observations

        if field_name_filter is not None:
            filtered_fields = self.filter_fields(
                batch.keys(), field_name_filter)
            batch = {
                field_name: batch[field_name]
                for field_name in filtered_fields
            }

        return batch
