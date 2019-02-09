import numpy as np

from .simple_replay_pool import SimpleReplayPool

# Observation space should be a dict
class HerPool(SimpleReplayPool):

    def __init__(self,
                 desired_goal_key,
                 achieved_goal_key,
                 reward_key,
                 terminal_key,
                 env,
                 *args, **kwargs):

        self._desired_goal_key = desired_goal_key
        self._achieved_goal_key = achieved_goal_key
        self._reward_key = reward_key
        self._terminal_key = terminal_key

        self.env = env

        # Fraction of samples we should resample goals for using the 'future' stragtegy
        self._fraction_future_goals = 0.25

        super(HerPool, self).__init__(*args, **kwargs)

        # For each sample keep track of the index of the last sample
        # in that episode
        self._episode_boundaries = np.zeros(self._max_size)


    def add_samples(self, num_samples, **kwargs):

        indices = np.arange(self._pointer, self._pointer + num_samples) % self._max_size
        self._episode_boundaries[indices] = indices[-1]

        return super(HerPool, self).add_samples(num_samples, **kwargs)


    def batch_by_indices(self, indices, field_name_filter=None, observation_keys=None):

        batch_size = len(indices)
        num_resamples = int(batch_size * self._fraction_future_goals)

        batch = {
            field_name: getattr(self, field_name)[indices]
            for field_name in self.field_names
        }

        if num_resamples > 0:
            desired_goals = batch[self._desired_goal_key]
            achieved_goals = batch[self._achieved_goal_key]
            rewards = batch[self._reward_key]
            terminals = batch[self._terminal_key]

            # Indices in the batch, not the buffer
            for idx in range(num_resamples):

                episode_boundary = self._episode_boundaries[idx] + 1

                # Episode crosses the end of the buffer
                # and wraps back to the beginning
                if episode_boundary < idx:
                    future_sample_offset = np.random.randint(0, self._max_size - idx + episode_boundary)
                else:
                    future_sample_offset = np.random.randint(0, episode_boundary - idx)

                future_sample_idx = (idx + future_sample_offset) % self._max_size

                future_sample = {
                    field_name: getattr(self, field_name)[future_sample_idx]
                    for field_name in self.field_names
                }

                desired_goals[idx] = future_sample[self._achieved_goal_key][0]

                rewards[idx] = self.env.compute_reward(achieved_goals[idx],
                                                       desired_goals[idx],
                                                       None)
                if future_sample_offset == 0:
                    terminals[idx] = True

            batch[self._desired_goal_key] = desired_goals
            batch[self._reward_key] = rewards
            batch[self._terminal_key] = terminals

        if observation_keys is None:
            observation_keys = tuple(self._observation_space.spaces.keys())

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
