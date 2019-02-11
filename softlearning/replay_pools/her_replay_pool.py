import numpy as np

from .simple_replay_pool import SimpleReplayPool

# Observation space should be a dict
class HerReplayPool(SimpleReplayPool):

    def __init__(self, env, *args, **kwargs):

        self._desired_goal_key = kwargs['desired_goal_key']
        self._achieved_goal_key = kwargs['achieved_goal_key']
        self._reward_key = kwargs['reward_key']
        self._terminal_key = kwargs['terminal_key']

        self.env = env

        # Fraction of samples we should resample goals for using the 'future' stragtegy
        self._fraction_future_goals = 0.25

        super(HerReplayPool, self).__init__(*args, env, **kwargs)

        # For each sample keep track of the index of the last sample
        # in that episode
        self._episode_boundaries = np.zeros(self._max_size)


    def add_samples(self, samples):

        indices = np.arange(self._pointer, self._pointer + len(samples)) % self._max_size
        self._episode_boundaries[indices] = indices[-1]

        return super(HerReplayPool, self).add_samples(samples)


    def batch_by_indices(self, indices, field_name_filter=None, observation_keys=None):

        batch_size = len(indices)
        num_resamples = int(batch_size * self._fraction_future_goals)

        batch = {
            field_name: self.fields[field_name][indices]
            for field_name in self.field_names
        }

        if num_resamples > 0:
            desired_goals = batch['observations.' + self._desired_goal_key]
            next_desired_goals = batch['next_observations.' + self._desired_goal_key]
            achieved_goals = batch['next_observations.' + self._achieved_goal_key]
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

                future_achieved_goal = self.fields['next_observations.' + self._achieved_goal_key][future_sample_idx]

                desired_goals[idx] = future_achieved_goal
                next_desired_goals[idx] = future_achieved_goal

                rewards[idx] = self.env.compute_reward(achieved_goals[idx],
                                                       desired_goals[idx],
                                                       None)
                if future_sample_offset == 0:
                    terminals[idx] = True

            batch['observations.' + self._desired_goal_key] = desired_goals
            batch['next_observations.' + self._desired_goal_key] = desired_goals
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
