from collections import defaultdict

import numpy as np
from gym.spaces import Box, Dict, Discrete

from .flexible_replay_pool import FlexibleReplayPool
import ipdb
st= ipdb.set_trace

def normalize_observation_fields(observation_space, name='observations'):
    if isinstance(observation_space, Dict):
        fields = [
            normalize_observation_fields(child_observation_space, name)
            for name, child_observation_space
            in observation_space.spaces.items()
        ]
        fields = {
            'observations.{}'.format(name): value
            for field in fields
            for name, value in field.items()
        }
    elif isinstance(observation_space, (Box, Discrete)):
        if 'image' in name:
            observation_space.dtype = np.uint8
        fields = {
            name: {
                'shape': observation_space.shape,
                'dtype': observation_space.dtype,
            }
        }
    else:
        raise NotImplementedError(
            "Observation space of type '{}' not supported."
            "".format(type(observation_space)))

    return fields


class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self,
                 env,
                 concat_observations=True,
                 filter_key=None,
                 *args,
                 **kwargs):

        self.concat_observations=concat_observations
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        st()

        observation_fields = normalize_observation_fields(self._observation_space)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have
        # to worry about termination conditions.
        observation_fields.update({
            'next_' + key: value
            for key, value in observation_fields.items()
        })

        fields = {
            **observation_fields,
            **{
                'actions': {
                    'shape': self._action_space.shape,
                    'dtype': 'float32'
                },
                'rewards': {
                    'shape': (1, ),
                    'dtype': 'float32'
                },
                # self.terminals[i] = a terminal was received at time i
                'terminals': {
                    'shape': (1, ),
                    'dtype': 'bool'
                },
            }
        }
        if filter_key:
            fields = {i:fields[i] for i in filter_key}


        super(SimpleReplayPool, self).__init__(
            *args, fields_attrs=fields, **kwargs)

    def add_samples(self, samples):
        from .utils import unnormalize_image

        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).add_samples(samples)


        dict_observations = defaultdict(list)
        for observation in samples['observations']:
            for key, value in observation.items():
                if 'image' in key and value is not None:
                    value = unnormalize_image(value)
                dict_observations[key].append(value)

        dict_next_observations = defaultdict(list)
        for next_observation in samples['next_observations']:
            for key, value in next_observation.items():
                if 'image' in key and value is not None:
                    value = unnormalize_image(value)
                dict_next_observations[key].append(value)

        samples.update(
           **{
               f'observations.{observation_key}': np.array(values)
               for observation_key, values in dict_observations.items()
           },
           **{
               f'next_observations.{observation_key}': np.array(values)
               for observation_key, values in dict_next_observations.items()
           },
        )

        del samples['observations']
        del samples['next_observations']

        return super(SimpleReplayPool, self).add_samples(samples)


    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         observation_keys=None):
        from .utils import normalize_image

        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).batch_by_indices(
                indices, field_name_filter=field_name_filter)

        batch = {
            field_name: self.fields[field_name][indices]
            for field_name in self.field_names
        }

        if observation_keys is None:
            observation_keys = tuple(self._observation_space.spaces.keys())

        for key, value in batch.items():
            if 'image' in key and value is not None:
                value = normalize_image(value)
                batch[key] = value

        if self.concat_observations:
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


    def terminate_episode(self):
        pass
