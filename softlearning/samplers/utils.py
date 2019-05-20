from copy import deepcopy

import numpy as np

from softlearning import replay_pools
from . import (
    dummy_sampler,
    extra_policy_info_sampler,
    remote_sampler,
    base_sampler,
    simple_sampler,
    multiagent_sampler)


SAMPLERS = {
    'DummySampler': dummy_sampler.DummySampler,
    'ExtraPolicyInfoSampler': (
        extra_policy_info_sampler.ExtraPolicyInfoSampler),
    'RemoteSampler': remote_sampler.RemoteSampler,
    'Sampler': base_sampler.BaseSampler,
    'SimpleSampler': simple_sampler.SimpleSampler,
    'MultiAgentSampler': multiagent_sampler.MultiAgentSampler,
}


def get_sampler_from_variant(variant, *args, **kwargs):

    sampler_params = variant['sampler_params']
    sampler_type = sampler_params['type']

    sampler_args = deepcopy(sampler_params.get('args', ()))
    sampler_kwargs = deepcopy(sampler_params.get('kwargs', {}))

    sampler = SAMPLERS[sampler_type](
        *sampler_args, *args, **sampler_kwargs, **kwargs)

    return sampler


def rollouts(n_paths,
             env,
             policy,
             path_length,
             sampler=None,
             callback=None,
             render_mode=None,
             break_on_terminal=True):

    pool = replay_pools.SimpleReplayPool(env, max_size=path_length)

    if sampler is None:
        sampler = simple_sampler.SimpleSampler(
            max_path_length=path_length,
            min_pool_size=None,
            batch_size=None,
            store_last_n_paths=n_paths,
        )

    sampler.initialize(env, policy, pool)

    sampler.clear_last_n_paths()
    sampler.store_last_n_paths = n_paths

    videos = []

    while len(sampler.get_last_n_paths()) < n_paths:

        sampler.reset()
        images = []
        t = 0

        for t in range(path_length):
            observation, reward, terminal, info = sampler.sample()

            if callback is not None:
                callback(observation)

            if render_mode is not None:
                if render_mode == 'rgb_array':
                    image = env.render(mode=render_mode)
                    images.append(image)
                else:
                    env.render()

            if np.count_nonzero(terminal):
                policy.reset()
                if break_on_terminal: break

        videos.append(images)

        #assert pool._size == t + 1

    paths = sampler.get_last_n_paths()

    if render_mode == 'rgb_array':
        for i, path in enumerate(paths):
            path['images'] = np.stack(videos[i], axis=0)

    return paths


def rollout(*args, **kwargs):
    path = rollouts(1, *args, **kwargs)[0]
    return path
