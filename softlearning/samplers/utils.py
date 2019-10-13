from copy import deepcopy
import ipdb
st = ipdb.set_trace
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
             do_cropping=False,
             memory3D=None,
             obs_ph=None,
             session=None,
             callback=None,
             render_mode=None,
             render_goals=False,
             break_on_terminal=True,
             batch_size=None,
             do_cropping=False):

    pool = replay_pools.SimpleReplayPool(env, max_size=path_length)

    if sampler is None:
        sampler = simple_sampler.SimpleSampler(
            max_path_length=path_length,
            min_pool_size=None,
            batch_size=batch_size,
            store_last_n_paths=n_paths,
        )

    sampler.initialize(env,
                       policy,
                       pool,
                       memory3D=memory3D,
                       obs_ph=obs_ph,
                       session=session)

    sampler.clear_last_n_paths()
    sampler.store_last_n_paths = n_paths

    videos = []
    if render_goals:
        goals = []

    while len(sampler.get_last_n_paths()) < n_paths:

        sampler.reset()
        images = []
        t = 0
        goal = None

        for t in range(path_length):
            observation, reward, terminal, info = sampler.sample(do_cropping)

            if callback is not None:
                callback(observation)

            if render_mode is not None:
                if render_mode == 'rgb_array':
                    if render_goals and goal is None:
                        image, goal = env.render(mode=render_mode, render_goal=True)
                    else:
                        image = env.render(mode=render_mode)
                    images.append(image)
                else:
                    env.render()

            if np.count_nonzero(terminal):
                policy.reset()
                if break_on_terminal: break

        videos.append(images)
        if render_goals:
            goals.append(goal)

        #assert pool._size == t + 1

    paths = sampler.get_last_n_paths()

    if render_mode == 'rgb_array':
        for i in range(len(videos)):
            paths[i]['images'] = np.stack(videos[i], axis=0)
            if render_goals:
                paths[i]['goal_image'] = goals[i]

    return paths


def rollout(*args, **kwargs):
    path = rollouts(1, *args, **kwargs)[0]
    return path
