from copy import deepcopy

import numpy as np

from . import (
    simple_replay_pool,
    extra_policy_info_replay_pool,
    union_pool,
    her_replay_pool,
    vae_replay_pool,
    trajectory_replay_pool)


POOL_CLASSES = {
    'SimpleReplayPool': simple_replay_pool.SimpleReplayPool,
    "SimpleReplayPoolTemp": simple_replay_pool.SimpleReplayPool,
    'TrajectoryReplayPool': trajectory_replay_pool.TrajectoryReplayPool,
    'ExtraPolicyInfoReplayPool': (
        extra_policy_info_replay_pool.ExtraPolicyInfoReplayPool),
    'UnionPool': union_pool.UnionPool,
    'HerReplayPool': her_replay_pool.HerReplayPool,
    'VAEReplayPool': vae_replay_pool.VAEReplayPool
}


def get_replay_pool_from_variant(variant, env, *args, **kwargs):
    replay_pool_params = variant['replay_pool_params']
    replay_pool_type = replay_pool_params['type']
    replay_pool_kwargs = deepcopy(replay_pool_params['kwargs'])

    replay_pool = POOL_CLASSES[replay_pool_type](
        *args,
        env,
        **replay_pool_kwargs,
        **kwargs)

    return replay_pool


def normalize_image(image):
    assert image.dtype == np.uint8
    return (np.float32(image) / 255.0) - 0.5


def unnormalize_image(image):
    assert image.dtype != np.uint8
    # image = np.concatenate([image,np.ones_like(image[...,:1])],-1)
    return np.uint8((image + 0.5) * 255.0)
