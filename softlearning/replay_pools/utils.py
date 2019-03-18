from copy import deepcopy

from . import (
    simple_replay_pool,
    extra_policy_info_replay_pool,
    union_pool,
    her_replay_pool,
    trajectory_replay_pool)


POOL_CLASSES = {
    'SimpleReplayPool': simple_replay_pool.SimpleReplayPool,
    'TrajectoryReplayPool': trajectory_replay_pool.TrajectoryReplayPool,
    'ExtraPolicyInfoReplayPool': (
        extra_policy_info_replay_pool.ExtraPolicyInfoReplayPool),
    'UnionPool': union_pool.UnionPool,
    'HerReplayPool': her_replay_pool.HerReplayPool
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
