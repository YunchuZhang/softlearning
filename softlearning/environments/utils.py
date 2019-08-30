import ipdb
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras
st = ipdb.set_trace
import gym
from .adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter,
)

from .adapters.remote_gym_adapter import RemoteGymAdapter
from .adapters.vae_wrapper import VAEWrappedEnv

ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
    'remote_gym': GYM_ENVIRONMENTS,
    'vae': GYM_ENVIRONMENTS
}

ADAPTERS = {
    'gym': GymAdapter,
    'remote_gym': RemoteGymAdapter,
    'vae': VAEWrappedEnv,
}


def get_environment(universe, domain, task, environment_params):
    # st()
    if "env" in environment_params and environment_params["env"]:
        domain = None
        task = None

    return ADAPTERS[universe](domain, task, **environment_params)


def get_environment_from_params(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    environment_kwargs = environment_params.get('kwargs', {}).copy()
    return get_environment(universe, domain, task, environment_kwargs)


def get_environment_from_params_custom(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    # st()
    environment_kwargs_gym = environment_params.get('kwargs', {}).copy()

    assert("map3D" not in environment_kwargs_gym)
    if "observation_keys" in environment_kwargs_gym:
      environment_kwargs_gym.pop("observation_keys")

    env = gym.make(f"{domain}-{task}",**environment_kwargs_gym)

    env_n = ImageEnv(env,
                     imsize=64,
                     normalize=True,
                     init_camera=init_multiple_cameras,
                     num_cameras=57,
                     num_views=4,
                     depth=True,
                     cam_angles=True,
                     reward_type="wrapped_env",
                     flatten=False)

    environment_kwargs = environment_params.get('kwargs', {}).copy()
    environment_kwargs["env"] = env_n
    return get_environment(universe, domain, task, environment_kwargs)
