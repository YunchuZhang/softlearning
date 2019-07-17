import ipdb
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras
st = ipdb.set_trace
import gym
from .adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter,
)

from .adapters.vae_wrapper import (
    VAEWrappedEnv
)

ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
    'vae': GYM_ENVIRONMENTS
}

ADAPTERS = {
    'gym': GymAdapter,
    'vae': VAEWrappedEnv,
}


def get_environment(universe, domain, task, environment_params):
    # st()
    if hasattr(environment_params, 'env'):
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
    if "map3D" in environment_kwargs_gym:
      environment_kwargs_gym.pop("map3D")
    if "observation_keys" in environment_kwargs_gym:
      environment_kwargs_gym.pop("observation_keys")
    env = gym.make(f"{domain}-{task}",**environment_kwargs_gym)

    env_n = ImageEnv(env,
                   imsize=84,
                   normalize=True,
                   init_camera=init_multiple_cameras,
                   num_cameras=4,
                   depth=True,
                   cam_angles=True,
                   reward_type="wrapped_env",
                   flatten=False)

    environment_kwargs = environment_params.get('kwargs', {}).copy()
    environment_kwargs["env"] = env_n
    return get_environment(universe, domain, task, environment_kwargs)
