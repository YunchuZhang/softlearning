from .adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter,
)
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras
import gym

ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
}

ADAPTERS = {
    'gym': GymAdapter,
}


def get_environment(universe, domain, task, environment_params):
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
    #environment_kwargs_gym.pop("map3D")
    #environment_kwargs_gym.pop("observation_keys")
    env = gym.make(f"{domain}-{task}",**environment_kwargs_gym)
    camera_space={'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}

    env = ImageEnv(
            wrapped_env=env,
            imsize=64,
            normalize=True,
            camera_space=camera_space,
            init_camera=(lambda x: init_multiple_cameras(x, camera_space)),
            num_cameras=4,
            depth=True,
            cam_info=True,
            reward_type='wrapped_env',
            flatten=False
        )
    environment_kwargs = environment_params.get('kwargs', {}).copy()
    environment_kwargs["env"] = env
    return get_environment(universe, None, None, environment_kwargs)
