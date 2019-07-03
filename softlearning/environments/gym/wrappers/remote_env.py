import gym
import ray

from softlearning.environments.gym.wrappers import NormalizeActionWrapper


@ray.remote
class RemoteGymEnv(object):

    def __init__(self,
                 name,
                 normalize=True,
                 env_params=None):

        env = gym.make(name, **env_params)

        if normalize:
            env = NormalizeActionWrapper(env)

        self._env = env


    def step(self, action):
        return self._env.step(action)


    def reset(self):
        return self._env.reset()


    def close(self):
        return self._env.close()


    def seed(self):
        return self._env.seed()


    def observation_space(self):
        return self._env.observation_space


    def action_space(self):
        return self._env.action_space


    def is_multiworld_env(self):
        # TODO: fix this to work with non-multitask environments
        return hasattr(self._env.env, 'compute_rewards')


    def compute_reward(self,
                       achieved_goal=None,
                       desired_goal=None,
                       info=None,
                       actions=None,
                       observations=None):
        # We assume that the reward computation is the same for all environments

        if self.is_multiworld_env:
            return self._env.env.compute_rewards(actions, observations)[0]
        else:
            return self._env.compute_reward(achieved_goal, desired_goal, info)
