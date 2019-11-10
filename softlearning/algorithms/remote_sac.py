from collections import OrderedDict

import os
import numpy as np
import matplotlib.pyplot as plt

import ray

from .rl_algorithm import RLAlgorithm
from .sac_agent import SACAgent as RemoteSACAgent
from softlearning.misc.utils import save_video


class RemoteSAC(RLAlgorithm):
    def __init__(
            self,
            variant,
            plotter=None,
            num_agents=7,
            tf_summaries=False,
            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,
            save_full_state=False,
            remote=True,
            n_initial_exploration_steps=0,
            avg_weights_every_n_steps=1,
            batch_size=None,
            observation_keys=None,
            pretrained_map3D=True,
            stop_3D_grads=False,
            do_cropping=False,
            **kwargs):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """
        #print("sac kwargs", kwargs)
        super(RemoteSAC, self).__init__(**kwargs)

        #RemoteSACAgent = ray.remote(SACAgent)

        self._num_agents = num_agents

        self._agents = [RemoteSACAgent.remote(
            variant,
            tf_summaries=tf_summaries,
            lr=lr,
            reward_scale=reward_scale,
            target_entropy=target_entropy,
            discount=discount,
            tau=tau,
            target_update_interval=target_update_interval,
            action_prior=action_prior,
            reparameterize=reparameterize,
            store_extra_policy_info=store_extra_policy_info,
            save_full_state=save_full_state,
            remote=remote,
            n_initial_exploration_steps=n_initial_exploration_steps,
            batch_size=batch_size,
            observation_keys=observation_keys,
            pretrained_map3D=pretrained_map3D,
            stop_3D_grads=stop_3D_grads,
            do_cropping=do_cropping
        ) for _ in range(num_agents)]

        self._weights = ray.get(self._agents[0].get_weights.remote())

        self._plotter = plotter
        self.avg_weights_every_n_steps = avg_weights_every_n_steps

    def _initial_exploration_hook(self):
        ray.get([agent.initial_exploration.remote() for agent in self._agents])

    def _init_training(self):
        ray.get([agent.init_training.remote() for agent in self._agents])

    def _init_sampler(self):
        ray.get([agent.init_sampler.remote() for agent in self._agents])

    def _total_samples(self):
        return np.sum(ray.get([agent.total_samples.remote()
                               for agent in self._agents]))
        #return ray.get(self._agents[0].total_samples.remote())

    def _do_sampling(self, timestep=None, steps=1):
        ray.get([agent.do_sampling.remote(timestep, steps) for agent in self._agents])

    def _terminate_sampler(self):
        ray.get([agent.terminate_sampler.remote() for agent in self._agents])

    def train(self, *args, **kwargs):
        """Initiate training of the SAC instance."""

        return self._train(
            *args,
            **kwargs)

    def _do_training(self, iteration, steps=1):
        for _ in range(0, steps, self.avg_weights_every_n_steps):
            weights_id = ray.put(self._weights)
            weights_list = ray.get(
                [agent.do_training.remote(iteration,
                                          steps=self.avg_weights_every_n_steps,
                                          weights=weights_id)
                 for agent in self._agents])
            self._weights = {
                    k: (sum(weights[k] for weights in weights_list) / self._num_agents)
                    for k in weights_list[0]
            }

    @property
    def ready_to_train(self):
        ready = all(ray.get([agent.ready_to_train.remote() for agent in self._agents]))
        #print("sac ready", ready)
        return ready

    def _training_batch(self, batch_size=None):
        return ray.get(self._agents[0].training_batch.remote(batch_size=batch_size))

    def _training_paths(self, epoch_length):
        return ray.get(self._agents[0].training_paths.remote(epoch_length))

    def _evaluation_paths(self):
        if self._eval_n_episodes < 1: return ()

        weight_ids = ray.put(self._weights)

        # TODO split across agents
        paths = ray.get(self._agents[0].evaluation_paths.remote(self._eval_n_episodes,
                                                                self._eval_deterministic,
                                                                self._eval_render_mode,
                                                                self._eval_render_goals,
                                                                weights=weight_ids))
        should_save_video = (
            self._video_save_frequency > 0
            and self._epoch % self._video_save_frequency == 0)

        if should_save_video:
            for i, path in enumerate(paths):
                if 'images' in path:
                    video_frames = path.pop('images')
                    video_file_name = f'evaluation_path_{self._epoch}_{i}.avi'
                    video_file_path = os.path.join(
                        os.getcwd(), 'videos', video_file_name)
                    save_video(video_frames, video_file_path)
                if 'goal_image' in path:
                    goal_file_name = f'goal_image_{self._epoch}_{i}.png'
                    goal_file_path = os.path.join(
                        os.getcwd(), 'videos', goal_file_name)
                    plt.imsave(goal_file_path, path['goal_image'], format='png')
                    

        return paths


    def _env_path_info(self, paths):
        return ray.get(self._agents[0].env_path_info.remote(paths))

    def _eval_env_path_info(self, paths):
        return ray.get(self._agents[0].eval_env_path_info.remote(paths))

    def _sampler_diagnostics(self):
        return ray.get(self._agents[0].sampler_diagnostics.remote())


    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        Q_values, Q_losses, alpha, global_step, policy_diagnostics = ray.get(self._agents[0].get_diagnostics.remote(iteration, batch))

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha,
        })

        #policy_diagnostics = ray.get(self._agents[0].policy_diagnostics.remote(batch))
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

        return diagnostics


    def _attempt_render(self, paths):
        #self.agent.render_rollouts(paths)
        pass

    def get_policy(self):
        return ray.get(self._agents[0].get_policy.remote())


    def get_Qs(self):
        return ray.get(self._agents[0].get_Qs.remote())


    def get_3Dmodel(self):
        return ray.get(self._agents[0].get_3Dmodel.remote())


    def get_agent_timings(self):
        return ray.get(self._agents[0].get_timings.remote())


    def reset_agent_timings(self):
        ray.get(self._agents[0].reset_timings.remote())
