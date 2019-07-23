import abc
from collections import OrderedDict
from itertools import count
import gtimer as gt
import os

import numpy as np
import time
import ipdb
st = ipdb.set_trace


class RLAlgorithm():
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            n_epochs=1000,
            train_every_n_steps=1,
            n_train_repeat=1,
            max_train_repeat_per_timestep=5,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render_mode=None,
            video_save_frequency=0,
            **kwargs
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            n_initial_exploration_steps: Number of steps in the beginning to
                take using actions drawn from a separate exploration policy.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render_mode (`str`): Mode to render evaluation rollouts in.
                None to disable rendering.
        """
        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._max_train_repeat_per_timestep = max(
            max_train_repeat_per_timestep, n_train_repeat)
        self._train_every_n_steps = train_every_n_steps
        self._epoch_length = epoch_length

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._video_save_frequency = video_save_frequency

        if self._video_save_frequency > 0:
            assert eval_render_mode != 'human', (
                "RlAlgorithm cannot render and save videos at the same time")
            self._eval_render_mode = 'rgb_array'
        else:
            self._eval_render_mode = eval_render_mode

        self._epoch = 0
        self._timestep = 0
        self._num_train_steps = 0

    def _initial_exploration_hook(self):
        pass

    def _training_before_hook(self):
        """Method called before the actual training loops."""
        pass

    def _training_after_hook(self):
        """Method called after the actual training loops."""
        pass

    def _timestep_before_hook(self, *args, **kwargs):
        """Hook called at the beginning of each timestep."""
        pass

    def _timestep_after_hook(self, *args, **kwargs):
        """Hook called at the end of each timestep."""
        pass

    def _epoch_before_hook(self):
        """Hook called at the beginning of each epoch."""
        self._train_steps_this_epoch = 0

    def _epoch_after_hook(self, *args, **kwargs):
        """Hook called at the end of each epoch."""
        pass

    @abc.abstractmethod
    def _training_batch(self, batch_size=None):
        raise NotImplementedError

    def _evaluation_batch(self, *args, **kwargs):
        return self._training_batch(*args, **kwargs)

    @property
    def _training_started(self):
        return self._total_timestep > 0

    @property
    def _total_timestep(self):
        total_timestep = self._epoch * self._epoch_length + self._timestep
        return total_timestep

    @abc.abstractmethod
    def _total_samples(self):
        raise NotImplementedError

    def _train(self):
        """Return a generator that performs RL training.
        """

        if not self._training_started:
            self._init_training()

            self._initial_exploration_hook()

        self._init_sampler()

        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

        self._training_before_hook()

        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            start_samples = self._total_samples()
            for i in count():
                samples_now = self._total_samples()
                self._timestep = samples_now - start_samples

                if (samples_now >= start_samples + self._epoch_length
                    and self.ready_to_train):
                    break

                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                self._do_sampling(timestep=self._total_timestep,
                                  steps=self._train_every_n_steps)
                gt.stamp('sample')

                # print(samples_now,start_samples,self._epoch_length,"params")
                gt.stamp('sample')
                # print("training")
                if self.ready_to_train:
                    self._do_training_repeats(timestep=self._total_timestep)

                gt.stamp('train')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

            training_paths = self._training_paths(self._epoch_length)
            gt.stamp('training_paths')

            evaluation_paths = self._evaluation_paths()
            gt.stamp('evaluation_paths')

            training_metrics = self._evaluate_rollouts(
                training_paths,
                self._env_path_info(training_paths))
            gt.stamp('training_metrics')
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths,
                    self._eval_env_path_info(evaluation_paths))
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}
            print("evaluation done")
            self._epoch_after_hook(training_paths)
            gt.stamp('epoch_after_hook')

            sampler_diagnostics = self._sampler_diagnostics()

            # st()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            time_diagnostics = gt.get_times().stamps.itrs

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'training/{key}', training_metrics[key])
                    for key in sorted(training_metrics.keys())
                ),
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                *(
                    (f'sampler/{key}', sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
            )))

            if self._eval_render_mode is not None:
                self._attempt_render(evaluation_paths)

            yield diagnostics

        self._terminate_sampler()

        self._training_after_hook()

        yield {'done': True, **diagnostics}

    @abc.abstractmethod
    def _training_paths(self, epoch_length):
        raise NotImplementedError

    @abc.abstractmethod
    def _evaluation_paths(self):
        raise NotImplementedError

    def _evaluate_rollouts(self, paths, env_infos):
        """Compute evaluation metrics for the given rollouts."""

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]

        diagnostics = OrderedDict((
            ('return-average', np.mean(total_returns)),
            ('return-min', np.min(total_returns)),
            ('return-max', np.max(total_returns)),
            ('return-std', np.std(total_returns)),
            ('episode-length-avg', np.mean(episode_lengths)),
            ('episode-length-min', np.min(episode_lengths)),
            ('episode-length-max', np.max(episode_lengths)),
            ('episode-length-std', np.std(episode_lengths)),
        ))

        for key, value in env_infos.items():
            diagnostics[f'env_infos/{key}'] = value

        return diagnostics

    @abc.abstractmethod
    def _env_path_info(self, paths):
        raise NotImplementedError

    @abc.abstractmethod
    def _eval_env_path_info(self, paths):
        raise NotImplementedError

    @abc.abstractmethod
    def _sampler_diagnostics(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        raise NotImplementedError

    @property
    def ready_to_train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_sampling(self, timestep, steps=1):
        raise NotImplementedError

    def _do_training_repeats(self, timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        trained_enough = (
            self._train_steps_this_epoch
            > self._max_train_repeat_per_timestep * self._timestep)
        if trained_enough: return

        #for i in range(self._n_train_repeat):
        self._do_training(iteration=timestep, steps=self._n_train_repeat)

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat

    @abc.abstractmethod
    def _do_training(self, iteration, steps=1):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_sampler(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _terminate_sampler(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _attempt_render(self, paths):
        raise NotImplementedError

    def __getstate__(self):
        state = {
            '_epoch_length': self._epoch_length,
            '_epoch': (
                self._epoch + int(self._timestep >= self._epoch_length)),
            '_timestep': self._timestep % self._epoch_length,
            '_num_train_steps': self._num_train_steps,
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
