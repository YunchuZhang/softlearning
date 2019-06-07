from .sac import SAC

from softlearning.replay_pools.vae_replay_pool import VAEReplayPool

class SAC_VAE(SAC):
    def __init__(self,
                 VAETrainer,
                 train_vae_every=1,
                 **kwargs):
        super(SAC_VAE, self).__init__(**kwargs)
        self.VAETrainer = VAETrainer
        self.vae_num_epoch = 0
        self._train_vae_every = train_vae_every
        self._init_train_batches = 100
        self._train_batches = 100

    def _training_before_hook(self):
        super()._training_before_hook()

        if self._init_train_batches > 0:
            assert(isinstance(self._pool, VAEReplayPool))
            print("starting vae training")
            # Should work other than the fact that vae trainer uses data['next_obs']
            self.VAETrainer.train_epoch(self.vae_num_epoch,
                                        sample_batch=self.sampler.random_batch,
                                        batches=self._init_train_batches)
            self.vae_num_epoch += 1
            self._pool.refresh_latents()

            print("finished training before hook")

    def _epoch_before_hook(self):
        super()._epoch_before_hook()
        if self._epoch and self._epoch % self._train_vae_every == 0:
            assert(isinstance(self._pool, VAEReplayPool))
            # Should work other than the fact that vae trainer uses data['next_obs']
            self.VAETrainer.train_epoch(self.vae_num_epoch,
                                        sample_batch=self.sampler.random_batch,
                                        batches=self._train_batches)
            self.vae_num_epoch += 1

            self._pool.refresh_latents()

        #print("finished epoch before hook")
