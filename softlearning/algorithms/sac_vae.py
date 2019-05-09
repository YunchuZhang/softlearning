from .sac import SAC

class SAC_VAE(SAC):
    def __init__(self, VAETrainer, train_vae_every=10, **kwargs):
        super(SAC_VAE, self).__init__(**kwargs)
        self.VAETrainer = VAETrainer
        self.vae_num_epoch = 0
        self.train_vae_every = train_vae_every

    def _training_before_hook(self):
        super()._training_before_hook()
        # Should work other than the fact that vae trainer uses data['next_obs']
        self.VAETrainer.train_epoch(self.vae_num_epoch, self.sampler.random_batch)
        self.vae_num_epoch += 1

    def _epoch_before_hook(self):
        super()._epoch_before_hook()
        if self._epoch and self._epoch % self.train_vae_every == 0:
            # Should work other than the fact that vae trainer uses data['next_obs']
            self.VAETrainer.train_epoch(self.vae_num_epoch, self.sampler.random_batch)
            self.vae_num_epoch += 1
