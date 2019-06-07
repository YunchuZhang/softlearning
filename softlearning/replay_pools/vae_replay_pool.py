import numpy as np

from .simple_replay_pool import SimpleReplayPool


class VAEReplayPool(SimpleReplayPool):

    def __init__(self,
                 env,
                 latent_key_pairs,
                 latent_obs_key,
                 *args,
                 **kwargs):

        self.env = env
        self._latent_key_pairs = latent_key_pairs
        self._latent_obs_key = latent_obs_key

        super(VAEReplayPool, self).__init__(*args, env, **kwargs)


    def refresh_latents(self):
        from .utils import normalize_image

        cur_idx = 0
        batch_size = 512
        next_idx = min(batch_size, self.size)

        while cur_idx < self.size:
            idxs = np.arange(cur_idx, next_idx)

            for base, latent in self._latent_key_pairs.items():
                self.fields[latent][idxs] = self.env._encode(
                    normalize_image(self.fields[base][idxs])
                )

            cur_idx = next_idx
            next_idx += batch_size
            next_idx = min(next_idx, self.size)

        self.env.vae.dist_mu =  np.mean(self.fields[self._latent_obs_key], axis=0)
        self.env.vae.dist_std = np.std(self.fields[self._latent_obs_key], axis=0)
