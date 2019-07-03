import tensorflow as tf
import numpy as np
import abc

class VAEBase(tf.keras.Model, metaclass=abc.ABCMeta):
    def __init__(self, representation_size, name="vae_base"):
        """
        Skipping the serialization part since it should not be a problem in ternsorflow
        """
        super(VAEBase, self).__init__(name)
        self.representation_size = representation_size

    @abc.abstractmethod
    def encode(self, input, training=True):
        """
        :param input:
        :return: latent_distribution_params
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rsample(self, latent_distribution_params, training=True):
        """

        :param latent_distribution_params:
        :return: latents
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reparameterize(self, latent_distribution_params, training=True):
        """

        :param latent_distribution_params:
        :return: latents
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, latents, training=True):
        """
        :param latents:
        :return: reconstruction, obs_distribution_params
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def logprob(self, inputs, obs_distribution_params, training=True):
        """
        :param inputs:
        :param obs_distribution_params:
        :return: log probability of input under decoder
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def kl_divergence(self, latent_distribution_params, training=True):
        """
        :param latent_distribution_params:
        :return: kl div between latent_distribution_params and prior on latent space
        """
        raise NotImplementedError()

    def call(self, input, training=True):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        latent_distribution_params = self.encode(input, training=training)
        latents = self.reparameterize(latent_distribution_params, training=training)
        reconstructions, obs_distribution_params = self.decode(latents, training=training)
        return reconstructions, obs_distribution_params, latent_distribution_params


class GaussianLatentVAE(VAEBase):
    def __init__(
            self,
            representation_size,
            name="gaussian_latent_vae"
    ):
        super(GaussianLatentVAE, self).__init__(representation_size, name="gaussian_latent_vae")
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)

    def rsample(self, latent_distribution_params, training=True):
        """
        input is mu and logvar
        it returns N(mu, e^(logvar / 2))
        """
        mu, logvar = latent_distribution_params
        stds = tf.keras.backend.exp(0.5 * logvar)
        epsilon = tf.random.normal(tf.shape(mu))
        latents = epsilon * stds + mu
        return latents

    def rsample_multiple_latents(self, latent_distribution_params,
                                 num_latents_to_sample=1, training=True):
        """
        input is mu and logvar

        same as rsample, except that it samples num_latents_to_sample examples
        """
        mu, logvar = latent_distribution_params
        mu = tf.keras.backend.expand_dims(mu, axis=1)
        stds = tf.keras.backend.exp(0.5 * logvar)
        stds = tf.keras.backend.expand_dims(stds, axis=1)
        mu_shape = tf.shape(mu)
        # TODO: This is hopefully expected behaviour
        epsilon = tf.random.normal(shape=[mu_shape[0], num_latents_to_sample, mu_shape[1]])
        latents = epsilon * stds + mu
        return latents

    def reparameterize(self, latent_distribution_params, training=True):
        if training:
            return self.rsample(latent_distribution_params, training=training)
        else:
            return latent_distribution_params[0]

    def kl_divergence(self, latent_distribution_params, training=True):
        """
        See Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

        Or just look it up.

        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        Note that sometimes people write log(sigma), but this is the same as
        0.5 * log(sigma^2).

        :param latent_distribution_params:
        :return:
        """
        mu, logvar = latent_distribution_params

        return -0.5 * tf.keras.backend.mean(
                           tf.keras.backend.sum(1 + logvar - tf.keras.backend.pow(mu, 2) - tf.keras.backend.exp(logvar), axis=1)
                      )
