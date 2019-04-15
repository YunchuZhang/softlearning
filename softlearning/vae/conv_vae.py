import tensorflow as tf
import numpy as np
from conv_networks import CNN, DCNN
from vae_base import GaussianLatentVAE

###### DEFAULT ARCHITECTURES #########

imsize48_default_architecture = dict(
    conv_args=dict(
        kernel_sizes=[5, 3, 3],
        n_channels=[16, 32, 64],
        strides=[3, 2, 2],
    ),
    conv_kwargs=dict(
        hidden_sizes=[],
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),
    deconv_args=dict(
        hidden_sizes=[],

        deconv_input_width=3,
        deconv_input_height=3,
        deconv_input_channels=64,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[3, 3],
        n_channels=[32, 16],
        strides=[2, 2],
    ),
    deconv_kwargs=dict(
        batch_norm_deconv=False,
        batch_norm_fc=False,
    )
)

imsize48_default_architecture_with_more_hidden_layers = dict(
    conv_args=dict(
        kernel_sizes=[5, 3, 3],
        n_channels=[16, 32, 64],
        strides=[3, 2, 2],
    ),
    conv_kwargs=dict(
        hidden_sizes=[500, 300, 150],
    ),
    deconv_args=dict(
        hidden_sizes=[150, 300, 500],

        deconv_input_width=3,
        deconv_input_height=3,
        deconv_input_channels=64,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[3, 3],
        n_channels=[32, 16],
        strides=[2, 2],
    ),
    deconv_kwargs=dict(
    )
)

imsize84_default_architecture = dict(
    conv_args=dict(
        kernel_sizes=[5, 5, 5],
        n_channels=[16, 32, 32],
        strides=[3, 3, 3],
    ),
    conv_kwargs=dict(
        hidden_sizes=[],
        batch_norm_conv=True,
        batch_norm_fc=False,
    ),
    deconv_args=dict(
        hidden_sizes=[],

        deconv_input_width=2,
        deconv_input_height=2,
        deconv_input_channels=32,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[5, 6],
        n_channels=[32, 16],
        strides=[3, 3],
    ),
    deconv_kwargs=dict(
        batch_norm_deconv=False,
        batch_norm_fc=False,
    )
)

def fanin_init():
    return tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="uniform")

class ConvVAE(GaussianLatentVAE):
    def __init__(
            self,
            representation_size,
            architecture,

            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation="linear",
            decoder_distribution="bernoulli",

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=fanin_init,
            name="conv_vae",
            encoder_name="cnn",
            decoder_name="dcnn"
    ):
        """

        :param representation_size:
        :param conv_args:
        must be a dictionary specifying the following:
            kernel_sizes
            n_channels
            strides
        :param conv_kwargs:
        a dictionary specifying the following:
            hidden_sizes
            batch_norm
        :param deconv_args:
        must be a dictionary specifying the following:
            hidden_sizes
            deconv_input_width
            deconv_input_height
            deconv_input_channels
            deconv_output_kernel_size
            deconv_output_strides
            deconv_output_channels
            kernel_sizes
            n_channels
            strides
        :param deconv_kwargs:
            batch_norm
        :param encoder_class:
        :param decoder_class:
        :param decoder_output_activation:
        :param decoder_distribution:
        :param input_channels:
        :param imsize:
        :param init_w:
        :param min_variance:
        :param hidden_init:
        """
        super(ConvVAE, self).__init__(representation_size, name)
        self.encoder_name = self.name + "_" + encoder_name
        self.decoder_name = self.name + "_" + decoder_name

        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))

        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize * self.imsize * self.input_channels

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size = deconv_args['deconv_input_width'] * \
                           deconv_args['deconv_input_height'] * \
                           deconv_args['deconv_input_channels']

        self.encoder = encoder_class(
            **conv_args,
            paddings=['valid' for i in range(len(conv_args['kernel_sizes']))],
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels,
            output_size=conv_output_size,
            init_w=init_w,
            hidden_init=hidden_init,
            name=self.encoder_name,
            **conv_kwargs)

        self.fc1 = tf.keras.layers.Dense(
                            representation_size,
                            kernel_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w),
                            bias_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w)
                   )

        self.fc2 = tf.keras.layers.Dense(
                            representation_size,
                            kernel_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w),
                            bias_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w)
                   )

        self.decoder = decoder_class(
            **deconv_args,
            fc_input_size=representation_size,
            init_w=init_w,
            output_activation=decoder_output_activation,
            paddings=['valid' for i in range(len(deconv_args['kernel_sizes']))],
            hidden_init=hidden_init,
            name=self.decoder_name,
            **deconv_kwargs)

        self.epoch = 0
        self.decoder_distribution = decoder_distribution
        self.flatten_decode = tf.keras.layers.Flatten()

    def encode(self, input, training=True):
        h = self.encoder(input, training=training)
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + tf.keras.backend.abs(self.fc2(h))
        return mu, logvar

    def decode(self, latents, training=True):
        decoded = self.decoder(latents, training=training)
        decoded = self.flatten_decode(decoded)

        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return tf.keras.backend.clip(decoded, 0, 1), [tf.keras.backend.clip(decoded, 0, 1), tf.keras.backend.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(
                self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params, training=True):
        if self.decoder_distribution == 'bernoulli':
            inputs = inputs[:, 0 : self.imlength]
            return -tf.losses.sigmoid_cross_entropy(inputs, obs_distribution_params[0])

        elif self.decoder_distribution == 'gaussian_identity_variance':
            inputs = inputs[:, 0 : self.imlength]

            logprob = -1 * tf.keras.backend.mean(tf.keras.backend.pow(inputs - obs_distribution_params, 2))
            return log_prob

        else:
            raise NotImplementedError('Distribution {} not supported'.format(
                self.decoder_distribution))
