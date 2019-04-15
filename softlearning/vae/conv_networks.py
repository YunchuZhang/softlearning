import tensorflow as tf
import numpy as np

class CNN(tf.keras.Model):
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            batch_norm_conv=False,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=tf.contrib.layers.xavier_initializer,
            hidden_activation=tf.keras.layers.ReLU(),
            output_activation="linear",
            name="cnn" # tf converts names to snakecase for some reason
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        super(CNN, self).__init__(name)

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.batch_norm_conv = batch_norm_conv
        self.batch_norm_fc = batch_norm_fc
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels

        self.reshape_to_conv = tf.keras.layers.Reshape((self.input_height, self.input_width, self.input_channels))

        self.conv_layers = []
        self.conv_norm_layers = []
        self.fc_layers = []
        self.fc_norm_layers = []

        self.flatten_layer = tf.keras.layers.Flatten()

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            # TODO: Make sure padding is 'same' or 'valid', and not a number
            conv = tf.keras.layers.Conv2D(out_channels,
                                          kernel_size,
                                          strides=stride,
                                          padding=padding,
                                          kernel_initializer=hidden_init(),
                                          bias_initializer=tf.keras.initializers.Zeros(),
                                         )
            self.conv_layers.append(conv)


            # Add a batch norm layer for each conv layer
            # epsilon and momentum based on torch default params
            # Torch and tf have a different meaning of momentum
            self.conv_norm_layers.append(tf.keras.layers.BatchNormalization(
                                                    epsilon=1e-5,
                                                    momentum=0.9,
                                                    ))

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = tf.keras.layers.Dense(
                            hidden_size,
                            kernel_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w),
                            bias_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w)
                       )
            self.fc_layers.append(fc_layer)

            self.fc_norm_layers.append(tf.keras.layers.BatchNormalization(
                                                epsilon=1e-5,
                                                momentum=0.9,
                                      ))

        self.last_fc = tf.keras.layers.Dense(
                            output_size,
                            kernel_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w),
                            bias_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w),
                            activation=self.output_activation
                       )

    def call(self, input, training=True):
        conv_input = input[:, :self.conv_input_length]

        if self.added_fc_input_size > 0:
            extra_fc_input = input[:, self.conv_input_length:self.conv_input_length + self.added_fc_input_size]

        h = self.reshape_to_conv(conv_input)

        h = self.apply_forward(h, self.conv_layers, self.conv_norm_layers,
                               use_batch_norm=self.batch_norm_conv, training=training)

        h = self.flatten_layer(h)

        if self.added_fc_input_size > 0:
            h = tf.keras.layers.concatenate([h, extra_fc_input], axis=1)

        h = self.apply_forward(h, self.fc_layers, self.fc_norm_layers,
                               use_batch_norm=self.batch_norm_fc, training=training)

        output = self.last_fc(h)
        return output

    def apply_forward(self, input, hidden_layers, norm_layers,
                      use_batch_norm=False, training=True):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h, training=training)
            h = self.hidden_activation(h)
        return h

class TwoHeadDCNN(tf.keras.Model):
    def __init__(
            self,
            fc_input_size,
            hidden_sizes,

            deconv_input_width,
            deconv_input_height,
            deconv_input_channels,

            deconv_output_kernel_size,
            deconv_output_strides,
            deconv_output_channels,

            kernel_sizes,
            n_channels,
            strides,
            paddings,

            batch_norm_deconv=False,
            batch_norm_fc=False,
            init_w=1e-3,
            hidden_init=tf.contrib.layers.xavier_initializer,
            hidden_activation=tf.keras.layers.ReLU(),
            output_activation="linear",
            name="two_head_dcnn" # tf converts names to snakecase for some reason
    ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        super(TwoHeadDCNN, self).__init__(name)

        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = self.deconv_input_channels * self.deconv_input_height * self.deconv_input_width
        self.batch_norm_deconv = batch_norm_deconv
        self.batch_norm_fc = batch_norm_fc

        self.reshape_to_deconv = tf.keras.layers.Reshape((self.deconv_input_height, self.deconv_input_width, self.deconv_input_channels))

        self.deconv_layers = []
        self.deconv_norm_layers = []
        self.fc_layers = []
        self.fc_norm_layers = []

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = tf.keras.layers.Dense(
                            hidden_size,
                            kernel_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w),
                            bias_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w)
                       )

            self.fc_norm_layers.append(tf.keras.layers.BatchNormalization(
                                                epsilon=1e-5,
                                                momentum=0.9,
                                      ))

        self.last_fc = tf.keras.layers.Dense(
                        deconv_input_size,
                        kernel_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w),
                        bias_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w)
                       )

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            # TODO: Make sure padding is 'same' or 'valid', and not a number
            deconv = tf.keras.layers.Conv2DTranspose(out_channels,
                                                     kernel_size,
                                                     strides=stride,
                                                     padding=padding,
                                                     kernel_initializer=hidden_init(),
                                                     bias_initializer=tf.keras.initializers.Zeros()
                                                    )
            self.deconv_layers.append(deconv)

            self.deconv_norm_layers.append(tf.keras.layers.BatchNormalization(
                                                    epsilon=1e-5,
                                                    momentum=0.9,
                                                    ))

        self.first_deconv_output = tf.keras.layers.Conv2DTranspose(
                                        deconv_output_channels,
                                        deconv_output_kernel_size,
                                        strides=deconv_output_strides,
                                        padding=padding,
                                        kernel_initializer=hidden_init(),
                                        bias_initializer=tf.keras.initializers.Zeros(),
                                        activation=self.output_activation
                                   )

        self.second_deconv_output = tf.keras.layers.Conv2DTranspose(
                                        deconv_output_channels,
                                        deconv_output_kernel_size,
                                        strides=deconv_output_strides,
                                        padding=padding,
                                        kernel_initializer=hidden_init(),
                                        bias_initializer=tf.keras.initializers.Zeros(),
                                        activation=self.output_activation
                                   )

    def call(self, input, training=True):
        h = self.apply_forward(input, self.fc_layers, self.fc_norm_layers,
                               use_batch_norm=self.batch_norm_fc, training=training)
        h = self.hidden_activation(self.last_fc(h))
        h = self.reshape_to_deconv(h)
        h = self.apply_forward(h, self.deconv_layers, self.deconv_norm_layers,
                               use_batch_norm=self.batch_norm_deconv, training=training)
        first_output = self.first_deconv_output(h)
        second_output = self.second_deconv_output(h)
        return first_output, second_output

    def apply_forward(self, input, hidden_layers, norm_layers,
                      use_batch_norm=False, training=True):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h, training=training)
            h = self.hidden_activation(h)
        return h

class DCNN(TwoHeadDCNN):
    def __init__(self, *args, **kwargs):
        """
        Make sure that 'name' is a keyword argument
        I don't know how to determine the argument from args
        (maybe position number would work?)
        """
        if 'name' not in kwargs:
            kwargs['name'] = 'dcnn' # tf converts names to snakecase for some reason
        super(DCNN, self).__init__(*args, **kwargs)

    def call(self, x, training=True):
        return super().call(x, training=training)[0]
