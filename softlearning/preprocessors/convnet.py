import tensorflow as tf

from softlearning.models.feedforward import feedforward_model
from softlearning.utils.keras import create_picklable_keras_model

from softlearning.map3D.nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain
from softlearning.map3D import constants as const


def convnet_preprocessor(
        input_shapes,
        image_shape,
        output_size,
        conv_filters=(32, 32),
        conv_kernel_sizes=((5, 5), (5, 5)),
        pool_type='MaxPool2D',
        pool_sizes=((2, 2), (2, 2)),
        pool_strides=(2, 2),
        dense_hidden_layer_sizes=(64, 64),
        data_format='channels_last',
        name="convnet_preprocessor",
        make_picklable=True,
        *args,
        **kwargs):

    if data_format == 'channels_last':
        H, W, C = image_shape
    elif data_format == 'channels_first':
        C, H, W = image_shape

    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    concatenated_input = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(inputs)

    images_flat, input_raw = tf.keras.layers.Lambda(
        lambda x: [x[..., :H * W * C], x[..., H * W * C:]]
    )(concatenated_input)

    images = tf.keras.layers.Reshape(image_shape)(images_flat)

    conv_out = images
    for filters, kernel_size, pool_size, strides in zip(
            conv_filters, conv_kernel_sizes, pool_sizes, pool_strides):
        conv_out = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="SAME",
            activation=tf.nn.relu,
            *args,
            **kwargs
        )(conv_out)
        conv_out = getattr(tf.keras.layers, pool_type)(
            pool_size=pool_size, strides=strides
        )(conv_out)

    flattened = tf.keras.layers.Flatten()(conv_out)
    concatenated_output = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )([flattened, input_raw])

    output = (
        feedforward_model(
            input_shapes=(concatenated_output.shape[1:].as_list(), ),
            output_size=output_size,
            hidden_layer_sizes=dense_hidden_layer_sizes,
            activation='relu',
            output_activation='linear',
            *args,
            **kwargs
        )([concatenated_output])
        if dense_hidden_layer_sizes
        else concatenated_output)

    model = create_picklable_keras_model(inputs, out, name)

    return model

def convnet3d_preprocessor(
        input_shapes,
        output_size,
        conv_filters=(32, 32),
        conv_kernel_sizes=((5, 5), (5, 5)),
        conv_strides=(1, 1),
        pool_type='MaxPool3D',
        pool_sizes=((2, 2), (2, 2)),
        pool_strides=(2, 2),
        dense_hidden_layer_sizes=(64, 64),
        data_format='channels_last',
        name="convnet_preprocessor",
        make_picklable=True,
        *args,
        **kwargs):
    # if data_format == 'channels_last':
    #     H, W, C = image_shape
    # elif data_format == 'channels_first':
    #     C, H, W = image_shape
    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    conv_out = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(inputs)

    # images_flat, input_raw = tf.keras.layers.Lambda(
    #     lambda x: [x[..., :H * W * C], x[..., H * W * C:]]
    # )(concatenated_input)

    # images = tf.keras.layers.Reshape(image_shape)(images_flat)

    # conv_out = images
    for filters, kernel_size, stride in zip(
            conv_filters,
            conv_kernel_sizes,
            conv_strides):

        print("FILTER SIZE:", filters)

        conv_out = tf.keras.layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            padding="SAME",
            activation=tf.nn.relu,
            strides=stride,
            *args,
            **kwargs
        )(conv_out)
        #conv_out = getattr(tf.keras.layers, pool_type)(
        #    pool_size=pool_size, strides=pool_stride
        #)(conv_out)
        print("CONV OUT SHAPE:", conv_out.get_shape())
    flattened = tf.keras.layers.Flatten()(conv_out)
    # print(flattened)
    # concatenated_output = tf.keras.layers.Lambda(
    #     lambda x: tf.concat(x, axis=-1)
    # )([flattened, input_raw])

    output = (
        feedforward_model(
            input_shapes=(flattened.shape[1:].as_list(), ),
            output_size=output_size,
            hidden_layer_sizes=dense_hidden_layer_sizes,
            activation='relu',
            output_activation='linear',
            *args,
            **kwargs
        )([flattened])
        if dense_hidden_layer_sizes
        else flattened)

    model = create_picklable_keras_model(inputs, output, name=name)

    return model

#def map3D_preprocessor(
#        input_shapes,
#        output_size,
#        mapping_model=None,
#        data_pos={},
#        data_format='channels_last',
#        filters=None,
#        kernal_sizes=None,
#        conv_strides=None,
#        activation=tf.nn.relu,
#        pool_type=None,
#        pool_sizes=None,
#        pool_strides=None,
#        dense_hidden_layer_sizes=(64, 64),
#        name='map3D_preprocessor',
#        *args,
#        **kwargs):
#    inputs = [
#        tf.keras.layers.Input(shape=input_shape)
#        for input_shape in input_shapes
#    ]
#
#    #TODO: Need to change this so that the inputs are mapped to the right place
#    # also not sure if this is the best way of doing this
#    conv_out = mapping_model(inputs[data_pos['images']], inputs[data_pos['zmaps']], inputs[data_pos['cam_angles']])
#
#    for num_filters, kernal_size, conv_stride, pool_size, pool_stride in zip(
#            filters,
#            kernal_sizes,
#            conv_strides,
#            pool_sizes,
#            pool_strides):
#
#        conv_out = tf.keras.layers.Conv3D(
#            filters=num_filters,
#            kernal_size=kernal_size,
#            strides=stride,
#            activation=activation
#        )(conv_out)
#        # Get the Pool based on the pool name
#        conv_out = getattr(tf.keras.layers, pool_type)(
#            pool_size=pool_size, strides=pool_stride
#        )(conv_out)
#
#    flattened = tf.keras.layers.Flatten()(conv_out)
#
#    output = (
#        feedforward_model(
#            input_shapes=(flattened[1:].as_list(), ),
#            output_size=output_size,
#            hidden_layer_sizes=dense_hidden_layer_sizes,
#            activation='relu',
#            output_activation='linear',
#            *args,
#            **kwargs
#        )([flattened])
#        if dense_hidden_layer_sizes
#        else flattened)
#
#    model = create_picklable_keras_model(inputs, output, name=name)
#
#    return model
#
#
#def map3D_preprocessor_nonkeras(
#        name='map3D_preprocessor',
#        mapping_model=None,
#        *args,
#        **kwargs):
#
#    #TODO: Need to change this so that the inputs are mapped to the right place
#    # also not sure if this is the best way of doing this
#    # conv_out = mapping_model(inputs[data_pos['images']], inputs[data_pos['zmaps']], inputs[data_pos['cam_angles']])
#    # model = BulletPush3DTensor4_cotrain()
#    model = mapping_model
#    const.set_experiment("rl_new")
#    return model



if __name__ == "__main__":
    input_shape = [(32,32,32,16)]
    convnet3d_preprocessor(input_shape,128,conv_filters=(16,32,64,128,128),conv_kernel_sizes=(4,4,4,4,3),pool_sizes=(2,2,2,2,2),pool_strides=(2,2,2,2,2))
