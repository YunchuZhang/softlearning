import tensorflow as tf
#import ipdb 
#st = ipdb.set_trace

from softlearning.utils.keras import create_picklable_keras_model


def feedforward_model(input_shapes,
                      output_size,
                      hidden_layer_sizes,
                      activation='relu',
                      output_activation='linear',
                      preprocessors=None,
                      name='feedforward_model',
                      *args,
                      **kwargs):
    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]
    # st()
    # TODO: Change this preprocessor stuff so all inputs are passed to the
    # same preprocessor, but not the action. Need to consider compatability
    # issues with convnet_preprocessor
    if preprocessors is None:
        preprocessors = (None, ) * len(inputs)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_ in zip(preprocessors, inputs)
    ]

    preprocessed_inputs = [tf.keras.layers.Flatten()(i) if len(i.shape) > 2 else i  for i in preprocessed_inputs]
    
    concatenated = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(preprocessed_inputs)

    out = concatenated
    for units in hidden_layer_sizes:
        out = tf.keras.layers.Dense(
            units, *args, activation=activation, **kwargs
        )(out)

    out = tf.keras.layers.Dense(
        output_size, *args, activation=output_activation, **kwargs
    )(out)

    model = create_picklable_keras_model(inputs, out, name)

    return model
