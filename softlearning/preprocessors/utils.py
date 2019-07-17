from copy import deepcopy
import ipdb
st = ipdb.set_trace

def get_convnet3d_preprocessor(observation_shape,
                             name='convnet_preprocessor',
                             **kwargs):
    from .convnet import convnet3d_preprocessor
    preprocessor = convnet3d_preprocessor(
        input_shapes=(observation_shape, ), name=name, **kwargs)

    return preprocessor

def get_convnet_preprocessor(observation_shape,
                             name='convnet_preprocessor',
                             **kwargs):
    from .convnet import convnet_preprocessor
    preprocessor = convnet_preprocessor(
        input_shapes=(observation_shape, ), name=name, **kwargs)

    return preprocessor


def get_map3D_preprocessor(observation_shape,
                           name='map3D_preprocessor',
                           **kwargs):
    from .convnet import map3D_preprocessor
    preprocessor = map3D_preprocessor(
        input_shapes=observation_shape, name=name, **kwargs
    )
    return preprocessor


def get_map3D_preprocessor_nonkeras(observation_shape,
                           name='map3D_preprocessor',
                           **kwargs):
    from .convnet import map3D_preprocessor_nonkeras
    preprocessor = map3D_preprocessor_nonkeras(name=name, **kwargs)
    return preprocessor

def get_feedforward_preprocessor(observation_shape,
                                 name='feedforward_preprocessor',
                                 **kwargs):
    from softlearning.models.feedforward import feedforward_model
    preprocessor = feedforward_model(
        input_shapes=(observation_shape, ), name=name, **kwargs)

    return preprocessor


PREPROCESSOR_FUNCTIONS = {
    'convnet_preprocessor': get_convnet_preprocessor,
    'convnet3d_preprocessor': get_convnet3d_preprocessor,
    'map3D_preprocessor': get_map3D_preprocessor,
    'map3D_preprocessor_nonkeras' : get_map3D_preprocessor_nonkeras,
    'feedforward_preprocessor': get_feedforward_preprocessor,
    None: lambda *args, **kwargs: None
}


def get_preprocessor_from_params(env, preprocessor_params, *args, **kwargs):
    if preprocessor_params is None:
        return None
    # st()
    preprocessor_type = preprocessor_params.get('type', None)
    preprocessor_input_shape = preprocessor_params.get('input_shape', None)
    preprocessor_kwargs = deepcopy(preprocessor_params.get('kwargs', {}))

    if preprocessor_type is None:
        return None

    preprocessor = PREPROCESSOR_FUNCTIONS[
        preprocessor_type](
            preprocessor_input_shape,
            *args,
            **preprocessor_kwargs,
            **kwargs)

    return preprocessor


def get_preprocessor_from_variant(variant, env, *args, **kwargs):
    preprocessor_params = variant['preprocessor_params']
    return get_preprocessor_from_params(
        env, preprocessor_params, *args, **kwargs)
