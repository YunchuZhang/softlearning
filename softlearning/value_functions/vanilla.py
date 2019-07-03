from softlearning.models.feedforward import feedforward_model
import ipdb 
st = ipdb.set_trace

def create_feedforward_Q_function(observation_shape,
                                  action_shape,
                                  *args,
                                  observation_preprocessor=None,
                                  name='feedforward_Q',
                                  **kwargs):
    # st()
    # if len(observation_shape) == 1:
    #   input_shapes = (observation_shape, action_shape)
    # else:
    input_shapes = (*observation_shape, action_shape)


    preprocessors = (observation_preprocessor, None)
    return feedforward_model(
        input_shapes,
        *args,
        output_size=1,
        preprocessors=preprocessors,
        name=name,
        **kwargs)


def create_feedforward_V_function(observation_shape,
                                  *args,
                                  observation_preprocessor=None,
                                  name='feedforward_V',
                                  **kwargs):
    input_shapes = observation_shape
    preprocessors = (observation_preprocessor, None)
    return feedforward_model(
        input_shapes,
        *args,
        output_size=1,
        preprocessors=preprocessors,
        **kwargs)
