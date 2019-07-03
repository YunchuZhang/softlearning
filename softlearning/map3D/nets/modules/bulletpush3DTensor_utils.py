import tensorflow as tf


def get_dict_state_t(frozen_states, t):
    out_frozen_state = dict()
    for key, value in frozen_states.items():
        out_frozen_state[key] = value[:, t, ...]

    return out_frozen_state


def concat_dict_states(frozen_states):
    out_frozen_states = dict()
    frozen_state_first = frozen_states[0]
    for key, value in frozen_state_first.items():
        out_frozen_states[key] = tf.stack([state[key] for state in frozen_states], 1)

    return out_frozen_states
