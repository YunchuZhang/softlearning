from nets.Net import Net
import constants as const
import utils_map as utils
import tensorflow as tf
import numpy as np
from munch import Munch
from tensorflow import summary as summ


class ConvfcNet:
    def __init__(self, node_input, predict_dim=None, \
                       scopename="graphnet", layers=[32, 32], is_normalize=False):
        """
        node_input: [batch_size, dims]
                    the first node_static_dim values are static features and will be
                    treated as const and will not be predicted
        default for is_normalize is layer_normalization
          
        """
        self.predict_dim = predict_dim
        self.layers = layers
        self.scopename = scopename
        self.is_normalize = is_normalize
    
    def predict_one_step(self, node_input, is_training=None, global_input = None, reuse=True):
        with tf.variable_scope(self.scopename, reuse=reuse):
            input = node_input
            if global_input is not None:
                input = tf.concat([node_input, global_input], 1)
            for layer_id, layer in enumerate(self.layers):
                #input = utils.nets.slim2_fc(input, layer, activation_fn=tf.nn.relu,\
                #                            is_normalize=self.is_normalize)
                input = utils.nets.slim2_fc(input, layer, is_normalize=self.is_normalize,\
                    activation_fn=tf.nn.relu, name=f"fc_{layer_id}")

            out = utils.nets.slim2_fc(input, self.predict_dim, activation_fn=None)

            return out
 
