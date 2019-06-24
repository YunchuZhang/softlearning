from nets.Net import Net
import constants as const
import utils_map as utils
import tensorflow as tf
import numpy as np
from munch import Munch
from tensorflow import summary as summ


class GraphNet:
    def __init__(self, node_input, node_predict_dim=None, node_static_dim = 0, \
                       edge_input=None, edge_static_dim=0,
                       global_input=None, global_static_dim=0,
                       edge_feat_dim=10, scopename="graphnet", reuse=False):
        """
        node_input: [batch_size, Nnodes, Dnodes]
                    the first node_static_dim values are static features and will be
                    treated as const and will not be predicted
        edge_input: if edge input is none (no edges information), we will automatically
                    assume there are [Nnodesx(Nnodes-1)] edges and they  will be initialized
                    with one dimensional vector with zero values
                    (edge_sender, edge_receiver, edge_value) 
          
        """
        self.edge_feat_dim = edge_feat_dim
        bs, Nnodes, Dnodes = node_input.get_shape()
        self.Nnodes = Nnodes
        if node_predict_dim:
            self.node_predict_dim = node_predict_dim
        else:
            self.node_predict_dim = Dnodes.value - node_static_dim
        self.scopename = scopename
        if not edge_input:
            Nedges = Nnodes.value * (Nnodes.value - 1)
            edge_sender = np.zeros((bs, Nedges, Nnodes.value))
            edge_receiver = np.zeros((bs, Nedges, Nnodes.value))
            edge_value = np.zeros((bs, Nedges, 1))
            edge_id = 0
            for i in range(Nnodes):
                for j in range(i+1, Nnodes):
                    edge_sender[:, edge_id, i] = 1
                    edge_receiver[:, edge_id, j] = 1
                    edge_id += 1
                    
                    edge_sender[:, edge_id, j] = 1
                    edge_receiver[:, edge_id, i] = 1
                     
                    edge_id += 1
            with tf.variable_scope(scopename, reuse=reuse):
                self.edge_sender = tf.constant(edge_sender, dtype=tf.float32)
                self.edge_receiver = tf.constant(edge_receiver, dtype=tf.float32)
                self.edge_value = tf.constant(edge_value, dtype=tf.float32)
    def predict_one_step(self, node_input, global_input, reuse=True, is_training=True):
        with tf.variable_scope(self.scopename, reuse=reuse):

            edge_out = self.phi_edge(self.edge_sender, self.edge_receiver, self.edge_value, node_input, is_training=is_training)

            node_input = self.rho_e_v(edge_out, self.edge_receiver, node_input, global_input)
            #if self.scopename=="agent_graphnet":
            #    import ipdb; ipdb.set_trace()
            node_out = self.phi_node(node_input)
            #node_out_pos = node_out[:,:,:3]
            #node_out_orn = tf.nn.l2_normalize(node_out[:,:,3:7], 2)
            #node_out_vel = node_out[:,:,7:]
            #node_out = tf.concat([node_out_pos, node_out_orn, node_out_vel], 2)
            return node_out
     
    def phi_edge(self, edge_sender, edge_receiver, edge_value, node_value, is_training):
        # batch_size x #edges x dim
        edge_input = tf.concat([tf.matmul(edge_sender, node_value), 
                     tf.matmul(edge_receiver, node_value), edge_value], 2)
        bs, Nedges, dim = edge_input.get_shape()
        
        edge_input = tf.reshape(edge_input, [-1, dim])
        out = utils.nets.slim2_fc(edge_input, 32, is_normalize=False, is_training=is_training)
        out = utils.nets.slim2_fc(out, self.edge_feat_dim, is_normalize=False)
        edge_out = tf.reshape(out, [bs, Nedges, self.edge_feat_dim]) 
   
        return edge_out

    def rho_e_v(self, edge_out, edge_receiver, node_value, global_input):
        # aggregate edges using sum
        _, Nnodes, _ = node_value.get_shape()
        
        edge_feat = tf.matmul(tf.transpose(edge_receiver, [0, 2, 1]), edge_out)
        #Nedges_per_node = tf.expand_dims(tf.reduce_sum(edge_receiver, 1), 2)
        #edge_feat = tf.divide(edge_feat, Nedges_per_node)
        
        global_input = tf.tile(tf.expand_dims(global_input, 1), [1, Nnodes, 1])
        #if self.scopename == "agent_graphnet":
        #       import ipdb; ipdb.set_trace()
        #       print("hello")
        
        return tf.concat([node_value, edge_feat, global_input], 2)
    
    def phi_node(self, node_out): 
        bs, Nnodes, dim = node_out.get_shape()
        
        node_out = tf.reshape(node_out, [-1, dim])
        out = utils.nets.slim2_fc(node_out, 32, is_normalize=False)
        out = utils.nets.slim2_fc(out, self.node_predict_dim, activation_fn=None)
        node_out = tf.reshape(out, [bs, Nnodes, self.node_predict_dim]) 
        return node_out
 
