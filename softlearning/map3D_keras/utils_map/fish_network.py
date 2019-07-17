import constants as const

import logging
from os.path import join

import numpy as np
import tensorflow as tf

#from .vendor import resnets
#from .utils import Config

from collections import OrderedDict
import math

from ipdb import set_trace as st


class MultiCameraGQN:
    """
    Keeps the network architecture for the multi-camera CNN. Each camera has its own
    CNN (sometimes called column). These CNNs are then joined/concatenated and passed
    through a few more layers.
    """

    class Config:
        # The network architecture for camera columns (see gpr.vision.networks.make_cnn)
        camera_network = [['Cvalid', 2, 2, 32 * 8],
                          ['RELU'],
                          ['sR', 2, 2, 32 * 8]]
        camera_post_network = [
                       ['sR', 3, 1, 32 * 8],
                       ['Cvalid', 1, 1, 32 * 8],
                       ['RELU'],
                       ['P', 16, 16]] #used to be P, 50, 50
        camera_cam_posrot_network = [['FC', 128]]
        """
        gqn_network = [['FC', 128],
                       ['FC', 128],
                       ['DC', 5, 2, 32],
                       ['DC', 5, 2, 16],
                       ['DC', 5, 2, 3]]
        """
        gqn_network =[['convLSTM', 512, 3, 1, 64]]
        # The network architecture post column concatenation (see gpr.vision.networks.make_cnn)
        shared_network = [['FC', 128]]
        # Whether to share weights between camera columns or not
        tie_camera_cnns = True
        gqn_loss = True
        gqn_output_size = None
        gqn_input_size = None
        # Weight decay regularization
        weight_decay = 0.001
        max_num_gqn_context = const.NUM_VIEWS
        merge_type="concat"
        last_output_exampt_list = ['cam_posrot_vision_cam_top', 'cam_posrot_vision_cam_right', 'cam_posrot_vision_cam_left']

    def __init__(self, inputs, output_dim: int,
                 config: Config = Config(), scope='multi_camera_cnn',
                 reuse=False, is_training=None, is_predict_cam_posrot=False):
        """
        Creates a multi-camera CNN, where the inputs is a list of image batch tensors.
        Exposes a few instance variables:
        .outputs : output of network
        .is_training : boolean placeholder for whether network is training or not;
            this regulates whether dropout is turned on.
        .camera_cnns : output of camera columns
        .camera_cnns_concat : concatenated camera column output
        :param inputs: list of input image tensors (NHWC-format), one for each camera
        :param output_dim: output dimensionality of network
        :param config: instance of MultiCameraCNN.Config
        :param scope: tensorflow scope
        :param reuse: whether to reuse variable scopes
        :param is_training: true if model is training, if None, a placeholder is created
        """
        self.config = config
        self.parent_scope = tf.get_variable_scope().name
        self.scope = scope
        self.pixel_std = tf.constant(2.0)
        self.max_num_gqn_context = config.max_num_gqn_context
        self.num_predict_gqn_context = len(inputs['images']) - config.max_num_gqn_context
        if config.gqn_output_size == None:
            self.gqn_output_size = inputs['images'][0].get_shape()[1]
        else:
            self.gqn_output_size = config.gqn_output_size

        self.layer_map = OrderedDict()
        self.is_predict_cam_posrot = is_predict_cam_posrot
        inputs_image = inputs['images'][:self.max_num_gqn_context]
        inputs_cam_posrot = inputs[f'cam_posrot'][:self.max_num_gqn_context]
        predict_cam_posrot = inputs[f'cam_posrot'][self.max_num_gqn_context:]

        self.predict_cam_posrot = predict_cam_posrot
        predict_image = inputs['images'][self.max_num_gqn_context:]
        predict_undistorted_image = inputs['undistorted_images'][self.max_num_gqn_context:]

        posrot_dim = inputs[f'cam_posrot'][0].shape[1].value
        if self.is_predict_cam_posrot:
            output_dim -= posrot_dim * self.max_num_gqn_context
        for input_ in inputs_image +  inputs.get('cam_posrot', []):
            assert input_.dtype == tf.float32, f'bad input {input_}'

        self.unsup_loss = dict()
        self.extra_images = dict() # for summary
        with tf.variable_scope(scope, reuse=reuse):
            if is_training is None:
                self.is_training = tf.placeholder(tf.bool, name='is_training')
            else:
                self.is_training = is_training

            logging.debug(f"Creating MultiCameraCNN with layers: \n"
                          f"{config.camera_network}\n"
                          f"{config.shared_network}")


            # stacks inputs together
            stacked_inputs_image = tf.concat(inputs_image, axis=0)

            if config.gqn_input_size:
                stacked_inputs_image = tf.image.resize_images(stacked_inputs_image, [config.gqn_input_size, config.gqn_input_size])

            stacked_inputs_cam_posrot = tf.concat(inputs_cam_posrot, axis=0)
            #self.pre_camera_cnns = []
            self.camera_cnns = []
            #self.camera_cam_posrot = []
            cnn_scope = 'camera_cnn'
            pre_camera_cnn, layer_map = make_cnn(stacked_inputs_image, config.camera_network,
                                             scope=cnn_scope, reuse=reuse,
                                             is_training=self.is_training,
                                             weight_decay=config.weight_decay)
            self.layer_map.update(layer_map)
            self.pre_camera_cnns = tf.split(pre_camera_cnn, self.max_num_gqn_context, axis=0)
            if self.is_predict_cam_posrot:
                _, h_, w_, c_ = pre_camera_cnn.shape
                pre_camera_cnn_flat =  tf.reshape(pre_camera_cnn, [-1, h_ * w_ * c_])
                camera_posrot_cnn, layer_map = make_cnn(pre_camera_cnn_flat, config.camera_cam_posrot_network,
                               scope=cnn_scope + "/cam_posrot", reuse=reuse, output_dim=posrot_dim,
                               is_training=self.is_training,
                               weight_decay=config.weight_decay)
                self.camera_cam_posrot = tf.split(camera_posrot_cnn, self.max_num_gqn_context, axis=0)
            # concate camera calibrations
            _, h, w, c = pre_camera_cnn.get_shape()
            cam_posrot = tf.expand_dims(tf.expand_dims(stacked_inputs_cam_posrot, 1), 1)
            cam_posrot_tiled = tf.tile(cam_posrot, [1, h, w, 1])
            pre_camera_cnn= tf.concat([pre_camera_cnn, cam_posrot_tiled], 3)

            camera_cnn, layer_map = make_cnn(pre_camera_cnn, config.camera_post_network,
                                  scope=cnn_scope + "/post_cam", reuse=reuse,
                                  is_training=self.is_training,
                                  weight_decay=config.weight_decay)

            self.layer_map.update(layer_map)
            self.camera_cnns = tf.split(camera_cnn, self.max_num_gqn_context, axis=0)
            
            print('WARNING: exiting out of the constructor early')
            return
        

            # gqn decoder
            if config.gqn_loss:
                self.gqn_outputs= []
                self.resize_input_images = []
                self.gqn_loss = 0
                self.kl_loss = 0
                #input_view_mask = np.ones((num_views, num_views))
                #for view_id in range(num_views):
                #    input_view_mask[view_id, view_id] = 0
                # batch_size * num_view * num_view
                #input_view_mask_tf = tf.tile(tf.expand_dims(tf.constant(input_view_mask, dtype=tf.float32), 0), [tf.shape(self.camera_cnns[0])[0], 1, 1])
                input_view_mask_tf = tf.cast(tf.greater(tf.random_uniform([tf.shape(self.camera_cnns[0])[0], 1, self.max_num_gqn_context]), 0.5), dtype=tf.float32)
                self.mask = input_view_mask_tf
                stacked_camera_cnn = tf.concat([tf.expand_dims(camera_cnn, axis=1) for camera_cnn in self.camera_cnns], axis=1)
                bs, nc, h, w, c = stacked_camera_cnn.get_shape()

                # batch x num_cameras x h*w*c
                stacked_camera_cnn = tf.reshape(stacked_camera_cnn, [tf.shape(stacked_camera_cnn)[0], nc.value, h.value * w.value * c.value] )
                # num_exp_per_batch (1) x batch x dimension
                add_minus_one = tf.reshape(tf.transpose(tf.matmul(input_view_mask_tf, stacked_camera_cnn), [1, 0, 2]), [-1, h.value, w.value, c.value])
                # [batch * num_output] x h x w x c
                add_minus_one = tf.reshape(tf.tile(tf.expand_dims(add_minus_one, 0), [self.num_predict_gqn_context, 1, 1, 1, 1]), [-1, h.value, w.value, c.value])
                _, posrot_dim = predict_cam_posrot[0].get_shape()
                # [batch * num_output] x dim
                predict_cam_posrot = tf.reshape(tf.concat([tf.expand_dims(cam_posrot, 0) for cam_posrot in predict_cam_posrot], 0), [-1, posrot_dim])
                cam_posrot = tf.expand_dims(tf.expand_dims(predict_cam_posrot, 1), 1)

                _, ho, wo, co = predict_image[0].get_shape()
                predict_image = tf.reshape(tf.concat([tf.expand_dims(img, 0) for img in predict_image], 0), [-1, ho, wo, co])
                resize_images = tf.image.resize_images(predict_image, [self.gqn_output_size, self.gqn_output_size])

                use_convlstm = False
                for layer_desc in config.gqn_network:
                    if layer_desc[0] == "convLSTM":
                        use_convlstm = True
                if use_convlstm:
                    gqn_output, extra = make_lstmConv(add_minus_one, cam_posrot, resize_images,
                                config.gqn_network,
                                scope='gqn_net', reuse=reuse,
                                is_training=self.is_training,
                                weight_decay=config.weight_decay)
                    self.kl_loss += extra['kl_loss']
                    self.gqn_output = gqn_output
                    _, ho, wo, co = predict_undistorted_image[0].get_shape()
                    predict_undistorted_image = tf.reshape(tf.concat([tf.expand_dims(img, 0) for img in predict_undistorted_image], 0), [-1, ho, wo, co])
                    undistorted_image = tf.image.resize_images(predict_undistorted_image, [self.gqn_output_size, self.gqn_output_size])
                    self.resize_input_images = undistorted_image
                    self.gqn_loss +=  tf.reduce_mean(negative_pixel_likelihood(undistorted_image, gqn_output, tf.pow(self.pixel_std, 2)))
                    # make images

                    stacked_inputs_image = tf.split(stacked_inputs_image, self.max_num_gqn_context, axis=0)
                    predict_undistorted_image = tf.split(predict_undistorted_image, self.num_predict_gqn_context, axis=0)
                    gqn_output = tf.split(self.gqn_output, self.num_predict_gqn_context, axis=0)

                    pad = -0.5 * tf.ones((self.gqn_output_size, 3, co))
                    for sample_id in range(1,3):
                        _, h, w, c = inputs_image[0].get_shape()
                        print_input_images = tf.concat([img[sample_id, :, :, :] * self.mask[sample_id, 0, img_id]
                                                        for img_id, img in enumerate(stacked_inputs_image)], 1)
                        print_gt_predict_images = tf.concat([img[sample_id, :, :, :]
                                                        for img in predict_undistorted_image], 1)
                        print_predict_images = tf.concat([img[sample_id, :, :, :]
                                                        for img in gqn_output], 1)


                        self.extra_images[f'images_gqn_camid_{sample_id}'] =  tf.clip_by_value( tf.concat([print_input_images, pad, print_gt_predict_images,
                                                                                                           pad, print_predict_images], 1) + 0.5, 0, 1)

                self.unsup_loss['gqn_loss'] = self.gqn_loss
                self.unsup_loss['kl_loss'] = self.kl_loss


            # flatten everything
            for cam_id in range(len(self.camera_cnns)):
                _, h_, w_, c_ = self.camera_cnns[cam_id].shape
                self.camera_cnns[cam_id] = tf.reshape(self.camera_cnns[cam_id], [-1, h_ * w_ * c_])
            if config.merge_type == "concat":
                print("============= concat merge ====================")
                self.camera_cnns_concat = tf.concat(self.camera_cnns, axis=1)
            elif config.merge_type == "add":
                print("============= add merge ====================")
                self.camera_cnns_concat = tf.add_n(self.camera_cnns)
            else:
                assert(1==2), "invalid merge type"


            self.layer_map[f'{scope}/camera_cnns_concat'] = self.camera_cnns_concat

            self.outputs, layer_map = make_cnn(self.camera_cnns_concat, config.shared_network,
                                               scope='shared_net', reuse=reuse,
                                               output_dim=output_dim, is_training=self.is_training,
                                               weight_decay=config.weight_decay)


            # create outputs
            # regular output from last layer
            output_last = []
            output_manual_add = []
            for output_id, output_name in enumerate(inputs['label_names']):
                if output_name not in config.last_output_exampt_list:
                    output_last.append((output_name, inputs['target_shapes'][output_id]))
                else:
                    output_manual_add.append(output_name)

            output_last_target_shape = [shape for name, shape in output_last]

            print('stcuk here, line 1909')
            print(self.outputs)
            outputs = dict([
                (output_last[tensor_id][0], tensor) for tensor_id, tensor in
                enumerate(tf.split(self.outputs, output_last_target_shape, axis=-1))
            ])

            camera_name_order = dict((name, id_) for id_, name in enumerate(inputs['input_names']))

            # add camera pos rotation outputs
            if self.is_predict_cam_posrot:
                for output_name in output_manual_add:
                    if output_name.startswith('cam_posrot_vision_'):
                        camera_id = camera_name_order[output_name[11:]]
                        outputs[output_name] = self.camera_cam_posrot[camera_id]
            self.outputs = outputs
            self.layer_map.update(layer_map)

    @property
    def trainable_vars(self):
        """List of trainable tensorflow variables for network."""
        scope = join(self.parent_scope, self.scope)
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def _parse_layer_params(layer_desc_, num_expected):
    """Extracts parameters from network description layer and raises if there are issues."""
    layer_type_, layer_params = layer_desc_[0], layer_desc_[1:]
    if len(layer_params) != num_expected:
        raise ValueError(f"Expected {num_expected} parameters for layer {layer_type_} but "
                         f"received {len(layer_params)}: {layer_params}")
    return layer_params

def negative_pixel_likelihood(x, mu_q, pixel_var):
    _, h, w, c = mu_q.get_shape()
    k = float(h.value * w.value * c.value)
    diff = x - mu_q
    return 0.5 * (k * math.log(2 * math.pi) + tf.reduce_sum(tf.log(pixel_var) + diff * diff / pixel_var, axis=[1, 2, 3]))

def compute_kl_loss(mu_q, ln_var_q, mu_p, ln_var_p):

    ln_det_q = tf.reduce_sum(ln_var_q, axis=[1, 2, 3])
    ln_det_p = tf.reduce_sum(ln_var_p, axis=[1, 2, 3])
    var_p = tf.exp(ln_var_p)
    var_q = tf.exp(ln_var_q)
    tr_qp = tf.reduce_sum(var_q / var_p, axis=[1, 2, 3])
    _, h, w, c = mu_q.get_shape()
    k = float(h.value * w.value * c.value)
    diff = mu_p - mu_q
    term2 = tf.reduce_sum(diff * diff / var_p, axis=[1, 2, 3])
    return 0.5 * (tr_qp + term2 - k + ln_det_p - ln_det_q)


def make_lstmConv(inputs, cam_posrot, output_image, network_description,
                  stochastic=True, weight_decay=0.0, is_training=True, scope='', reuse=False, output_debug=False):
    is_convLSTM_start = False
    out = inputs
    extra = dict()
    with tf.variable_scope(scope, reuse=reuse):
        for i, layer_desc in enumerate(network_description):
            layer_type = layer_desc[0]

            if layer_type == 'convLSTM':

                if not is_convLSTM_start:
                    '''
                    lstm_size: number of channels used in lstm state/cell
                    n_filters: number of channels in output
                    number_steps: number of lstm steps
                    code_size: #channels in latent code
                    '''
                    lstm_size, n_filters, number_steps, code_size = _parse_layer_params(layer_desc, 4)
                    input_shape = tf.shape(out)
                    _, h_in, w_in, c_in = out.get_shape()
                    _, h, w, c = output_image.get_shape()
                    sh = int(h)//4
                    sw = int(w)//4

                    if cam_posrot is not None:
                        lstm_v = tf.layers.conv2d_transpose(
                            cam_posrot,
                            12,
                            sh,
                            strides=(sh, sh),
                            use_bias=False,
                            padding = "same",
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_v{i}')
                    else:
                        bs = inputs.shape.as_list()[0]
                        lstm_v = tf.zeros((bs, sh, sh, 0))

                        #if int(h_in) != sh: # use convolution to change its size

                    #set to True if input is B x 1 x 1 x C
                    if h_in == 1:
                        lstm_r = tf.layers.conv2d_transpose(
                            out,
                            int(c_in),
                            sh,
                            strides=(sh, sh),
                            use_bias=False,
                            padding = "same",
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_r{i}')
                    else:
                        assert h_in == sh
                        lstm_r = out

                    lstm_h = tf.zeros([input_shape[0], int(h)//4, int(w)//4, lstm_size], dtype=tf.float32)
                    lstm_c = tf.zeros([input_shape[0], int(h)//4, int(w)//4, lstm_size], dtype=tf.float32)
                    lstm_u = tf.zeros(tf.shape(output_image), dtype=tf.float32)


                    g_mu_var = []
                    lstm_h_g = []
                    lstm_u_g = []
                    lstm_reuse=reuse
                    for step_id in range(number_steps):
                        if step_id > 0: lstm_reuse=True

                        lstm_input = tf.concat([lstm_h, lstm_r, lstm_v], 3)
                        lstm_u_g.append(lstm_u)
                        lstm_h_g.append(lstm_h)
                        if stochastic:
                            lstm_input_mu_var = tf.layers.conv2d(
                                lstm_h,
                                code_size * 2,
                                5,
                                strides=1,
                                padding = "same",
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                name=f'convLSTM_mu{i}',
                                reuse=lstm_reuse)

                            lstm_input_mu, lstm_input_log_var = tf.split(lstm_input_mu_var, 2, axis=3)
                            g_mu_var.append((lstm_input_mu, lstm_input_log_var))
                            epsilon = tf.random_normal(tf.shape(lstm_input_mu))
                            lstm_z = lstm_input_mu + tf.exp(0.5 * lstm_input_log_var) * epsilon
                            lstm_input = tf.concat([lstm_input, lstm_z], 3)


                        lstm_input_all = tf.layers.conv2d(
                            lstm_input,
                            lstm_size * 4,
                            5,
                            strides=1,
                            padding = "same",
                            activation = None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_input_c{i}',
                            reuse=lstm_reuse)
                        lstm_input_c, lstm_input_i1, lstm_input_i2, lstm_out = tf.split(lstm_input_all, 4, axis=3)
                        lstm_input_c = tf.nn.sigmoid(lstm_input_c)
                        lstm_input_i1 = tf.nn.sigmoid(lstm_input_i1)
                        lstm_input_i2 = tf.nn.tanh(lstm_input_i2)
                        lstm_out = tf.nn.sigmoid(lstm_out)

                        lstm_input = tf.multiply(lstm_input_i1, lstm_input_i2)
                        lstm_c = tf.multiply(lstm_c, lstm_input_c) + lstm_input
                        lstm_h = tf.multiply(tf.nn.tanh(lstm_c), lstm_out)

                        lstm_final_out = tf.layers.conv2d_transpose(
                            lstm_h,
                            n_filters,
                            4,
                            strides=(4, 4),
                            padding = "same",
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_o_{i}',
                            reuse=lstm_reuse)

                        lstm_u = lstm_u + lstm_final_out
                    out = lstm_u

                    out = tf.layers.conv2d(
                            out,
                            n_filters,
                            1,
                            strides=(1, 1),
                            padding = "same",
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_final_{i}',
                            reuse=reuse)

                    e_mu_var = []
                    if stochastic:
                        lstm_x_q_e = tf.layers.conv2d(
                            output_image,
                            16,
                            5,
                            strides=4,
                            padding = "same",
                            activation=tf.nn.sigmoid,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_x_q_e_{i}',
                            reuse=reuse)
                        lstm_h_e = tf.zeros([input_shape[0], int(h)//4, int(w)//4, code_size], dtype=tf.float32)
                        lstm_c_e = tf.zeros([input_shape[0], int(h)//4, int(w)//4, code_size], dtype=tf.float32)

                        for step_id in range(number_steps):

                            lstm_e_reuse = reuse or step_id > 0
                            lstm_input_mu_var_e = tf.layers.conv2d(
                                lstm_h_e,
                                code_size * 2,
                                5,
                                strides=1,
                                padding = "same",
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                name=f'convLSTM_mu_e{i}',
                                reuse=lstm_e_reuse)
                            lstm_input_mu_e, lstm_input_log_var_e = tf.split(lstm_input_mu_var_e, 2, axis=3)

                            e_mu_var.append((lstm_input_mu_e, lstm_input_log_var_e))
                            if step_id == number_steps - 1:
                                break

                            lstm_u_e = tf.layers.conv2d(
                               lstm_u_g[step_id],
                               16,
                               5,
                               strides=4,
                               padding = "same",
                               activation=tf.nn.sigmoid,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                               name=f'convLSTM_u_e_{i}',
                               reuse=lstm_e_reuse)
                            lstm_input_e = tf.concat([lstm_h_e, lstm_h_g[step_id], lstm_u_e, lstm_r, lstm_v, lstm_x_q_e], axis=3)

                            lstm_input_all = tf.layers.conv2d(
                                lstm_input_e,
                                code_size * 4,
                                5,
                                strides=1,
                                padding = "same",
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                name=f'convLSTM_input_c_e{i}',
                                reuse=lstm_e_reuse)

                            lstm_input_c_e, lstm_input_i1_e, lstm_input_i2_e, lstm_out_e = tf.split(lstm_input_all, 4, axis=3)
                            lstm_input_c_e = tf.nn.sigmoid(lstm_input_c_e)
                            lstm_input_i1_e = tf.nn.sigmoid(lstm_input_i1_e)
                            lstm_input_i2_e = tf.nn.tanh(lstm_input_i2_e)
                            lstm_out_e = tf.nn.sigmoid(lstm_out_e)

                            lstm_input_e = tf.multiply(lstm_input_i1_e, lstm_input_i2_e)
                            lstm_c_e = tf.multiply(lstm_c_e, lstm_input_c_e) + lstm_input_e
                            lstm_h_e = tf.multiply(tf.nn.tanh(lstm_c_e), lstm_out_e)

                    kl_loss = 0
                    for layer_id, g_mu_var_ in enumerate(g_mu_var):
                        kl_loss += tf.reduce_mean(compute_kl_loss(
                            e_mu_var[layer_id][0],
                            e_mu_var[layer_id][1],
                            g_mu_var_[0],
                            g_mu_var_[1]
                        ))
                        
                    extra['kl_loss'] = kl_loss
            else:
                raise ValueError(f"Unknown layer type '{layer_type}' with params {layer_desc[1:]}")


    return out, extra

def make_deconv2(inputs, output_image_template, network_description, weight_decay=0.0, is_training=True, scope='', reuse=False, output_debug=False):

    _, s_h, s_w, c = output_image_template.get_shape()
    out = inputs
    """
    def _parse_layer_params(layer_desc_, num_expected):
    """
    """Extracts parameters from network description layer and raises if there are issues."""
    """
        layer_type_, layer_params = layer_desc_[0], layer_desc[1:]
        if len(layer_params) != num_expected:
            raise ValueError(f"Expected {num_expected} parameters for layer {layer_type_} but "
                             f"received {len(layer_params)}: {layer_params}")
        return layer_params
    """

    # calculate how many deconv layers, location for the first deconv
    # output image: batch x 200 x 200 x 3
    with tf.variable_scope(scope, reuse=reuse):
        for i, layer_desc in enumerate(network_description):
            layer_type = layer_desc[0]
            if layer_type == 'DC':
                kernel_size, strides, n_filters = _parse_layer_params(layer_desc, 3)

                out = tf.layers.conv2d_transpose(
                    out,
                    n_filters,
                    kernel_size,
                    strides=(strides, strides),
                    padding = "same",
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'deconv{i}',
                    reuse=reuse)
            elif layer_type == 'ATT':
                _, h, w, c = out.get_shape()
                query_conv = tf.layers.conv2d(
                    out, int(c)//8, 1,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'att_query{i}',
                    reuse=reuse)
                key_conv = tf.layers.conv2d(
                    out, int(c)//8, 1,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'att_key{i}',
                    reuse=reuse)
                value_conv = tf.layers.conv2d(
                    out, int(c), 1,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'att_value{i}',
                    reuse=reuse)
                gamma = tf.Variable(0, name=f'gamma{i}', dtype=tf.float32)

                inner = tf.reduce_sum(tf.multiply(tf.reshape(query_conv, [-1, h*w, int(c)//8]), tf.reshape(key_conv, [-1, h*w, int(c)//8])), 2)
                attention = tf.reshape(tf.nn.softmax(inner, 1), [-1, h, w, 1])

                out = attention * value_conv * gamma + out

            elif layer_type == 'BN':
                out = tf.layers.batch_normalization(out,
                                                    training=is_training,
                                                    fused=True, reuse=reuse,
                                                    name=f'batchnorm{i}')
            elif layer_type == 'RELU':
                out = tf.nn.relu(out)
            elif layer_type == 'TANH':
                out = tf.nn.tanh(out)
            elif layer_type == 'C':
                kernel_size, strides, n_filters = _parse_layer_params(layer_desc, 3)
                out = tf.layers.conv2d(
                    out, n_filters, kernel_size,
                    strides=strides,
                    padding="same",
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'conv{i}',
                    reuse=reuse)
            elif layer_type == 'FC':
                n_units, = _parse_layer_params(layer_desc, 1)
                out = tf.layers.dense(
                    out, n_units,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'dense{i}',
                    reuse=reuse)
            else:
                raise ValueError(f"Unknown layer type '{layer_type}' with params {layer_desc[1:]}")
    return out



def make_cnn(inputs, network_description, *,
             output_dim=None, weight_decay=0.0, is_training=True, scope='', reuse=False,
             custom_getter=None):
    """
    Creates a layered convolutional neural network (CNN), based on the desired network
    description. The network description is a list of tuples, with each tuple describing
    the layer type and parameters. Available layers are:
        C (kernel_size, strides, n_filters): conv2d layer
        P (pool_size, strides): max-pooling layer
        FLAT (): flattening layer
        FC (n_units): fully-connected layer
        D (keep_prob): dropout layer
        BN (): batch-normalization
        SSM (): spatial softmax
        R (block_type, strides, n_blocks, filters): resnet layer
    :param inputs: input image tensor
    :param network_description: network description (see above)
    :param output_dim: output dimension of network (if it's none, will just take output of
        last layer)
    :param weight_decay: weight regularization
    :param is_training: whether network is training or not (for dropout)
    :param scope: tensorflow variable scope
    :param reuse: share variables
    :return: outputs - the last layer of the network, and layer_map: a dictionary of all
             intermediate layers of the network
    """
    out = inputs

    with tf.variable_scope(scope, reuse=reuse, custom_getter=custom_getter):
        layer_map = OrderedDict()
        for i, layer_desc in enumerate(network_description):
            layer_type = layer_desc[0]
            name = None
            if layer_type == 'C':
                kernel_size, strides, n_filters = _parse_layer_params(layer_desc, 3)
                name = f'conv{i}'
                out = tf.layers.conv2d(
                    out, n_filters, kernel_size,
                    strides=strides,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=name,
                    reuse=reuse)
            elif layer_type == 'Cvalid':
                kernel_size, strides, n_filters = _parse_layer_params(layer_desc, 3)
                out = tf.layers.conv2d(
                    out, n_filters, kernel_size,
                    strides=strides,
                    padding='valid',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'conv_valid{i}',
                    reuse=reuse)
            elif layer_type == 'Csame':
                kernel_size, strides, n_filters = _parse_layer_params(layer_desc, 3)
                out = tf.layers.conv2d(
                    out, n_filters, kernel_size,
                    strides=strides,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'conv_same{i}',
                    reuse=reuse)
            elif layer_type == 'RELU':
                out = tf.nn.relu(out)
            elif layer_type == 'TANH':
                out = tf.nn.tanh(out)
            elif layer_type == 'P':
                pool_size, strides = _parse_layer_params(layer_desc, 2)
                out = tf.layers.average_pooling2d(out, pool_size, strides, name=f'pool{i}')
            elif layer_type == 'FLAT':
                out = tf.contrib.layers.flatten(out)
            elif layer_type == 'FC':
                n_units, = _parse_layer_params(layer_desc, 1)
                name = f'dense{i}'
                out = tf.layers.dense(
                    out, n_units,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=name,
                    reuse=reuse)
            elif layer_type == 'LINEAR':
                n_units, = _parse_layer_params(layer_desc, 1)
                name = f'dense{i}'
                out = tf.layers.dense(
                    out, n_units,
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=name,
                    reuse=reuse)
            elif layer_type == 'D':
                keep_prob, = _parse_layer_params(layer_desc, 1)
                out = tf.contrib.layers.dropout(out,
                                                keep_prob=keep_prob,
                                                is_training=is_training)
            elif layer_type == 'BN':
                out = tf.layers.batch_normalization(out,
                                                    training=is_training,
                                                    fused=True, reuse=reuse,
                                                    name=f'batchnorm{i}')
            elif layer_type in ('SSM', 'SMM'):
                # accept "SMM" to be backwards compatible with old typo
                if layer_type == 'SMM':
                    print("WARNING: The layer type SMM has been renamed SSM")
                out = spatial_softmax(out)
            elif layer_type == 'sR': # resnet that shrinks half in block size

                kernel, stride, c = _parse_layer_params(layer_desc, 3)
                residual = tf.layers.conv2d(
                    out, int(c), kernel,
                    strides=stride,
                    activation=None,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'sR_res{i}',
                    reuse=reuse)
                conv1 =  tf.layers.conv2d(
                    out, int(c)//2, 3,
                    strides=1,
                    activation=tf.nn.relu,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'sR_conv1{i}',
                    reuse=reuse)
                conv2 =  tf.layers.conv2d(
                    conv1, int(c), kernel,
                    strides=stride,
                    activation=tf.nn.relu,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'sR_conv2{i}',
                    reuse=reuse)
                out = residual + conv2
            elif layer_type == 'sR2':
                kernel, stride, c = _parse_layer_params(layer_desc, 3)
                residual = tf.layers.conv2d(
                    out, int(c), 1,
                    strides=stride,
                    activation=None,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'sR_res{i}',
                    reuse=reuse)
                skip = tf.layers.conv2d(
                    out, int(c), kernel,
                    strides=stride,
                    activation=tf.nn.relu,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                    name=f'sR_conv1{i}',
                    reuse=reuse)
                out = residual + skip

            elif layer_type == 'R':
                block_type, strides, n_blocks, filters = _parse_layer_params(layer_desc, 4)

                if block_type == 'building_block':
                    block_fn = resnets.building_block
                else:
                    block_fn = resnets.bottleneck_block
                name = f'res_block{i}'
                out = resnets.block_layer(inputs=out,
                                          filters=filters,
                                          block_fn=block_fn,
                                          blocks=n_blocks,
                                          strides=strides,
                                          is_training=is_training,
                                          name=name,
                                          data_format='channels_last')
            else:
                raise ValueError(f"Unknown layer type '{layer_type}' with params {layer_desc[1:]}")
            if name is not None:
                layer_map[f'{scope}/{name}'] = out

    if output_dim is not None:
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.layers.dense(out, output_dim,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              reuse=reuse,
                              name=f'last_dense')

    return out, layer_map


def spatial_softmax(inputs, data_format='channels_last'):
    """
    Spatial softmax layer
    :param inputs: input tensor
    :param data_format: 'channels_last' (default) or 'channels_first'
    :return: output after spatial softmax
    """
    # TODO: add test
    if data_format == 'channels_last':
        n, h, w, c = inputs.get_shape()
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
    elif data_format == 'channels_first':
        n, h, w, c = inputs.get_shape()
    else:
        raise ValueError(f"Unrecognized data_format '{data_format}'")

    out = tf.reshape(inputs, [-1, h.value * w.value])
    out = tf.nn.softmax(out)
    out = tf.reshape(out, [-1, c.value, h.value, w.value])
    out = tf.transpose(out, [0, 2, 3, 1])
    out = tf.expand_dims(out, -1)

    inds = np.expand_dims(np.arange(h.value), -1).repeat(h.value, axis=1)
    image_coords_np = np.concatenate([np.expand_dims(inds, -1), np.expand_dims(inds.T, -1)],
                                     axis=-1)
    image_coords = tf.constant(image_coords_np.astype(np.float32))
    image_coords = tf.expand_dims(image_coords, 2)

    out = tf.reduce_sum(out * image_coords, reduction_indices=[1, 2])

    return out


def reverse_gradient(inputs, weight=1.0):
    inputs_backward_mode = -weight * inputs
    return inputs_backward_mode + tf.stop_gradient(inputs - inputs_backward_mode)


def stop_gradient_getter(getter, name, *args, **kwargs):
    return tf.stop_gradient(getter(name, *args, **kwargs))
