import tensorflow as tf
import tensorflow.contrib.slim as slim
import constants as const
from . import tfpy
from . import voxel
from munch import Munch
from . import convlstm
from . import tfutil
from . import utils
from tensorflow import summary as summ
import ipdb
st = ipdb.set_trace
slim2 = Munch()

def summary_wrap(func):
    def func_(*args, **kwargs):
        rval = func(*args, **kwargs)
        print('histogram for', rval.name)
        summ.histogram(rval.name, rval)
        return rval
    return func_

if const.inject_summaries:
    for funcname in ['conv2d', 'conv2d_transpose', 'conv3d', 'conv3d_transpose']:
        func = getattr(slim, funcname)
        setattr(slim2, funcname, summary_wrap(func))
else:
    slim2 = slim    

def batch_norm(input, is_train, name=None):

    #return tf.keras.layers.BatchNormalization(input, is_training=is_train, decay=0.9,
    #           scope=name + "/batch_normalization_tf")
    return tf.contrib.layers.batch_norm(input, is_training=is_train, decay=const.BN_DECAY,
               scope=name + "/batch_normalization_tf")
    #return tf.keras.layers.BatchNormalization()(input)

    # return slim2.batch_norm(input, decay=0.9, is_training=is_train)
def unproject(inputs, cam_int=None, resize = False):
    """
    avoid using this function, please use unproject2
    """
    assert(1==2, "should try to use unproject2")

    if resize:
        inputs = tf.image.resize_images(inputs, (const.S, const.S))
    size = int(inputs.shape[1])
    bs = int(inputs.shape[0])

    #now unproject, to get our starting point
    # batch_size x h x w x c -> batch_size x d x h x w x c
    inputs = voxel.unproject_image(inputs, cam_int=cam_int, debug=True)

    #in addition, add on a z-map, and a local bias
    #copied from components.py
    meshgridz = tf.range(size, dtype = tf.float32)
    meshgridz = tf.reshape(meshgridz, (1, size, 1, 1))
    meshgridz = tf.tile(meshgridz, (bs, 1, size, size))
    meshgridz = tf.expand_dims(meshgridz, axis = 4)
    meshgridz = (meshgridz + 0.5) / (size/2) - 1.0 #now (-1,1)
    meshgridz = meshgridz # * const.boundary_to_center + const.radius
    #get the rough outline
    unprojected_mask = utils.binarize(tf.expand_dims(inputs[:,:,:,:,0], 4), 1)
    depth_tmp = tf.expand_dims(inputs[:,:,:,:,1], 4)
    unprojected_depth = (tf.expand_dims(inputs[:,:,:,:,1], 4) - const.radius) * (1/const.boundary_to_center)
    if const.USE_OUTLINE:
        inputs = inputs[:,:,:,:,2:]
    if const.H > 32:
        outline_thickness = 0.1
    else:
        outline_thickness = 0.2
    #outline_thickness =0.2 #0.2

    outline = tf.cast(tf.logical_and(
        unprojected_depth <= meshgridz,
        unprojected_depth + outline_thickness > meshgridz
    ), tf.float32)
    #outline = tf.cast(unprojected_depth <= meshgridz, tf.float32)
    outline_tmp = outline
    """
    from utils.vis_np import save_voxel
    for batch_id in range(2):
        save_voxel(outline[batch_id, :, :, :, 0], f"dump/unproject_b{batch_id}.binvox")
    """
    outline *= unprojected_mask
    if const.DEBUG_UNPROJECT:
        #return tf.expand_dims(inputs[:,:,:,:,0], 4) #this is the unprojected mask
        #assert(1==2)
        # unprojected_mask
        # depth_tmp: (32, 64, 64, 64, 1)
        # outline_tmp: (32, 64, 64, 64, 1)
        # outline: (32, 64, 64, 64, 1)
        return  tf.concat([meshgridz, unprojected_depth, outline, unprojected_mask, inputs[:,:,:,:,0:1]], 4)
    # (32, 64, 64, 64, 32)
    inputs_ = [inputs]
    if const.USE_MESHGRID:
        inputs_.append(meshgridz)
    if const.USE_OUTLINE:
        #inputs_.append(outline)
        inputs_ = [outline * inp for inp in inputs_]
    inputs = tf.concat(inputs_, axis = 4)
    return inputs



def unproject2(inputs, use_outline=False, use_meshgrid=False, debug_unproject = False,
               cam_int=None, resize = False):
    # st()
    if resize:
        inputs = tf.image.resize_images(inputs, (const.S, const.S))
    size = int(inputs.shape[1])
    bs = int(inputs.shape[0])

    #now unproject, to get our starting point
    # batch_size x h x w x c -> batch_size x d x h x w x c
    inputs = voxel.unproject_image(inputs, cam_int=cam_int, debug=True)

    #in addition, add on a z-map, and a local bias
    #copied from components.py
    if use_outline or use_meshgrid:
        meshgridz = tf.range(size, dtype = tf.float32)
        meshgridz = tf.reshape(meshgridz, (1, size, 1, 1))
        meshgridz = tf.tile(meshgridz, (bs, 1, size, size))
        meshgridz = tf.expand_dims(meshgridz, axis = 4)
        meshgridz = (meshgridz + 0.5) / (size/2) - 1.0 #now (-1,1)
        meshgridz = meshgridz # * const.boundary_to_center + const.radius
        #get the rough outline
    if use_outline:
        unprojected_mask = utils.binarize(tf.expand_dims(inputs[:,:,:,:,0], 4), 1)
        depth_tmp = tf.expand_dims(inputs[:,:,:,:,1], 4)
        unprojected_depth = (tf.expand_dims(inputs[:,:,:,:,1], 4) - const.radius) * (1/const.boundary_to_center)
        inputs = inputs[:,:,:,:,2:]
        """
        if const.H > 32:
            outline_thickness = 0.1
        else:
            outline_thickness = 0.2
        """
        outline_thickness =0.2 #0.2
        outline = tf.cast(tf.logical_and(
            unprojected_depth <= meshgridz,
            unprojected_depth + outline_thickness > meshgridz
        ), tf.float32)
        #outline = tf.cast(unprojected_depth <= meshgridz, tf.float32)
        outline *= unprojected_mask

    if debug_unproject:
        #return tf.expand_dims(inputs[:,:,:,:,0], 4) #this is the unprojected mask
        #assert(1==2)
        # unprojected_mask
        # depth_tmp: (32, 64, 64, 64, 1)
        # outline_tmp: (32, 64, 64, 64, 1)
        # outline: (32, 64, 64, 64, 1)
        return  tf.concat([meshgridz, unprojected_depth, outline, unprojected_mask, inputs[:,:,:,:,2:3]], 4)
    # (32, 64, 64, 64, 32)
    inputs_ = [inputs]
    if use_meshgrid:
        inputs_.append(meshgridz)
    if use_outline:
        #inputs_.append(outline)
        inputs_ = [outline * inp for inp in inputs_]
    inputs = tf.concat(inputs_, axis = 4)
    return inputs

def voxel_net_3d(inputs, aux = None, bn = True, outsize = 128, d0 = 16):
    #aux is used for the category input
    bn_trainmode = ((const.mode != 'test') and (not const.rpvx_unsup))
    if const.force_batchnorm_trainmode:
        bn_trainmode = True
    if const.force_batchnorm_testmode:
        bn_trainmode = False

    normalizer_params={'is_training': bn_trainmode, 
                       'decay': 0.99,
                       'epsilon': 1e-5,
                       'scale': True,
                       'updates_collections': None}

    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm if bn else None,
                        normalizer_params=normalizer_params
                        ):
        
        dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
        ksizes = [4, 4, 4, 4, 8]
        strides = [2, 2, 2, 2, 1]
        paddings = ['SAME'] * 4 + ['VALID']

        if const.S == 64:
            ksizes[-1] = 4
        elif const.S == 32:
            ksizes[-1] = 2

        net = inputs

        skipcons = [net]
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = slim2.conv3d(net, dim, ksize, stride=stride, padding=padding)

            skipcons.append(net)
        
        #BS x 4 x 4 x 4 x 256

        if aux is not None:
            aux = tf.reshape(aux, (const.BS, 1, 1, 1, -1))
            net = tf.concat([aux, net], axis = 4)

        #fix from here..
        chans = [8*d0, 4*d0, 2*d0, d0, 1]
        strides = [1, 2, 2, 2, 2]
        ksizes = [8, 4, 4, 4, 4]
        paddings = ['VALID'] + ['SAME'] * 4
        activation_fns = [tf.nn.relu] * 4 + [None] #important to have the last be none

        if const.S == 64:
            ksizes[0] = 4
        elif const.S == 32:
            ksizes[0] = 2
        
        decoder_trainable = not const.rpvx_unsup #don't ruin the decoder by FTing

        skipcons.pop() #we don't want the innermost layer as skipcon

        foo = lambda: None
        #we'll use this object to smuggle additional information out
        
        for i, (chan, stride, ksize, padding, activation_fn) \
            in enumerate(zip(chans, strides, ksizes, paddings, activation_fns)):

            # if i == len(chans)-1:
            #     norm_fn = None
            # else:
            norm_fn = slim.batch_norm

            net = slim2.conv3d_transpose(
                net, chan, ksize, stride=stride, padding=padding, activation_fn = activation_fn,
                normalizer_fn = norm_fn, trainable = decoder_trainable
            )

            #now concatenate on the skip-connection
            net = tf.concat([net, skipcons.pop()], axis = 4)

        #one last 1x1 conv to get the right number of output channels
        net = slim2.conv3d(
            net, 1, 1, 1, padding='SAME', activation_fn = None,
            normalizer_fn = None, trainable = decoder_trainable
        )

    logit = net
    net = tf.nn.sigmoid(net)

    return Munch(pred = net, features = foo, logit = logit)

#def convlstm(*args, scopename='convlstm', **kwargs):
#    with tf.variable_scope(scopename, reuse = False):
#        return convgru_(*args, **kwargs)

def convlstm_(inputs, kernel=[3, 3, 3], filters=32):
    bs, l, h, w, d, ch = inputs.get_shape().as_list()

    conv_gru = convlstm.ConvLSTMCell(
        shape=[h, w, d],
        initializer=slim.initializers.xavier_initializer(),
        kernel=kernel,
        filters=filters
    )

    seq_length = [l]*bs

    outputs, state = tf.nn.dynamic_rnn(
        conv_gru,
        inputs,
        parallel_iterations=1,
        sequence_length=seq_length,
        dtype=tf.float32,
        time_major=False,
        swap_memory = True,
    )

    return Munch(outputs = outputs, state = state)

def convgru(*args, scopename='convgru', **kwargs):
    with tf.variable_scope(scopename, reuse = False):
        return convgru_(*args, **kwargs)

def convgru_(inputs, kernel=[3, 3, 3], filters=32):
    bs, l, h, w, d, ch = inputs.get_shape().as_list()

    conv_gru = convlstm.ConvGRUCell(
        shape=[h, w, d],
        initializer=slim.initializers.xavier_initializer(),
        kernel=kernel,
        filters=filters
    )
    
    seq_length = [l]*bs
    
    outputs, state = tf.nn.dynamic_rnn(
        conv_gru,
        inputs,
        parallel_iterations=1,
        sequence_length=seq_length,
        dtype=tf.float32,
        time_major=False,
        swap_memory = True,
    )
    
    return Munch(outputs = outputs, state = state)


class convgru_cell():
    def __init__(self, shape, kernel=[5,5,5], filters=32):
        self.conv_gru = convlstm.ConvGRUCell(
            shape=shape,
            initializer=slim.initializers.xavier_initializer(),
            kernel=kernel,
            filters=filters
        )
    def __call__(self,x, h, scope=None, reuse=False):
        out = self.conv_gru(x, h, scope=scope, reuse=reuse)
        return out[0]


def gru_aggregator(inputs, name="", is_multi_t_output=False):
    inputs = tf.stack(inputs, axis = 1)
    filter_size = inputs.get_shape()[-1]
    layers = []
    bs, t, w, h, d, c = inputs.get_shape()
    kernel = [5, 5, 5]
    if d.value == 1:
        kernel = [5, 5, 1]
    for i in range(2):
        layers.append(convgru(inputs, kernel = kernel, filters = filter_size, scopename=f'convgru_{name}_{i}'))
        inputs = layers[-1].outputs
    if not is_multi_t_output:
        return sum((l.state for l in layers))
    else:
        seq_len = layers[0]["outputs"].get_shape()[1]
        output = []
        for t in range(seq_len):
            # using features on different gru layers
            out_t = sum((l.outputs[:, :t+1, :, :, :, :] for l in layers))
            output.append(tf.reduce_mean(out_t, 1))
        return tf.concat(output, 0)
    #return tf.concat([l.state for l in layers], axis = -1)

def encoder3D(inputs, aux = None, d0 = 16):
    raise Exception('you should probably not be using this')
    
    net = inputs    
    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm):
        dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
        ksizes = [4, 4, 4, 4, 8]
        strides = [2, 2, 2, 2, 1] 
        paddings = ['SAME'] * 4 + ['VALID']

        skipcons = [net]
        
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = slim2.conv3d(net, dim, ksize, stride=stride, padding=padding)
            skipcons.append(net)
        
        if aux is not None:
            aux = tf.reshape(aux, (const.BS, 1, 1, 1, -1))
            net = tf.concat([aux, net], axis = 4)

        chans = [8*d0, 4*d0, 2*d0, d0, 1]
        strides = [1, 2, 2, 2, 2]
        ksizes = [8, 4, 4, 4, 4]
        paddings = ['VALID'] + ['SAME'] * 4
        activation_fns = [tf.nn.relu] * 5

        skipcons.pop() #we don't want the innermost layer as skipcon
        skipcons.pop(0)

        #skipcons contains feature tensors of sizes: 64, 32, 16, 8
        return skipcons


def depth_channel_net_v2(feature):
    # pool by 8 on the depth axis
    # then flatten last two axes and do a 1x1 conv

    if False:
        return DRC(feature)
    
    if const.W == const.H == 128:
        K = 8
    elif const.W == const.H == 64:
        K = 4
        
    feature = tf.nn.pool(feature, [1,1,K], 'MAX', 'SAME', strides = [1,1,K])
    
    bs, h, w, d_, c = map(int, feature.shape)
    feature = tf.reshape(feature, (bs, h, w, d_*c))

    S = voxel.getsize(feature)

    if const.W == const.H == 128:
        dim = 4096//S
    elif const.W == const.H == 64:
        dim = 2048//S

    with tf.variable_scope("depthconv_%d" % S):
        feature = slim2.conv2d(feature, dim, 1, stride=1, padding='SAME')

    return feature

foo = None
def DRC(feature):
    global foo
    #take 1x1 conv to get occupancy
    occ = slim2.conv3d(feature, 1, 1, activation_fn = None)
    occ = tf.nn.sigmoid(occ - 0.5) #start with a low bias!!
    not_occ = 1.0-occ
    not_occ_cum = tf.cumprod(not_occ, axis = 3, exclusive = True)
    hit_here = not_occ_cum * occ

    if occ.shape.as_list()[1] == 32:
        print('asdf')
        foo = occ
    return tf.reduce_sum(hit_here * feature, axis = 3)
    
def decoder2D(features, is_training, apply_3to2 = True, dim_base=32, out_dim=3, last_stride=1, is_not_bn=False):
    features = features[::-1]
    if apply_3to2:
        features = [depth_channel_net_v2(feature) for feature in features]
    dims = [dim_base*8, dim_base*4, dim_base*2, dim_base]
        
    ksizes = [4]*4
    strides = [2]*4
    paddings = ['SAME']*4

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        activation_fn=tf.nn.relu):

        net = 0
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            if features:
                net += features.pop()
            #net = slim2.conv2d_transpose(net, dim, ksize, stride = stride, padding = padding)
            #net = slim2.conv2d(net, dim, ksize, stride = 1, padding = 'SAME')
            net = slim2.conv2d(net, dim, ksize, stride = 1, padding = padding)
            if not is_not_bn:
                net = batch_norm(net, is_train=is_training, name=f"layer{i}")
            net = tf.image.resize_nearest_neighbor(net, [net.shape.as_list()[1]*2]*2)

    net = slim2.conv2d(net, out_dim, 3, stride = last_stride, padding = 'SAME', activation_fn = None)
    net = tfutil.tanh01(net)
    return net

def slim2_conv2d(input, dim, kernel, stride=1, padding="SAME", activation_fn = None, is_normalize=False):

    normalizer_fn = None
    if is_normalize:
        normalizer_fn = slim.layer_norm
    return slim2.conv2d(input, dim, kernel, stride=stride, padding=padding,\
        activation_fn = activation_fn, normalizer_fn = normalizer_fn)


def encoder2D(net, is_training, nlayer=4, is_not_bn=False):
    #return in 64, 32, 16, and 8

    outputs = []
    dim = const.fs_2D
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        padding = 'SAME'):

        dims = [dim, dim*2, dim*4, dim*8]
        ksizes = [3, 3, 3, 3]

        for i, (dim, ksize) in enumerate(zip(dims, ksizes)):
            if i == nlayer:
                break
            net = slim2.conv2d(net, dim, ksize, stride=2)
            if not is_not_bn:
                net = batch_norm(net, is_train=is_training, name=f"layer{i}")

            net = slim2.conv2d(net, dim, ksize, stride=1)
            if not is_not_bn:
                net = batch_norm(net, is_train=is_training, name=f"layer{i}_2")

            outputs.append(net)
    return outputs


def encoder_decoder2D(inputs, nlayer=4, is_not_bn=False):

    #return in 64, 32, 16, and 8
    net = inputs.pop()
    features = []
    d0 = const.fs_2D * 2
    dim = const.fs_2D
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm):

        dims = [dim, dim*2, dim*4, dim*8]
        ksizes = [4, 4, 4, 4]
        strides = [2, 2, 2, 4]
        paddings = ["SAME"] * 3 + ['VALID']

        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            if i == nlayer:
                break
            net = slim2.conv2d(net, dim, ksize, stride=stride)
            #net = slim2.conv2d(net, dim, ksize, stride=1)
            features.append(net)

    dims = dims[:-1][::-1] + [d0//2]
    ksizes = ksizes[::-1]
    strides = strides[::-1]
    paddings = paddings[::-1]


    features.pop()
    outputs = []
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm):

        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = slim2.conv2d_transpose(net, dim, ksize, stride=stride, padding=padding)
            if features:
                net += features.pop()
            #net = slim2.conv2d_transpose(net, dim, ksize, stride = stride, padding = padding)
            #net = slim2.conv2d(net, dim, ksize, stride = 1, padding = 'SAME')
            outputs.append(net)

    return outputs

def slim2_conv3d(input, dim, kernel, stride=1, padding="SAME", scope=None):
    return slim2.conv3d(input, dim, kernel, stride=stride, padding=padding, activation_fn = None, scope=scope)



def slim2_fc_debug_bn(input, dim, is_training, activation_fn=tf.nn.sigmoid, is_normalize=False, reuse=True):
    #if is_normalize:
    #    normalizer_fn = slim.layer_norm

    #return slim.batch_norm(input, is_training=is_training, scope="bn_debug_slim/batch_normalization")
    #out = slim.fully_connected(input,dim, activation_fn=activation_fn, normalizer_fn=None, \
    #      scope="bn_debug")
    return batch_norm(input, is_train=is_training, name="bn_debug")


def slim2_fc(input, dim, activation_fn=tf.nn.sigmoid, is_normalize=False, name=""):
    normalizer_fn = None
    if is_normalize:
        normalizer_fn = slim.layer_norm

    return slim.fully_connected(input,dim, activation_fn=activation_fn,\
        normalizer_fn=normalizer_fn, scope=name)


def slim2_fc_bad(input, dim, is_training, activation_fn=tf.nn.sigmoid):
    normalizer_fn = slim.batch_norm

    return slim.fully_connected(input,dim, activation_fn=activation_fn, normalizer_fn=normalizer_fn)


def slim2_fc_ok(input, dim, is_training, activation_fn=tf.nn.sigmoid, name="", reuse=True):
    out = slim.fully_connected(input,dim, activation_fn=activation_fn)
    out = batch_norm(out, is_train=is_training, name= name, reuse=reuse)
    return out

def small_encoder_decoder3D(inputs, is_training, is_not_bn=False, nlayers=2, use_skipcons=True):
    inputs = inputs[::-1]
    net = inputs.pop()
    outputs = [net]
    bs, s, s, s, c = net.get_shape()
    skipcons = [net]
    d0 = c.value
    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None):
        dims = [2*i*d0 for i in range(1, nlayers + 1)]
        ksizes = [4 for i in range(nlayers)]
        strides = [2 for i in range(nlayers)]
        paddings = ['SAME' for i in range(nlayers)]
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = slim2.conv3d(net, dim, ksize, stride=stride, padding=padding)
            if not is_not_bn:
                net = batch_norm(net, is_train=is_training, name=f"encoder_{i}")
            if inputs:
                net += inputs.pop()
            outputs.append(net)
            skipcons.append(net)
        outputs.append(skipcons.pop())
        dims = dims[:-1][::-1] + [d0]
        ksizes = ksizes[::-1]
        strides = strides[::-1]
        paddings = paddings[::-1]
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = slim2.conv3d_transpose(net, dim, ksize, stride=stride, padding=padding)
            if not is_not_bn:
                net = batch_norm(net, is_train=is_training, name=f"decoder_{i}")
            if skipcons and use_skipcons:
                if len(skipcons) == 1 and skipcons[0].get_shape()[-1].value\
                    is not net.get_shape()[-1].value:
                    pass
                else:
                    net += skipcons.pop()
                #net = tf.concat([net, skipcons.pop()], axis = -1)
            outputs.append(net)
        return outputs

def encoder_decoder3D(inputs, is_training, aux = None, is_not_bn=False):
    inputs = inputs[::-1]
    d0 = const.fs_2D * 2
    outputs = []
    net = inputs.pop()
    skipcons = [net]
    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                        activation_fn=tf.nn.relu, normalizer_fn=None):
        if const.H == const.W == 128:
            #64 -> 32 -> 16 -> 8 -> 4 -> 1            
            dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
            ksizes = [4, 4, 4, 4, 4]
            strides = [2, 2, 2, 2, 4]
            paddings = ['SAME'] * 4 + ['VALID']
        elif const.H == const.W == 64:
            #32 -> 16 -> 8 -> 4 -> 1
            dims = [d0, 2*d0, 4*d0, 8*d0]
            ksizes = [4, 4, 4, 4]
            strides = [2, 2, 2, 4]
            paddings = ['SAME'] * 3 + ['VALID']

        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = slim2.conv3d(net, dim, ksize, stride=stride, padding=padding)
            if not is_not_bn:
                net = batch_norm(net, is_train=is_training, name=f"encoder_{i}")
            if inputs:
                #net = tf.concat([net, inputs.pop()], axis = -1)
                net += inputs.pop()
            skipcons.append(net)
        #return skipcons[:-2][::-1] #if we can also truncate the encoder here
        skipcons.pop() #we don't want the innermost layer as skipcon

        #1 -> 4 -> 8 -> 16 -> 32 -> 64
        dims = dims[:-1][::-1] + [d0//2]
        ksizes = ksizes[::-1]
        strides = strides[::-1]
        paddings = paddings[::-1]
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = slim2.conv3d_transpose(net, dim, ksize, stride=stride, padding=padding)
            if not is_not_bn:
                net = batch_norm(net, is_train=is_training, name=f"decoder_{i}")
            if skipcons:
                if len(skipcons) == 1 and skipcons[0].get_shape()[-1].value\
                    is not net.get_shape()[-1].value:
                    pass
                else:
                    net += skipcons.pop()
                #net = tf.concat([net, skipcons.pop()], axis = -1)
            outputs.append(net)

    if const.H == const.W == 128:
        outputs.pop(0) #don't want 4x4x4

    return outputs

def MnistAE(net):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        activation_fn = tf.nn.relu,
                        normalizer_fn = slim.batch_norm,
                        padding = 'SAME'):
        
        net = slim2.conv2d(net, 8, [3,3], stride = [2,2])
        net = slim2.conv2d(net, 16, [3,3], stride = [2,2])
        net = slim2.conv2d(net, 64, [7,7], stride = [1,1], padding = 'VALID')
        
        net = slim2.conv2d_transpose(net, 16, [7, 7], stride = [1, 1], padding= 'VALID')
        net = slim2.conv2d_transpose(net, 8, [3,3], stride = [2,2])
        net = slim2.conv2d_transpose(net, 1, [3,3], stride = [2,2],
                                     normalizer_fn = None, activation_fn = None)
        
    return tf.nn.sigmoid(net)
        
    
def MnistAEconvlstm(x):
    from .network_down import make_lstmConv

    net = x
    net = slim2.conv2d(net, 8, [3,3], stride = [2,2])
    net = slim2.conv2d(net, 16, [3,3], stride = [2,2])
    net = slim2.conv2d(net, 64, [7,7], stride = [1,1], padding = 'VALID')
    #B x 1 x 1 x 64

    out, extra = make_lstmConv(
        net,
        None, 
        x, 
        [['convLSTM', 64, 1, 8, 64]],
        stochastic = const.MNIST_CONVLSTM_STOCHASTIC,
        weight_decay = 0.0001,
        is_training = True,
        reuse = False,
        output_debug = False
    )
    out = tf.nn.sigmoid(out)

    out.loss = extra['kl_loss']
    return out
