import cv2
import tensorflow as tf
#import hyperparams as hyp
#from utils_basic import *
import os
# import utils_misc
#import utils_geom
#import utils_misc
#import voxelizer
# import utils_rpn_tf
import numpy as np
import matplotlib
import matplotlib.cm
from itertools import combinations


from .utils_basic import normalize
EPS = 1e-6


def summ_oned(name='oned', im=None, norm=True, is3D=False, max_outputs=1, collections=[]):
    if is3D:
        im = prep_birdview_vis(im)
    flow_img = tf.reverse(oned2inferno(im, norm=norm), axis=[1])
    
    tf.summary.image(name, flow_img, max_outputs=max_outputs, collections=collections)
    return flow_img

def oned2inferno(d, norm=True):
    if len(d.get_shape())==3:
        d = tf.expand_dims(d, axis=3)
    # convert a 1chan input to a 3chan image output
    if norm:
        d = normalize(d)
        rgb = colorize(d, cmap='inferno')
    else:
        rgb = colorize(d, vmin=0., vmax=1., cmap='inferno')
    rgb = tf.cast(255.0*rgb, tf.uint8)
    # rgb = tf.reshape(rgb, [-1, hyp.H, hyp.W, 3])
    # rgb = tf.expand_dims(rgb, axis=0)
    return rgb



def summ_3D_flow(flow, mod='', clip=1.0, collections=[]):
    tf.summary.histogram('flow_x%s' % mod, flow[:,:,:,:,0], collections=collections)
    tf.summary.histogram('flow_y%s' % mod, flow[:,:,:,:,1], collections=collections)
    tf.summary.histogram('flow_z%s' % mod, flow[:,:,:,:,2], collections=collections)
    # flow is B x H x W x D x 3; inside the 3 it's XYZ
    flow_xz = tf.stack([flow[:,:,:,:,0], flow[:,:,:,:,2]], axis=4) # grab x, z
    flow_xy = tf.stack([flow[:,:,:,:,0], flow[:,:,:,:,1]], axis=4) # grab x, y
    flow_yz = tf.stack([flow[:,:,:,:,1], flow[:,:,:,:,2]], axis=4) # grab y, z
    # these are B x H x W x D x 2
    flow_xz = tf.reduce_mean(flow_xz, axis=1) # reduce over H (y)
    flow_xy = tf.reduce_mean(flow_xy, axis=3) # reduce over D (z)
    flow_yz = tf.reduce_mean(flow_yz, axis=2) # reduce over W (x)
    summ_flow('flow_xz%s' % mod, flow_xz, clip=clip, is3D=True) # rot90 for interp
    summ_flow('flow_xy%s' % mod, flow_xy, clip=clip)
    summ_flow('flow_yz%s' % mod, flow_yz, clip=clip)
    flow_mag = tf.reduce_mean(tf.reduce_sum(tf.sqrt(EPS+tf.square(flow)), axis=4, keepdims=True), axis=1)
    return summ_oned('flow_mag%s' % mod, flow_mag, is3D=True, collections=collections)

def prep_birdview_vis(image):
    # # vox is B x Y x X x Z x C

    # # discard the vertical dim
    # image = tf.reduce_mean(vox, axis=1)

    # right now, X will be displayed vertically, which is confusing... 

    # make "forward" point up, and make "right" point right
    image = tf.map_fn(tf.image.rot90, image)

    return image



def summ_flow(name='flow', im=None, clip=0.0, is3D=False, collections=[]):
    if is3D:
        im = prep_birdview_vis(im)
    tf.summary.image(name, flow2color(im, clip=clip), max_outputs=1, collections=collections)

def flow2color(flow, clip=50.0):
    """
    :param flow: Optical flow tensor.
    :return: RGB image normalized between 0 and 1.
    """
    with tf.name_scope('flow_visualization'):
        # B, H, W, C dimensions.
        abs_image = tf.abs(flow)
        flow_mean, flow_var = tf.nn.moments(abs_image, axes=[1, 2, 3])
        flow_std = tf.sqrt(flow_var)

        if clip:
            mf = clip
            flow = tf.clip_by_value(flow, -mf, mf)/mf
        else:
            # Apply some kind of normalization. Divide by the perceived maximum (mean + std)
            flow = flow / tf.expand_dims(tf.expand_dims(
                tf.expand_dims(flow_mean + flow_std + 1e-10, axis=-1), axis=-1), axis=-1)

        radius = tf.sqrt(tf.reduce_sum(tf.square(flow), axis=-1))
        radius_clipped = tf.clip_by_value(radius, 0.0, 1.0)
        angle = tf.atan2(-flow[..., 1], -flow[..., 0]) / np.pi

        hue = tf.clip_by_value((angle + 1.0) / 2.0, 0.0, 1.0)
        saturation = tf.ones(shape=tf.shape(hue), dtype=tf.float32) * 0.75
        value = radius_clipped
        hsv = tf.stack([hue, saturation, value], axis=-1)
        flow = tf.image.hsv_to_rgb(hsv)
        flow = tf.cast(flow*255.0, tf.uint8)
        return flow


def colorize(value, normalize=True, vmin=None, vmax=None, cmap=None, vals=255):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.

    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.

    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
      - vals: the number of values in the cmap minus one

    Example usage:

    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """
    value = tf.squeeze(value, axis=3)

    if normalize:
        vmin = tf.reduce_min(value) if vmin is None else vmin
        vmax = tf.reduce_max(value) if vmax is None else vmax
        value = (value - vmin) / (vmax - vmin) # vmin..vmax

        # dma = tf.reduce_max(value)
        # dma = tf.Print(dma, [dma], 'dma', summarize=16)
        # tf.summary.histogram('dma', dma) # just so tf.Print works
        
        # quantize
        indices = tf.cast(tf.round(value * float(vals)), tf.int32)
    else:
        # quantize
        indices = tf.cast(value, tf.int32)

    # 00 Unknown 0 0 0
    # 01 Terrain 210 0 200
    # 02 Sky 90 200 255
    # 03 Tree 0 199 0
    # 04 Vegetation 90 240 0
    # 05 Building 140 140 140
    # 06 Road 100 60 100
    # 07 GuardRail 255 100 255
    # 08 TrafficSign 255 255 0
    # 09 TrafficLight 200 200 0
    # 10 Pole 255 130 0
    # 11 Misc 80 80 80
    # 12 Truck 160 60 60
    # 13 Car:0 200 200 200
  
    if cmap=='vkitti':
        colors = np.array([0, 0, 0,
                           210, 0, 200,
                           90, 200, 255,
                           0, 199, 0,
                           90, 240, 0,
                           140, 140, 140,
                           100, 60, 100,
                           255, 100, 255,
                           255, 255, 0,
                           200, 200, 0,
                           255, 130, 0,
                           80, 80, 80,
                           160, 60, 60,
                           200, 200, 200,
                           230, 208, 202]);
        colors = np.reshape(colors, [15, 3]).astype(np.float32)/255.0
        colors = tf.constant(colors)
    else:
        # gather
        cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
        if cmap=='RdBu' or cmap=='RdYlGn':
            colors = cm(np.arange(256))[:, :3]
        else:
            colors = cm.colors
        colors = np.array(colors).astype(np.float32)
        colors = np.reshape(colors, [-1, 3])
        colors = tf.constant(colors, dtype=tf.float32)
    
    value = tf.gather(colors, indices)
    # value is float32, in [0,1]
    return value
