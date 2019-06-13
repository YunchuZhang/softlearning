import tensorflow as tf
# from numpy import *
from math import sqrt
from .utils_basic import print_shape
import constants as const
import numpy as np
from . import voxel

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def split_rt(rt):
    r = rt[:, :3, :3]
    t = rt[:, :3, 3]
    t = tf.reshape(t, [-1, 3])
    return r, t

def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3]
    t = tf.reshape(t, [3])
    return r, t

def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3

    if r is None and t is None:
        assert(False) # you have to provide either r or t
        
    if r is None:
        shape = t.get_shape()
        B = int(shape[0])
        r = tf.eye(3, batch_shape=[B])
    elif t is None:
        shape = r.get_shape()
        B = int(shape[0])
        
        t = tf.zeros([B, 3])
    else:
        shape = r.get_shape()
        B = int(shape[0])
        
    bottom_row = tf.tile(tf.reshape(tf.stack([0.,0.,0.,1.]),[1,1,4]),
                         [B,1,1])
    rt = tf.concat(axis=2,values=[r,tf.expand_dims(t,2)])
    rt = tf.concat(axis=1,values=[rt,bottom_row])
    return rt

def rigid_transform_3D(A, B, mask):
    A = A.numpy()
    B = B.numpy()
    mask = mask.numpy()
    assert len(A) == len(B)

    N = A.shape[0] # total points


    print('mask stats: %.2f, %.2f, %.2f' % (
        np.min(mask), np.mean(mask), np.max(mask)))
    print('N = %d' % N)
    if N > 3:
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # center the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))

        # dot is matrix multiplication for array
        H = np.dot(AA.T, BB)

        U, S, Vt = np.linalg.svd(H)

        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[2,:] *= -1
           R = np.dot(Vt.T, U.T)

        t = np.dot(-R, centroid_A.T) + centroid_B.T
        t = np.reshape(t, [3])
    else:
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)

    # print('R, t:')
    # print(R)
    # print(R.shape)
    # print(t)
    # print(t.shape)

    return R, t

def get_rigid_transform_from_flow_and_mask(flow, mask, rois):
    # flow is B x N x H x W x D x 3
    # mask is B x N x H x W x D x 1

    B, N, H, W, D, C = flow.get_shape().as_list()
    assert(C==3)

    """
    print('REPLACING MASK')
    flow_mag = tf.reduce_sum(tf.square(flow), axis=5, keepdims=True)
    mask = tf.cast(tf.greater(flow_mag, 0.01), tf.float32)
    """

    flow_ = tf.reshape(flow, [B*N, H, W, D, 3])
    mask_ = tf.reshape(mask, [B*N, H, W, D, 1])
    # rois_ = tf.reshape(rois, [B*N, H, W, D, 1])

    f = lambda x: get_rigid_transform_from_flow_and_mask_single(*x)
    rt_ = tf.map_fn(f, [flow_, mask_, rois], dtype=tf.float32)

    r_, t_ = split_rt(rt_)
    
    # rt = tf.reshape(rt_, [B, N, 4, 4])
    # flow_ = tf.reshape(flow, [B*N, H, W, D, 3])
    # mask_ = tf.reshape(mask, [B*N, H, W, D, 1])
    
    r = tf.reshape(r_, [B, N, 3, 3])
    t = tf.reshape(t_, [B, N, 3])

    return r, t
    
def get_rigid_transform_from_flow_and_mask_single(flow, mask, roi, crop_size=16):
    # flow is H x W x D x 3
    # mask is H x W x D x 1
    print(':'*10)

    Z, Y, X, C = flow.get_shape().as_list()
    assert(C==3)
    # rt = tf.eye(4, batch_shape=[])

    def meshgrid(depth, height, width):
        with tf.variable_scope("bbox_mesh_grid"):
            x_t = tf.reshape(
                tf.tile(
                    tf.range(width), [height * depth]),
                    [depth, height, width])
            y_t = tf.reshape(
                tf.tile(
                    tf.range(height), [width * depth]),
                    [depth, width, height])
            y_t = tf.transpose(y_t, [0, 2, 1])
            z_t = tf.reshape(
                tf.tile(
                    tf.range(depth), [width * height]),
                    [height, width, depth])
            z_t = tf.transpose(z_t, [2, 0, 1])
            grid = tf.cast(tf.stack([z_t, y_t, x_t], axis=3), dtype=tf.float32)
        return grid

    # # z, y, x = meshgrid(Z, Y, X)
    # pos0 = meshgrid(Z, Y, X)


    def meshgrid_origin_centered_(depth, height, width):
        with tf.variable_scope('_meshgrid'):
            x_t = tf.reshape(
                tf.tile(tf.linspace(-1.0, 1.0, width), [height * depth]),
                [depth, height, width])
            x_t = x_t * ((width - 1) /width)
            y_t = tf.reshape(
                tf.tile(tf.linspace(-1.0, 1.0, height), [width * depth]),
                [depth, width, height])
            y_t = tf.transpose(y_t, [0, 2, 1])
            y_t = y_t * ((height - 1) /height)
            sample_grid = tf.tile(
                tf.linspace(-1.0, 1.0, depth), [width * height])
            z_t = tf.reshape(sample_grid, [height, width, depth])
            z_t = tf.transpose(z_t, [2, 0, 1])
            z_t = z_t * ((depth - 1) /depth)

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            d_t_flat = tf.reshape(z_t, (1, -1))

            #ones = tf.ones_like(x_t_flat)
            grid = tf.concat([d_t_flat, y_t_flat, x_t_flat], 0)
            return grid
    
    pos0 = meshgrid_origin_centered_(Z, Y, X) * const.boundary_to_center
    # pos0 = tf.expand_dims(pos0, axis=0)
    # roi = tf.expand_dims(roi, axis=0)
    # pos0 = voxel.crop_and_resize_3d_box2_pad(pos0, roi, crop_size)
    # pos0 = tf.squeeze(pos0, axis=0)
    
    # pos0 = meshgrid(Z, Y, X)
    pos0 = tf.reshape(pos0, [Z, Y, X, 3])
    pos1 = pos0 + flow
    # mask = tf.cast(tf.greater(mask, 0.5), tf.float32)
    mask = tf.cast(tf.greater(mask, 0.001), tf.float32)

    pos0 = tf.reshape(pos0, [-1, 3])
    pos1 = tf.reshape(pos1, [-1, 3])
    mask = tf.reshape(mask, [-1])
    
    inds = tf.where(mask)
    pos0 = tf.gather(pos0, inds)
    pos1 = tf.gather(pos1, inds)
    
    pos0 = tf.reshape(pos0, [-1, 3])
    pos1 = tf.reshape(pos1, [-1, 3])

    # rt = rigid_transform_3D(A, B)
    r, t = tf.py_function(rigid_transform_3D, [pos0, pos1, mask], [tf.float32, tf.float32])
    r = tf.reshape(r, [1, 3, 3])
    t = tf.reshape(t, [1, 3])

    # r = tf.expand_dims(r, axis=0)
    # t = tf.expand_dims(t, axis=0)
    rt = merge_rt(r,t)
    # rt = tf.squeeze(rt, axis=0)

    rt = tf.reshape(rt, [4 ,4])
    
    return rt
    

    # return 0, 0
