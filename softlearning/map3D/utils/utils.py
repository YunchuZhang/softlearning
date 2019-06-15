import os
import copy
import psutil
import numpy as np
import math
import tensorflow as tf
def nyi():
    raise Exception('not yet implemented')
def binarize(input, threshold):
    return tf.where(input < threshold, tf.zeros_like(input), tf.ones_like(input))

def get_3dbbox(voxel):
    z_dim_sum = tf.reduce_sum(voxel, [2, 3])
    y_dim_sum = tf.reduce_sum(voxel, [1, 3])
    x_dim_sum = tf.reduce_sum(voxel, [1, 2])

    bs = z_dim_sum.get_shape()[0]
    pad = tf.zeros((bs, 1))
    z_dim_bin = tf.where(z_dim_sum > 0, tf.ones_like(z_dim_sum), tf.zeros_like(z_dim_sum))
    z_value = tf.concat([pad, z_dim_bin[:, :-1]], 1) - z_dim_bin

    z_bbox_min = tf.argmin(z_value, 1)
    z_len = z_value.get_shape()[-1].value
    z_bbox_max = z_len -1 - tf.argmax(tf.reverse(z_value, axis=[1]), 1)

    y_dim_bin = tf.where(y_dim_sum > 0, tf.ones_like(y_dim_sum), tf.zeros_like(y_dim_sum))
    y_value = tf.concat([pad, y_dim_bin[:, :-1]], 1) - y_dim_bin
    y_bbox_min = tf.argmin(y_value, 1)
    y_len = y_value.get_shape()[-1].value
    y_bbox_max = y_len - 1 - tf.argmax(tf.reverse(y_value, axis=[1]), 1)

    x_dim_bin = tf.where(x_dim_sum > 0, tf.ones_like(x_dim_sum), tf.zeros_like(x_dim_sum))
    x_value = tf.concat([pad, x_dim_bin[:, :-1]], 1) - x_dim_bin
    x_bbox_min = tf.argmin(x_value, 1)
    x_len = x_value.get_shape()[-1].value
    x_bbox_max = x_len - 1 - tf.argmax(tf.reverse(x_value, axis=[1]), 1)

    return tf.stack([z_bbox_min, z_bbox_max, y_bbox_min, y_bbox_max, x_bbox_min, x_bbox_max], 1)
def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def exchange_scope(name, scope, oldscope):
    head, tail = name.split(scope)
    assert head == ''
    return oldscope + tail


def onmatrix():
    return 'Linux compute' in os.popen('uname -a').read()


def iscontainer(obj):
    return isinstance(obj, list) or isinstance(obj, dict) or isinstance(obj, tuple)

#an alias
iscollection = iscontainer

def strip_container(container, fn=lambda x: None):
    assert iscontainer(container), 'not a container'

    if isinstance(container, list) or isinstance(container, tuple):
        return [(strip_container(obj, fn) if iscontainer(obj) else fn(obj))
                for obj in container]
    else:
        return {k: (strip_container(v, fn) if iscontainer(v) else fn(v))
                for (k, v) in list(container.items())}


def memory_consumption():
    #print map(lambda x: x/1000000000.0, list(psutil.Process(os.getpid()).memory_info()))
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    #return psutil.virtual_memory().used / 1000000000.0 #oof


 

def check_numerics(stuff):
    if isinstance(stuff, dict):
        for k in stuff:
            if not check_numerics(stuff[k]):
                raise Exception('not finite %s').with_traceback(k)
        return True
    elif isinstance(stuff, list) or isinstance(stuff, tuple):
        for x in stuff:
            check_numerics(x)
    else:
        return np.isfinite(stuff).all()

def apply_recursively(collection, f):

    def dict_valmap(g, dct):
        return {k:g(v) for (k,v) in dct.items()}

    def f_prime(x):
        if iscollection(x):
            return apply_recursively(x, f)
        else:
            return f(x)
    
    if not iscollection(collection):
        return f(collection)
    else:
        map_fn = dict_valmap if isinstance(collection, dict) else map
        return type(collection)(map_fn(f_prime, collection))

def map_if_list(x, fn):
    if isinstance(x, list):
        return list(map(fn, x))
    return fn(x)
    

def degrees(x):
    return x * 180/np.pi

def radians(x):
    return x * np.pi/180
