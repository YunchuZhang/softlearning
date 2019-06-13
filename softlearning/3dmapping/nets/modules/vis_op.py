import tensorflow as tf
import utils.vis_tf as vis_tf
import numpy as np
def meshgrid(depth, height, width):
    with tf.variable_scope('meshgrid'):
        x_t = tf.reshape(
            tf.tile(tf.linspace(-1.0, 1.0, width), [height * depth]),
            [depth, height, width])
        y_t = tf.reshape(
            tf.tile(tf.linspace(-1.0, 1.0, height), [width * depth]),
            [depth, width, height])
        y_t = tf.transpose(y_t, [0, 2, 1])
        sample_grid = tf.tile(
            tf.linspace(-1.0, 1.0, depth), [width * height])
        z_t = tf.reshape(sample_grid, [height, width, depth])
        z_t = tf.transpose(z_t, [2, 0, 1])

    out = tf.stack([z_t, y_t, x_t], -1)
    return out


def make_dot_3D(points, out_size, sigma=0.1):
    """
    points: batch_size x 3
    """
    bs, _ = points.get_shape()
    grid = meshgrid(out_size, out_size, out_size)
    dist = tf.reduce_sum(tf.square(tf.tile(tf.expand_dims(grid, 0),\
                                           [bs, 1, 1, 1, 1])\
           - tf.reshape(points, [bs, 1, 1, 1, 3])), -1)
    gauss_map = (1.0/np.sqrt(2.0 * np.pi * sigma**2))* tf.exp((-0.5) *\
                dist * (1.0/(sigma**2)))
    return gauss_map
def make_dot_figure(point, image_template, nobjs, sigma = 0.05, boundary_to_center=1.0):
    """
    This is for dynamic learning, which create heatmap from top view and side view 
    point: batch_size x 3
    bounary_to_center: how large is your 3d tensor crop in the original coordinate system.

    """
    hand_center_xz = (-1) * tf.stack([point[:, 0] * (1/boundary_to_center),\
                                    point[:, 2] * (1/boundary_to_center)],
                                    1)  
    hand_center_xz = tf.reshape(tf.tile(tf.expand_dims(hand_center_xz, 1), [1, nobjs, 1]), [-1, 2])         
    robot_hand_image_xz = vis_tf.draw_dot(image_template, hand_center_xz, sigma=sigma)
    hand_center_xy = (-1) * tf.stack([point[:, 1] * (1/boundary_to_center),\
                                    point[:, 2] * (1/boundary_to_center)],
                                    1)  
    hand_center_xy = tf.reshape(tf.tile(tf.expand_dims(hand_center_xy, 1), [1, nobjs, 1]), [-1, 2])         
    robot_hand_image_xy = vis_tf.draw_dot(image_template, hand_center_xy, sigma=sigma)
    return robot_hand_image_xz, robot_hand_image_xy

def top_side_images_from_voxel(tensor_3d, out_c=3):
    object_in_3D_image_xz = tf.reverse(tf.reduce_mean(tensor_3d, 2), [1,2])
    object_in_3D_image_xz = tf.divide(object_in_3D_image_xz, \
                            tf.math.reduce_max(object_in_3D_image_xz, [1,2], keepdims=True) + 0.00000001)
    object_in_3D_image_xz = tf.tile(object_in_3D_image_xz, [1,1,1,out_c])

    object_in_3D_image_xy = tf.reverse(tf.reduce_mean(tensor_3d, 1), [1,2])
    object_in_3D_image_xy = tf.divide(object_in_3D_image_xy, \
                            tf.math.reduce_max(object_in_3D_image_xy, [1,2], keepdims=True) + 0.00000001)
    object_in_3D_image_xy = tf.tile(object_in_3D_image_xy, [1,1,1,out_c])
    return object_in_3D_image_xz, object_in_3D_image_xy


def image_sum_by_overlap(image, image_top, threshold=0.5):
    mask = tf.where( tf.greater(image_top, threshold), tf.ones_like(image_top), tf.zeros_like(image_top))
    final_image= image * (1-mask) + mask * image_top
    return final_image

def merge_image(list_of_images, bs, nobjs):
    nrows = len(list_of_images)
    concat_images = []
    for list_img in list_of_images:
        concat_images.append(tf.concat(list_img, 2)) 
    final_image = tf.concat(concat_images, 2)
    _, img_h, img_w, img_c = final_image.get_shape()
    final_image = tf.reshape(final_image, [bs, nobjs, img_h, img_w, img_c])
    return final_image

