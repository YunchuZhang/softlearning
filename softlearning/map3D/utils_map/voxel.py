from . import tfpy
import tensorflow as tf
from . import camera
import constants as const
import numpy as np
from . import tfutil

import tfquaternion as tfq

"""

* crop_mask_to_full_mask_precise:
    put mask back the the whole tensor given the bounding box location

* resize_voxel:
* crop_and_resize_2d_bbox2:
    crop region according to bounding boxes and resize.
    the boxes are in the format of batch_size x nrois x 3 x 2

* rotate_helper:
    this is for cross-correlation. If you always search for the sample set of
    angles, this helper helps you precomputed the sampling indices so you
    do not need to recompute them during run time.

* rotate_and_project_voxel

* rotate_voxel2:
    voxel rotation without using the "transform" function. The transform function
    has a dependency on focal length and radius from the camera.
    rotate_voxel2 eliminate such dependency.
    this one takes voxel and rotation matrix(3x3) as input

* translate_given_angle2:
    rotate using specify dtheta, dphi. This one calls the rotate_voxel2 function

* rotate_voxel:
    Deprecated. This one rotates using the transform function.
    please use rotate_voxel2.
    The input rotation matrix is 4x4 considering translation

* project_voxel:

* unproject_image:

* translate_given_angles:
   rotate using dtheta, phi1, phi2.
   This one is using the rotate_voxel function.

"""

def crop_mask_to_full_mask_precise_bs(rois, mask, output_size, class_labels=None):
    """
    rois: batch_size x 3 x 2
    mask: batch_size x 128 x 128 x 128 x c
    output_size: scalar
    """

    if class_labels is None:
        class_labels = tf.ones(tf.shape(rois)[0])
    voxel_c = mask.get_shape()[-1]
    f = lambda x: crop_mask_to_full_mask_precise(*x)
    pad_out, roi  = tf.map_fn(f,\
        [rois, class_labels, mask, output_size * tf.ones((tf.shape(rois)[0]), dtype=tf.int32)],\
        dtype = (tf.float32, tf.int32))
    pad_out = tf.reshape(pad_out, [tf.shape(rois)[0], output_size, output_size, output_size, voxel_c])
    return pad_out

def crop_mask_to_full_mask_precise_greedy(roi_indices, class_labels, mask, output_size):
    return crop_mask_to_full_mask_precise(roi_indices, class_labels, mask, output_size, greedy=True)


def crop_mask_to_full_mask_precise(roi_indices, class_labels, mask, output_size, greedy=False):
    """
    roi_indices: 3x2
    class_labels:()
    mask:16x16x16x1
    output_size:()
    """

    if greedy:
        roi_indices_d = tf.stack([tf.math.floor(roi_indices[:,0] * tf.cast(output_size, dtype=tf.float32)),
            tf.math.ceil(roi_indices[:,1] * tf.cast(output_size, dtype=tf.float32))], 1)

    roi_indices_d = tf.stack([tf.round(roi_indices[:,0] * tf.cast(output_size, dtype=tf.float32)),
        tf.round(roi_indices[:,1] * tf.cast(output_size, dtype=tf.float32))], 1)
    roi_indices_d = tf.clip_by_value(tf.cast(roi_indices_d, dtype=tf.int32), 0, output_size)
    Sd, Sh, Sw, c = mask.get_shape()
    # make sure rois are valid, 3x2
    roi_indices_d = tf.stack([roi_indices_d[:,0],
        tf.maximum(roi_indices_d[:,1], roi_indices_d[:,0])], 1)

    dhw_d = roi_indices_d[:,1] - roi_indices_d[:,0]
    valid = tf.cast(class_labels, dtype=tf.float32) *\
            tf.cast(tf.greater(dhw_d[0], 0), dtype=tf.float32) *\
            tf.cast(tf.greater(dhw_d[1], 0), dtype=tf.float32) *\
            tf.cast(tf.greater(dhw_d[2], 0), dtype=tf.float32)

    # calculate distortion
    roi_center = tf.reduce_mean(roi_indices, 1)
    len_to_center = tf.stack([roi_center - roi_indices[:,0], roi_indices[:, 1] - roi_center], 1)
    roi_indices_d_norm = tf.cast(roi_indices_d, tf.float32)/tf.cast(output_size, dtype=tf.float32)
    len_to_center_d = tf.stack([roi_center - roi_indices_d_norm[:,0], roi_indices_d_norm[:, 1] - roi_center], 1)
    bbox_scale = tf.divide(len_to_center_d, len_to_center)
    source_bbox = tf.stack([0.5 - 0.5*bbox_scale[:,0], 0.5 + 0.5 * bbox_scale[:,1]], 1)
    source_bbox_center = tf.reduce_mean(source_bbox, 1)
    pad_size = [int(Sd.value * 0.5), int(Sh.value * 0.5), int(Sw.value * 0.5)]
    padding = [[pad_size[0], pad_size[0]], \
                [pad_size[1], pad_size[1]], \
                [pad_size[2], pad_size[2]], \
                [0,0]]
    pad_mask =  tf.pad(mask, tf.constant(padding))
    Sd_p, Sh_p, Sw_p, c = pad_mask.get_shape()

    source_bbox_center_pad = tf.divide(source_bbox_center * tf.constant([Sd.value, Sh.value, Sw.value], dtype=tf.float32)\
    + tf.constant(pad_size, dtype=tf.float32), tf.constant([Sd_p.value, Sh_p.value, Sw_p.value], dtype=tf.float32))

    source_bbox_dhw = 0.5*(bbox_scale[:,0] + bbox_scale[:,1]) *\
        tf.constant([Sd.value/Sd_p.value, Sh.value/Sh_p.value, Sw.value/Sw_p.value], dtype=tf.float32)

    d3_bbox = tf.stack([source_bbox_center_pad - source_bbox_dhw * 0.5, source_bbox_center_pad + source_bbox_dhw * 0.5], 1)
    d3_bbox_yx = tf.tile(tf.reshape(tf.transpose(d3_bbox[1:,:], [1, 0]), [1, -1]), [Sd_p, 1])
    #bbox = tf.concat([tf.zeros((Sd, 2)), tf.ones((Sd, 2))], 1)
    crop_image_yx = tf.case({
        tf.equal(valid, 0): lambda:tf.zeros((1,1,1,c)),
        tf.not_equal(valid, 0): lambda:tfutil.image_crop_and_resize(pad_mask, d3_bbox_yx, tf.range(Sd_p), dhw_d[1:])}, exclusive=True)
    Snh = tf.shape(crop_image_yx)[1]
    Snw = tf.shape(crop_image_yx)[2]

    #bbox = tf.concat([tf.zeros((Snh, 2)), tf.ones((Snh, 2))], 1)
    d3_bbox_zx = tf.tile(tf.reshape(tf.stack([d3_bbox[0,:], tf.constant([0, 1], dtype=tf.float32)], 1), [1,-1]), [Snh, 1])
    resize_voxel = tf.case({
        tf.equal(valid, 0): lambda:tf.zeros((1,1,1, c)),
        tf.not_equal(valid, 0): lambda:tfutil.image_crop_and_resize(tf.transpose(crop_image_yx, [1, 0, 2, 3]), d3_bbox_zx,
        tf.range(Snh), [dhw_d[0], dhw_d[2]])},
        exclusive=True)
    resize_voxel = tf.transpose(resize_voxel, [1, 0, 2, 3])

    pad = tf.stack([roi_indices_d[:,0], output_size - roi_indices_d[:,1]], 1)
    pad = tf.concat([pad, tf.zeros((1,2), dtype=tf.int32)], 0)
    full_size_voxel = tf.case({
        tf.equal(valid, 0): lambda:tf.zeros((output_size, output_size, output_size, c)),
        tf.not_equal(valid, 0): lambda:tf.pad(resize_voxel, pad, constant_values=0)}, exclusive=True)
    return full_size_voxel, roi_indices_d #, bbox_scale, d3_bbox

def crop_mask_to_full_mask(roi_indices, class_labels, mask, output_size):
    """
    Deprecated: please use crop_mask_to_full_mask_precise
    """
    roi_indices = tf.stack([tf.round(roi_indices[:,0] * tf.cast(output_size, dtype=tf.float32)),
        tf.round(roi_indices[:,1] * tf.cast(output_size, dtype=tf.float32))], 1)
    #roi_indices = tf.stack([tf.floor(roi_indices[:,0] * tf.cast(output_size, dtype=tf.float32)),
    #    tf.ceil(roi_indices[:,1] * tf.cast(output_size, dtype=tf.float32))], 1)
    roi_indices = tf.clip_by_value(tf.cast(roi_indices, dtype=tf.int32), 0, output_size)
    Sd, Sh, Sw, c = mask.get_shape()
    # make sure rois are valid
    roi_indices = tf.stack([roi_indices[:,0],
        tf.maximum(roi_indices[:,1], roi_indices[:,0])], 1)

    dhw = roi_indices[:,1] - roi_indices[:,0]
    valid = tf.cast(class_labels, dtype=tf.float32) * tf.cast(tf.greater(dhw[0], 0), dtype=tf.float32) * tf.cast(tf.greater(dhw[1], 0), dtype=tf.float32) * tf.cast(tf.greater(dhw[2], 0), dtype=tf.float32) 

    bbox = tf.concat([tf.zeros((Sd, 2)), tf.ones((Sd, 2))], 1)
    crop_image_yx = tf.case({
        tf.equal(valid, 0): lambda:tf.zeros((1, 1, 1,c)),
        tf.not_equal(valid, 0): lambda:tfutil.image_crop_and_resize(mask, bbox, tf.range(Sd), dhw[1:])}, exclusive=True)
    Snh = tf.shape(crop_image_yx)[1]
    Snw = tf.shape(crop_image_yx)[2]

    bbox = tf.concat([tf.zeros((Snh, 2)), tf.ones((Snh, 2))], 1)
    resize_voxel = tf.case({
        tf.equal(valid, 0): lambda:tf.zeros((1,1,1, c)),
        tf.not_equal(valid, 0): lambda:tfutil.image_crop_and_resize(tf.transpose(crop_image_yx, [1, 0, 2, 3]), bbox,
        tf.range(Snh), [dhw[0], dhw[2]])},
        exclusive=True)
    resize_voxel = tf.transpose(resize_voxel, [1, 0, 2, 3])

    pad = tf.stack([roi_indices[:,0], output_size - roi_indices[:,1]], 1)
    pad = tf.concat([pad, tf.zeros((1,2), dtype=tf.int32)], 0)
    full_size_voxel = tf.case({
        tf.equal(valid, 0): lambda:tf.zeros((output_size, output_size, output_size, c)),
        tf.not_equal(valid, 0): lambda:tf.pad(resize_voxel, pad, constant_values=0)}, exclusive=True)
    return full_size_voxel, roi_indices

def resize_voxel(voxel, scale, threshold=0.5):
    """
    resize voxels to a desired scale
    voxel: [batch_size x Sd x Sh x Sw x c]
    scale: float
    """
    bs = tf.shape(voxel)[0]
    _, Sd, Sh, Sw, c = voxel.get_shape()
    sd = int(Sd.value * scale)
    sh = int(Sh.value * scale)
    sw = int(Sw.value * scale)

    crop_box = tf.tile(tf.constant([[0, 0, 0, 1, 1, 1]], dtype=tf.float32), [bs, 1])
    box_ind = tf.range(bs)

    crop_voxels = tfutil.tensor3d_crop_and_resize(voxel, crop_box, box_ind, tf.constant([sd, sh, sw]))

    return crop_voxels

def crop_and_resize_3d_box2_pad(voxels, crop_box, crop_size, debug=False):
    """
    crop a region in 3d tensor and resize it to a tensor with size\
    (crop_size x crop_size x crop_size), if box exceeds tensor boundary,
    use padding to fill the empty space
    voxels: [batch_size x Sd x Sh x Sw x c]
    crop_box: [batch_size x nrois x 3 x 2]
    crop_size: an integer
    """
    crop_box_min = crop_box[:,:,:,0]
    crop_box_len = tf.expand_dims(crop_box[:,:,:,1] - crop_box[:,:,:,0], -1)
    crop_box_clip_min = tf.clip_by_value(crop_box[:,:,:,0], 0, 1)
    crop_box_clip_max = tf.clip_by_value(crop_box[:,:,:,1], 0, 1)

    crop_feat = crop_and_resize_3d_box2(voxels, crop_box, crop_size, debug=False)

    crop_box_in_ori_box = tf.divide(tf.stack([crop_box_clip_min, crop_box_clip_max], -1)\
                          - tf.expand_dims(crop_box_min, -1), crop_box_len + 0.00000000000001)

    feat_c = crop_feat.get_shape()[-1]
    bs, nobjs, _, _ = crop_box_in_ori_box.get_shape()
    crop_box_in_ori_box = tf.reshape(crop_box_in_ori_box, [bs * nobjs, 3, 2])
    crop_feat = tf.reshape(crop_feat, [bs * nobjs, crop_size, crop_size, crop_size, feat_c]) 
    final_out = crop_mask_to_full_mask_precise_bs(crop_box_in_ori_box, crop_feat, crop_size)

    final_out = tf.reshape(final_out, [bs, nobjs, crop_size, crop_size, crop_size, feat_c]) 
    return final_out

def crop_and_resize_3d_box2(voxels, crop_box, crop_size, debug=False):
    """
    crop a region in 3d tensor and resize it to a tensor with size (crop_size x crop_size x crop_size)
    voxels: [batch_size x Sd x Sh x Sw x c]
    crop_box: [batch_size x nrois x 3 x 2]
    crop_size: an integer
    """
    batch_size, Sd, Sh, Sw, voxel_c = voxels.get_shape()
    batch_size = tf.shape(voxels)[0]
    nrois = tf.shape(crop_box)[1]
    crop_box = tf.reshape(tf.transpose(crop_box, [0,1,3,2]), [-1, 6])
    box_ind = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [-1,1]), [1, nrois]), [-1])
    crop_voxels = tfutil.tensor3d_crop_and_resize(voxels, crop_box, box_ind,\
        tf.constant([crop_size, crop_size, crop_size], dtype=tf.int32))
    crop_voxels = tf.reshape(crop_voxels, [batch_size, nrois, crop_size, crop_size, crop_size, voxel_c])
    return crop_voxels

def crop_and_resize_3d_box(voxels, crop_box, crop_size, debug=False):
    """
    Deprecated, please use crop_and_resize_3d_bbox2
    crop a region in 3d tensor and resize it to a tensor with size (crop_size x crop_size x crop_size)
    voxels: [batch_size x Sd x Sh x Sw x c]
    crop_box: [batch_size x nrois x 3 x 2]
    crop_size: an integer
    """

    crop_box = tf.stack([crop_box[:,:,:,0], crop_box[:,:,:,1]], 3)
    bs, Sd, Sh, Sw, c = voxels.get_shape()
    _, nrois, _, _ = crop_box.get_shape()
    bs = tf.shape(voxels)[0]
    Sd = tf.shape(voxels)[1]
    nrois = tf.shape(crop_box)[1]
    #crop along z axis
    # y1,x1,y2,x2
    crop_box_yx = tf.reshape(tf.tile(tf.expand_dims(tf.transpose(\
            crop_box[:, :, 1:, :], [0, 1, 3, 2]), 1), [1, Sd, 1, 1, 1]), [-1, 4])
    #if debug:
    #    import ipdb; ipdb.set_trace()
    box_idx = tf.reshape(tf.tile(tf.reshape(tf.range(bs * Sd), [-1, 1]), [1, nrois]), [-1])
    # [batch_sizexSd, Sh, Sw, c]
    image_to_crop_yx = tf.reshape(voxels, [-1, Sh, Sw, c])
    yx_image = tfutil.image_crop_and_resize(image_to_crop_yx, crop_box_yx, box_idx, [crop_size, crop_size])
    # [bs x Sd x nrois] x crop_size x crop_size x c
    # -> [bs x nois] x Sd x crop_size x crop_size x c
    if debug:
        return tf.reshape(yx_image, [bs, -1, crop_size, crop_size, c])
    yx_image = tf.reshape(tf.transpose(tf.reshape(yx_image, [-1, Sd, nrois, crop_size, crop_size, c]), [0, 2, 1, 3, 4, 5]), [-1, Sd, crop_size, crop_size, c])
    crop_box_zx = tf.reshape(tf.tile(tf.expand_dims(tf.stack([crop_box[:,:,0, 0], tf.zeros((bs, nrois)),\
        crop_box[:,:,0, 1], tf.ones((bs, nrois))], 2), 2), [1, 1, crop_size, 1]), [-1, 4])
    image_to_crop_zx = tf.reshape(tf.transpose(yx_image, [0, 2, 1, 3, 4]), [-1, Sd, crop_size, c])
    #box_idx = tf.reshape(tf.tile(tf.reshape(tf.range(bs * crop_size), [-1, 1]), [1, nrois]), [-1])
    box_idx = tf.range(bs * nrois * crop_size)
    zx_image = tfutil.image_crop_and_resize(image_to_crop_zx, crop_box_zx, box_idx, \
        [crop_size, crop_size])
    zx_image = tf.transpose(tf.reshape(zx_image, [bs, nrois, crop_size, crop_size, crop_size, c]),\
        [0, 1, 3, 2, 4, 5])
    return zx_image


def getbs(voxel):
    return int(voxel.shape[0])

def getsize(voxel):
    return int(voxel.shape[1])

def interpolate_grid(in_size, x, y, z, out_size):
    """Bilinear interploation layer.

    Args:
        insize: A 4-dim list [num_batch, depth, height, width].
                It is the input volume for the transformation layer (float).
        x: A tensor of size [num_batch, out_depth, out_height, out_width]
                representing the inverse coordinate mapping for x (tf.float32).
        y: A tensor of size [num_batch, out_depth, out_height, out_width]
                representing the inverse coordinate mapping for y (tf.float32).
        z: A tensor of size [num_batch, out_depth, out_height, out_width]
                representing the inverse coordinate mapping for z (tf.float32).
        out_size: A tuple representing the output size of transformation layer
                (float).

    Returns:
    A transformed tensor (tf.float32).

    """
    with tf.variable_scope('_interpolate'):
        """
        num_batch = im.get_shape().as_list()[0]
        depth = im.get_shape().as_list()[1]
        height = im.get_shape().as_list()[2]
        width = im.get_shape().as_list()[3]
        channels = im.get_shape().as_list()[4]
        """
        num_batch = in_size[0]
        depth = in_size[1]
        height = in_size[2]
        width = in_size[3]

        x = tf.to_float(x)
        y = tf.to_float(y)
        z = tf.to_float(z)
        depth_f = tf.to_float(depth)
        height_f = tf.to_float(height)
        width_f = tf.to_float(width)
        # Number of disparity interpolated.
        out_depth = out_size[0]
        out_height = out_size[1]
        out_width = out_size[2]
        zero = tf.zeros([], dtype='int32')
            # 0 <= z < depth, 0 <= y < height & 0 <= x < width.
        max_z = tf.to_int32(depth - 1)
        max_y = tf.to_int32(height - 1)
        max_x = tf.to_int32(width - 1)

        # Converts scale indices from [-1, 1] to [0, width/height/depth].
        x = (x + 1.0) * (width_f - 1.0) / 2.0
        y = (y + 1.0) * (height_f - 1.0) / 2.0
        z = (z + 1.0) * (depth_f - 1.0) / 2.0

        x0 = tf.to_int32(tf.floor(x))
        x1 = x0 + 1
        y0 = tf.to_int32(tf.floor(y))
        y1 = y0 + 1
        z0 = tf.to_int32(tf.floor(z))
        z1 = z0 + 1

        x0_clip = tf.clip_by_value(x0, zero, max_x)
        x1_clip = tf.clip_by_value(x1, zero, max_x)
        y0_clip = tf.clip_by_value(y0, zero, max_y)
        y1_clip = tf.clip_by_value(y1, zero, max_y)
        z0_clip = tf.clip_by_value(z0, zero, max_z)
        z1_clip = tf.clip_by_value(z1, zero, max_z)
        dim3 = width
        dim2 = width * height
        dim1 = width * height * depth

        #repeat can only be run on cpu
        #base = _repeat(
        #    tf.range(num_batch) * dim1, out_depth * out_height * out_width)
        base = tf.constant(
            np.concatenate([np.array([i] * out_depth * out_height * out_width)
                            for i in range(num_batch)]).astype(np.int32)
        )
        #only works for bs = 1
        #base = tf.zeros((out_depth * out_height * out_width), dtype=tf.int32)

        base_z0_y0 = base * dim1 + z0_clip * dim2 + y0_clip * dim3
        base_z0_y1 = base * dim1 + z0_clip * dim2 + y1_clip * dim3
        base_z1_y0 = base * dim1 + z1_clip * dim2 + y0_clip * dim3
        base_z1_y1 = base * dim1 + z1_clip * dim2 + y1_clip * dim3

        idx_z0_y0_x0 = base_z0_y0 + x0_clip
        idx_z0_y0_x1 = base_z0_y0 + x1_clip
        idx_z0_y1_x0 = base_z0_y1 + x0_clip
        idx_z0_y1_x1 = base_z0_y1 + x1_clip
        idx_z1_y0_x0 = base_z1_y0 + x0_clip
        idx_z1_y0_x1 = base_z1_y0 + x1_clip
        idx_z1_y1_x0 = base_z1_y1 + x0_clip
        idx_z1_y1_x1 = base_z1_y1 + x1_clip


        # Finally calculate interpolated values.
        x0_f = tf.to_float(x0)
        x1_f = tf.to_float(x1)
        y0_f = tf.to_float(y0)
        y1_f = tf.to_float(y1)
        z0_f = tf.to_float(z0)
        z1_f = tf.to_float(z1)
        # Check the out-of-boundary case.
        x0_valid = tf.to_float(
            tf.less_equal(x0, max_x) & tf.greater_equal(x0, 0))
        x1_valid = tf.to_float(
            tf.less_equal(x1, max_x) & tf.greater_equal(x1, 0))
        y0_valid = tf.to_float(
            tf.less_equal(y0, max_y) & tf.greater_equal(y0, 0))
        y1_valid = tf.to_float(
            tf.less_equal(y1, max_y) & tf.greater_equal(y1, 0))
        z0_valid = tf.to_float(
            tf.less_equal(z0, max_z) & tf.greater_equal(z0, 0))
        z1_valid = tf.to_float(
            tf.less_equal(z1, max_z) & tf.greater_equal(z1, 0))

        w_z0_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                     (z1_f - z) * x1_valid * y1_valid * z1_valid),
                                    1)
        w_z0_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                     (z1_f - z) * x0_valid * y1_valid * z1_valid),
                                    1)
        w_z0_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                         (z1_f - z) * x1_valid * y0_valid * z1_valid),
                                    1)
        w_z0_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                     (z1_f - z) * x0_valid * y0_valid * z1_valid),
                                    1)
        w_z1_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                     (z - z0_f) * x1_valid * y1_valid * z0_valid),
                                    1)
        w_z1_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                     (z - z0_f) * x0_valid * y1_valid * z0_valid),
                                    1)
        w_z1_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                     (z - z0_f) * x1_valid * y0_valid * z0_valid),
                                    1)
        w_z1_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                     (z - z0_f) * x0_valid * y0_valid * z0_valid),
                                    1)
        weights_summed = (
            w_z0_y0_x0 +
            w_z0_y0_x1 +
            w_z0_y1_x0 +
            w_z0_y1_x1 +
            w_z1_y0_x0 +
            w_z1_y0_x1 +
            w_z1_y1_x0 +
            w_z1_y1_x1
        )
        idxs = (idx_z0_y0_x0,
                idx_z0_y0_x1,
                idx_z0_y1_x0,
                idx_z0_y1_x1,
                idx_z1_y0_x0,
                idx_z1_y0_x1,
                idx_z1_y1_x0,
                idx_z1_y1_x1)
        ws = (w_z0_y0_x0,
              w_z0_y0_x1,
              w_z0_y1_x0,
              w_z0_y1_x1,
              w_z1_y0_x0,
              w_z1_y0_x1,
              w_z1_y1_x0,
              w_z1_y1_x1)
        return idxs, ws

def meshgrid(depth, height, width, z_near, z_far):
    # from voxel to canvas
    # so mapping is from xyz -> XYZ
    # xyz: [-1, 1] -> XYZ (boundary_to_center)
    with tf.variable_scope('_meshgrid'):
        x_t = tf.reshape(
            tf.tile(tf.linspace(-1.0, 1.0, width), [height * depth]),
            [depth, height, width])
        y_t = tf.reshape(
            tf.tile(tf.linspace(-1.0, 1.0, height), [width * depth]),
            [depth, width, height])
        y_t = tf.transpose(y_t, [0, 2, 1])
        sample_grid = tf.tile(
            tf.linspace(float(z_near), float(z_far), depth), [width * height])
        z_t = tf.reshape(sample_grid, [height, width, depth])
        z_t = tf.transpose(z_t, [2, 0, 1])

        z_t = 1 / z_t
        d_t = 1 / z_t
        x_t /= z_t
        y_t /= z_t

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        d_t_flat = tf.reshape(d_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([d_t_flat, y_t_flat, x_t_flat, ones], 0)
        return grid

def noproj_meshgrid(depth, height, width, z_near, z_far): # avoid using this function
    with tf.variable_scope('_meshgrid'):
        x_t = tf.reshape(
            tf.tile(tf.linspace(-const.boundary_to_center, const.boundary_to_center, width), [height * depth]),
            [depth, height, width])
        y_t = tf.reshape(
            tf.tile(tf.linspace(-const.boundary_to_center, const.boundary_to_center, height), [width * depth]),
            [depth, width, height])
        y_t = tf.transpose(y_t, [0, 2, 1])

        sample_grid = tf.tile(
            tf.linspace(float(z_near), float(z_far), depth), [width * height])
        z_t = tf.reshape(sample_grid, [height, width, depth])
        z_t = tf.transpose(z_t, [2, 0, 1])

        z_t = 1 / z_t
        d_t = 1 / z_t

        #originally: X = x*z/fx
        #x_t /= z_t
        #y_t /= z_t

        #new change
        x_t *= const.focal_length
        y_t *= const.focal_length

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        d_t_flat = tf.reshape(d_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([d_t_flat, y_t_flat, x_t_flat, ones], 0)
        return grid

def invproj_meshgrid(depth, height, width, z_near, z_far):
    with tf.variable_scope('_meshgrid'):
        # Here I use [-1, 1] since we crop a box of const.bounadry_to_center^3, but eventuallt we resize it to
        x_t = tf.reshape(
            tf.tile(tf.linspace(-const.boundary_to_center, const.boundary_to_center, width), [height * depth]), #const.boundary_to_center, const.boundary_center
            [depth, height, width])
        y_t = tf.reshape(
            tf.tile(tf.linspace(-const.boundary_to_center, const.boundary_to_center, height), [width * depth]),
            [depth, width, height])
        y_t = tf.transpose(y_t, [0, 2, 1])

        sample_grid = tf.tile(
            tf.linspace(float(z_near), float(z_far), depth), [width * height])
        z_t = tf.reshape(sample_grid, [height, width, depth])
        z_t = tf.transpose(z_t, [2, 0, 1])

        z_t = 1 / z_t
        d_t = 1 / z_t

        #originally: X = x*z/fx
        #x_t /= z_t
        #y_t /= z_t

        #new change
        x_t *= z_t
        y_t *= z_t

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        d_t_flat = tf.reshape(d_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([d_t_flat, y_t_flat, x_t_flat, ones], 0)
        return grid

def interpolate_with_idx(im, idxs, ws):
    """Bilinear interploation layer.

    Args:
        im: A 5D tensor of size [num_batch, depth, height, width, num_channels].
            It is the input volume for the transformation layer (tf.float32).
    Returns:
    A transformed tensor (tf.float32).

    """
    idx_z0_y0_x0,\
        idx_z0_y0_x1,\
        idx_z0_y1_x0,\
        idx_z0_y1_x1,\
        idx_z1_y0_x0,\
        idx_z1_y0_x1,\
        idx_z1_y1_x0,\
        idx_z1_y1_x1 = idxs

    w_z0_y0_x0,\
        w_z0_y0_x1,\
        w_z0_y1_x0,\
        w_z0_y1_x1,\
        w_z1_y0_x0,\
        w_z1_y0_x1,\
        w_z1_y1_x0,\
        w_z1_y1_x1 = ws

    num_batch, out_depth, out_height, out_width, channels = im.get_shape().as_list()
    # Use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.to_float(im_flat)
    i_z0_y0_x0 = tf.gather(im_flat, idx_z0_y0_x0)
    i_z0_y0_x1 = tf.gather(im_flat, idx_z0_y0_x1)
    i_z0_y1_x0 = tf.gather(im_flat, idx_z0_y1_x0)
    i_z0_y1_x1 = tf.gather(im_flat, idx_z0_y1_x1)
    i_z1_y0_x0 = tf.gather(im_flat, idx_z1_y0_x0)
    i_z1_y0_x1 = tf.gather(im_flat, idx_z1_y0_x1)
    i_z1_y1_x0 = tf.gather(im_flat, idx_z1_y1_x0)
    i_z1_y1_x1 = tf.gather(im_flat, idx_z1_y1_x1)

    output = tf.add_n([
       w_z0_y0_x0 * i_z0_y0_x0, w_z0_y0_x1 * i_z0_y0_x1,
       w_z0_y1_x0 * i_z0_y1_x0, w_z0_y1_x1 * i_z0_y1_x1,
       w_z1_y0_x0 * i_z1_y0_x0, w_z1_y0_x1 * i_z1_y0_x1,
       w_z1_y1_x0 * i_z1_y1_x0, w_z1_y1_x1 * i_z1_y1_x1
    ])

    output = tf.reshape(
                output,
             tf.stack([num_batch, out_depth, out_height, out_width, channels]))

    return output


def transformer(voxels,
                theta,
                out_size,
                z_near,
                z_far,
                name='PerspectiveTransformer',
                do_project = True):

    BS = getbs(voxels)
    # global __k
    # if not const.eager:
    #     print(('TRANSFORMER WAS CALLED %d' % __k))
    # __k+=1
    
    """Perspective Transformer Layer.

    Args:
        voxels: A tensor of size [num_batch, depth, height, width, num_channels].
            It is the output of a deconv/upsampling conv network (tf.float32).
        theta: A tensor of size [num_batch, 16].
            It is the inverse camera transformation matrix (tf.float32).
        out_size: A tuple representing the size of output of
            transformer layer (float).
        z_near: A number representing the near clipping plane (float).
        z_far: A number representing the far clipping plane (float).

    Returns:
        A transformed tensor (tf.float32).

    """
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([
                    n_repeats,
                ])), 1), [1, 0])
            rep = tf.to_int32(rep)
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, z, out_size):
        """Bilinear interploation layer.

        Args:
            im: A 5D tensor of size [num_batch, depth, height, width, num_channels].
                It is the input volume for the transformation layer (tf.float32).
            x: A tensor of size [num_batch, out_depth, out_height, out_width]
                representing the inverse coordinate mapping for x (tf.float32).
            y: A tensor of size [num_batch, out_depth, out_height, out_width]
                representing the inverse coordinate mapping for y (tf.float32).
            z: A tensor of size [num_batch, out_depth, out_height, out_width]
                representing the inverse coordinate mapping for z (tf.float32).
            out_size: A tuple representing the output size of transformation layer
                (float).

        Returns:
        A transformed tensor (tf.float32).

        """
        idxs, ws = interpolate_grid(im.get_shape()[:4], x, y, z, out_size)
        output = interpolate_with_idx(im, idxs, ws)
            
        return output


    def _transform(theta, input_dim, out_size, z_near, z_far):
        """
        input_dim: 10 x 64 x 64 x 64 x 34
        """
        with tf.variable_scope('_transform'):
            num_batch = input_dim.get_shape().as_list()[0]
            num_channels = input_dim.get_shape().as_list()[4]
            theta = tf.reshape(theta, (-1, 4, 4))
            theta = tf.cast(theta, 'float32')

            out_depth = out_size[0]
            out_height = out_size[1]
            out_width = out_size[2]

            if do_project is True:
                grid = meshgrid(out_depth, out_height, out_width, z_near, z_far)
            elif do_project == 'invert':
                grid = invproj_meshgrid(out_depth, out_height, out_width, z_near, z_far)
            else: # don't use this to rotate voxel, please use rotate_voxel2
                grid = noproj_meshgrid(out_depth, out_height, out_width, z_near, z_far)
            grid_tmp = grid
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 4, -1]))

            #grid = tfpy.summarize_tensor(grid, 'grid')

            def printgrid(grid_):
                #z in 3, 5
                #x/y in 5, -5
                zs = grid_[:, 0, :]
                print('===')
                print(zs.shape)
                print(np.mean(zs))
                print(np.max(zs))
                print(np.min(zs))

            #grid = tfpy.inject_callback(grid, printgrid)
            # Transform A x (x_t', y_t', 1, d_t)^T -> (x_s, y_s, z_s, 1).
            t_g = tf.matmul(theta, grid)

            #z_s = tf.slice(t_g, [0, 0, 0], [-1, 1, -1])
            #y_s = tf.slice(t_g, [0, 1, 0], [-1, 1, -1])
            #x_s = tf.slice(t_g, [0, 2, 0], [-1, 1, -1])
            #this gives a different shape, but it'll be reshaped anyway
            z_s = t_g[:, 0, :]
            y_s = t_g[:, 1, :]
            x_s = t_g[:, 2, :]

            z_s = z_s * (2/(z_far - z_near))
            if do_project is not "invert":
                y_s = y_s * (1/const.boundary_to_center)
                x_s = x_s * (1/const.boundary_to_center)

            #z_s = tfpy.summarize_tensor(z_s, 'z_s') #-1, 1
            #y_s = tfpy.summarize_tensor(y_s, 'y_s') #-1.34, 1.34
            #x_s = tfpy.summarize_tensor(x_s, 'x_s')

            z_s_flat = tf.reshape(z_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            x_s_flat = tf.reshape(x_s, [-1])

            idxs, ws = interpolate_grid(input_dim.get_shape()[:4], x_s_flat,
                                        y_s_flat, z_s_flat, out_size)

            output = interpolate_with_idx(input_dim, idxs, ws)
            #input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, z_s_flat,
            #                                 out_size)


            return output #, {"x_debug": x_debug, "y_debug":y_debug, "z_debug": z_debug}

    with tf.variable_scope(name):
        #with tf.device('/cpu:0'):
        output = _transform(theta, voxels, out_size, z_near, z_far)
        return output

class rotate_helper_concat:
    def __init__(self, HV, VV, MINH, MINV, HDELTA, VDELTA):
        """
        in_size: (depth, height, width)
        out_size: (depth, height, width)
        """
        self.prepare_info(HV, VV, MINH, MINV, HDELTA, VDELTA)

    def prepare_info(self, HV, VV, MINH, MINV, HDELTA, VDELTA):
        self.num_batch = HV * VV
        dthetas = np.tile(np.array([(MINH + i * HDELTA) * (1/180.0) for i in range(HV)]), [VV, 1])
        self.thetas = tf.constant(np.ndarray.flatten(dthetas), dtype=tf.float32)

        phis = np.tile(np.array([(MINV + i * VDELTA) * (1/180.0) for i in range(VV)]).reshape([-1, 1]),
                                 [1, HV])

        self.phis = tf.constant(np.ndarray.flatten(phis), dtype = tf.float32)
        self.cam_pos = tf.stack([self.thetas, self.phis], 1)

    def concat(self, context_features):
        context = []
        for layer_features in context_features:
            num_views = len(layer_features)
            lfeat = tf.tile(tf.expand_dims(tf.concat(layer_features, 0), 1), [1, self.num_batch, 1, 1, 1])

            bs, _, h, w, c = lfeat.get_shape()
            tiled_cam_pos = tf.tile(tf.reshape(self.cam_pos, [1, self.num_batch, 1, 1, 2]),
                                [bs, 1, h, w, 1])
            concat_lfeat = tf.concat([lfeat, tiled_cam_pos], 4)
            _, _, h, w, c = concat_lfeat.get_shape()
            concat_lfeat = tf.reshape(concat_lfeat, [-1, h, w, c])
            context.append(tf.split(concat_lfeat, num_views))
        return context
    def reshape_bs_nr(self, input):
        input_tmp = []
        for layer_feat in input:
            layer_feat_tmp = []
            for feat in layer_feat:
                bs, h, w, c = feat.get_shape()
                layer_feat_tmp.append(tf.reshape(feat, [-1, self.num_batch, h, w, c]))
            input_tmp.append(layer_feat_tmp)
        return input_tmp

class rotate_helper:
    """
    This helper precomputed a fixed grid of indices for rotation.
    This is suitable when you always want to rotate a fixed list of angles for all
    the voxels in the batch and you don't want to compute the rotation matrix
    everytimes you want to transform

    """
    def __init__(self, in_size, out_size, HV, VV, MINH, MINV, HDELTA, VDELTA):
        """
        in_size: (depth, height, width)
        out_size: (depth, height, width)
        """
        self.prepare_info(in_size, out_size, HV, VV, MINH, MINV, HDELTA, VDELTA)

    def prepare_info(self, in_size, out_size, HV, VV, MINH, MINV, HDELTA, VDELTA):
        self.num_batch = HV * VV
        self.in_size = in_size
        dthetas = np.tile(np.array([0 - (MINH + i * HDELTA) for i in range(HV)]), [VV, 1])
        self.dthetas = tf.constant(np.ndarray.flatten(dthetas), dtype=tf.float32)

        phi1s = np.tile(np.array([MINV + i * VDELTA for i in range(VV)]).reshape([-1, 1]),
                                 [1, HV])
        self.phi1s = tf.constant(np.ndarray.flatten(phi1s), dtype = tf.float32)
        self.phi2s = tf.constant(np.zeros(HV * VV), dtype = tf.float32)

        rot_mat_1 = get_transform_matrix_tf([0.0] * self.num_batch, -self.phi1s)
        rot_mat_2 = get_transform_matrix_tf(self.dthetas, self.phi2s)

        self.idxs1, self.ws1 = self._rotation_to_idx(
           in_size,
           tf.reshape(rot_mat_1, (self.num_batch, 16)),
           out_size,
           3.0,
           5.0,
           do_project = False,
        )

        self.idxs2, self.ws2 = self._rotation_to_idx(
           in_size,
           tf.reshape(rot_mat_2, (self.num_batch, 16)),
           out_size,
           3.0,
           5.0,
           do_project = False,
        )


    def _rotation_to_idx(self, in_size, theta, out_size, z_near, z_far, do_project):
       with tf.variable_scope('transform'):
            num_batch = self.num_batch
            theta = tf.reshape(theta, (-1, 4, 4))
            theta = tf.cast(theta, 'float32')

            out_depth = out_size[0]
            out_height = out_size[1]
            out_width = out_size[2]

            if do_project is True:
                grid = meshgrid(out_depth, out_height, out_width, z_near, z_far)
            elif do_project == 'invert':
                grid = invproj_meshgrid(out_depth, out_height, out_width, z_near, z_far)
            else:
                grid = noproj_meshgrid(out_depth, out_height, out_width, z_near, z_far)
            grid_tmp = grid
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 4, -1]))

            t_g = tf.matmul(theta, grid)

            z_s = t_g[:, 0, :]
            y_s = t_g[:, 1, :]
            x_s = t_g[:, 2, :]

            z_s_flat = tf.reshape(z_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            x_s_flat = tf.reshape(x_s, [-1])

            return interpolate_grid((num_batch, out_depth, out_height, out_width), x_s_flat,
                                        y_s_flat, z_s_flat, out_size)
    def transform(self, voxel):

        """
        voxel: tensor with size [num_batch, depth, height, width, channel]
        """

        bs, out_depth, out_height, out_width, c = voxel.get_shape().as_list()
        voxel_tmp = tf.expand_dims(tf.transpose(voxel, [1, 2, 3, 4, 0]), 0)
        voxel_tmp = tf.reshape(tf.tile(voxel_tmp, [self.num_batch, 1, 1, 1, 1, 1]),
                               [self.num_batch, out_depth, out_height, out_width, -1])

        foo = interpolate_with_idx(voxel_tmp, self.idxs1, self.ws1)
        input_transformed = interpolate_with_idx(foo, self.idxs2, self.ws2)

        output = tf.reshape(
            input_transformed,
            tf.stack([self.num_batch, out_depth, out_height, out_width, c, bs]))
        output = tf.transpose(output, [5, 0, 1, 2, 3, 4])

        return output #, {"x_debug": x_debug, "y_debug":y_debug, "z_debug": z_debug}

def get_transform_matrix_tf(theta, phi, invert_rot = False, invert_focal = False):
    if isinstance(theta, list):
        theta = tf.stack(theta)
    if isinstance(phi, list):
        phi = tf.stack(phi)
        
    return tf.map_fn(
        lambda t_p: get_transform_matrix_tf_(t_p[0], t_p[1], invert_rot, invert_focal),
        [theta, phi],
        parallel_iterations = 1000,
        dtype = tf.float32
    )
    
def get_transform_matrix_tf_(theta, phi, invert_rot = False, invert_focal = False):
    #INPUT IN DEGREES
    
    #extrinsic matrix:
    #
    # RRRD
    # RRRD
    # RRRD
    # 000D

    sin_phi = tf.sin(phi / 180 * np.pi)
    cos_phi = tf.cos(phi / 180 * np.pi)
    sin_theta = tf.sin(theta / 180.0 * np.pi) #why is theta negative???
    cos_theta = tf.cos(theta / 180.0 * np.pi)

    #these are inverted from normal!
    rotation_azimuth_flat = [
        cos_theta, 0.0, -sin_theta,
        0.0, 1.0, 0.0,
        sin_theta, 0.0, cos_theta
    ]

    rotation_elevation_flat = [
        cos_phi, sin_phi, 0.0,
        -sin_phi, cos_phi, 0.0,
        0.0, 0.0, 1.0
    ]

    f = lambda x: tf.reshape(tf.stack(x), (3, 3))
    rotation_azimuth = f(rotation_azimuth_flat)
    rotation_elevation = f(rotation_elevation_flat)

    rotation_matrix = tf.matmul(rotation_azimuth, rotation_elevation)
    if invert_rot:
        rotation_matrix = tf.linalg.inv(rotation_matrix)

    displacement = np.zeros((3, 1), dtype=np.float32)
    displacement[0, 0] = const.radius
    displacement = tf.constant(displacement, dtype = np.float32)
    displacement = tf.matmul(rotation_matrix, displacement)

    bottom_row = np.zeros((1, 4), dtype = np.float32)
    bottom_row[0,3] = 1.0
    bottom_row = tf.constant(bottom_row)

    #print rotation_matrix
    #print bottom_row
    #print displacement
    
    extrinsic_matrix = tf.concat([
        tf.concat([rotation_matrix, -displacement], axis = 1),
        bottom_row
    ], axis = 0)
    
    if invert_focal:
        intrinsic_diag = [1.0, float(const.focal_length), float(const.focal_length), 1.0]
    else:
        intrinsic_diag = [1.0, 1.0/float(const.focal_length), 1.0/float(const.focal_length), 1.0]
    intrinsic_matrix = tf.diag(tf.constant(intrinsic_diag, dtype = tf.float32))
    
    camera_matrix = tf.matmul(extrinsic_matrix, intrinsic_matrix)
    return camera_matrix
    
def get_transform_matrix(theta, phi, invert_rot = False, invert_focal = False, debug=False):
    """Get the 4x4 Perspective Transfromation matrix used for PTN."""

    #extrinsic x intrinsic
    camera_matrix = np.zeros((4, 4), dtype=np.float32)

    intrinsic_matrix = np.eye(4, dtype=np.float32)
    extrinsic_matrix = np.eye(4, dtype=np.float32)

    sin_phi = np.sin(float(phi) / 180.0 * np.pi)
    cos_phi = np.cos(float(phi) / 180.0 * np.pi)
    sin_theta = np.sin(float(-theta) / 180.0 * np.pi)
    cos_theta = np.cos(float(-theta) / 180.0 * np.pi)

    #theta rotation
    rotation_azimuth = np.zeros((3, 3), dtype=np.float32)
    rotation_azimuth[0, 0] = cos_theta
    rotation_azimuth[2, 2] = cos_theta
    rotation_azimuth[0, 2] = -sin_theta
    rotation_azimuth[2, 0] = sin_theta
    rotation_azimuth[1, 1] = 1.0

    #phi rotation
    rotation_elevation = np.zeros((3, 3), dtype=np.float32)
    rotation_elevation[0, 0] = cos_phi
    rotation_elevation[0, 1] = sin_phi
    rotation_elevation[1, 0] = -sin_phi
    rotation_elevation[1, 1] = cos_phi
    rotation_elevation[2, 2] = 1.0

    #rotate phi, then theta
    rotation_matrix = np.matmul(rotation_azimuth, rotation_elevation)
    if invert_rot:
        rotation_matrix = np.linalg.inv(rotation_matrix)

    displacement = np.zeros((3, 1), dtype=np.float32)
    displacement[0, 0] = const.radius
    displacement = np.matmul(rotation_matrix, displacement)

    #assembling 4x4 from R + T
    extrinsic_matrix[0:3, 0:3] = rotation_matrix
    extrinsic_matrix[0:3, 3:4] = -displacement

    if invert_focal:
        intrinsic_matrix[2, 2] = float(const.focal_length)
        intrinsic_matrix[1, 1] = float(const.focal_length)
    else:
        intrinsic_matrix[2, 2] = 1.0 / float(const.focal_length)
        intrinsic_matrix[1, 1] = 1.0 / float(const.focal_length)

    camera_matrix = np.matmul(extrinsic_matrix, intrinsic_matrix)
    return camera_matrix


def project_and_postprocess(voxel):
    voxel_size = int(voxel.shape[1])
    return transformer_postprocess(project_voxel(voxel))
    

def rotate_and_project_voxel(voxel, rotmat):
    BS = getbs(voxel)
    S = getsize(voxel)
    
    r = tfutil.rank(voxel)
    assert r in [4,5]
    if r == 4:
        voxel = tf.expand_dims(voxel, axis = 4)

    voxel = transformer_preprocess(voxel)
        
    out = transformer(
        voxel,
        tf.reshape(rotmat, (BS, 16)),
        (S, S, S),
        3.0,
        5.0,
    )

    out = transformer_postprocess(out)
        
    if r == 4:
        out = tf.squeeze(out, axis = 4)

    return out


def translate_given_mat(vox, mat, debug=False):
    foo = rotate_voxel2(vox, mat, is_rot_inverse=True, debug=debug)
    if debug:
        import ipdb; ipdb.set_trace()
    #foo = rotate_voxel2(foo, rot_mat_2)
    return foo

def translate_given_angles2(dtheta, dphi, vox, debug=False):
    BS = dphi.get_shape()[0].value
    #rot_mat_1 = get_transform_matrix_easy_tf([0.0] * BS, -phi1)
    #rot_mat_2 = get_transform_matrix_easy_tf(dtheta, phi2)
    rot_mat_2 = get_transform_matrix_easy_tf(dtheta, dphi)
    if debug:
        import ipdb; ipdb.set_trace()
    foo = rotate_voxel2(vox, rot_mat_2)
    #foo = rotate_voxel2(foo, rot_mat_2)
    return foo

def get_transform_matrix_easy_tf(theta, phi):
    if isinstance(theta, list):
        theta = tf.stack(theta)
    if isinstance(phi, list):
        phi = tf.stack(phi)

    rotmat =  tf.map_fn(
        lambda t_p: get_transform_matrix_easy_tf_(t_p[0], t_p[1]),
        [theta, phi],
        parallel_iterations = 1000,
        dtype = tf.float32
    )
    return rotmat

def get_transform_matrix_easy_tf_(theta, phi):
    """
    Calculate rotation matrixes
    Assuming the thing is rotated using the center of the tensor,
    no translation
    """
    sin_phi = tf.sin(phi / 180 * np.pi)
    cos_phi = tf.cos(phi / 180 * np.pi)
    sin_theta = tf.sin(theta / 180.0 * np.pi) #why is theta negative???
    cos_theta = tf.cos(theta / 180.0 * np.pi)

    #these are inverted from normal!
    rotation_azimuth_flat = [
        cos_theta, 0.0, -sin_theta,
        0.0, 1.0, 0.0,
        sin_theta, 0.0, cos_theta
    ]

    rotation_elevation_flat = [
        cos_phi, sin_phi, 0.0,
        -sin_phi, cos_phi, 0.0,
        0.0, 0.0, 1.0
    ]
    f = lambda x: tf.reshape(tf.stack(x), (3, 3))
    rotation_azimuth = f(rotation_azimuth_flat)
    rotation_elevation = f(rotation_elevation_flat)
    rotation_matrix = tf.matmul(rotation_azimuth, rotation_elevation)
    displacement = np.zeros((3, 1), dtype=np.float32)
    displacement = tf.constant(displacement, dtype = np.float32)

    bottom_row = np.zeros((1, 4), dtype = np.float32)
    bottom_row[0,3] = 1.0
    bottom_row = tf.constant(bottom_row)
    extrinsic_matrix = tf.concat([
        tf.concat([rotation_matrix, -displacement], axis = 1),
        bottom_row
    ], axis = 0)
    return extrinsic_matrix


def compute_gt_flow_from_objects(boundary_to_center, object_state_t, object_state_t_1, mask_t, mask_t_1, voxel_size):
    """
    to input coordinate is defined as -boundary_to_center to boundary_to_center
    object_state: bs x nobjs x 13,  is the center of the object
    object_state_t_1: bs x nobjs x 13,  is the center of the object
    voxel: bs x nobjs x 128 x 128 x 128

    """
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

    with tf.variable_scope('rotate'):
        #
        grid = meshgrid_origin_centered_(voxel_size, voxel_size, voxel_size) * boundary_to_center
        # grid center at the location of the object
        bs, nobjs, _ = object_state_t.shape
        grid_center_at_object_t = tf.tile(tf.expand_dims(tf.expand_dims(grid, 0), 0), [bs, nobjs, 1, 1]) - tf.expand_dims(object_state_t[..., :3], -1)

        trans = tf.expand_dims(object_state_t_1[..., :3] - object_state_t[..., :3], -1)
        quat1 = tfq.Quaternion(object_state_t[...,3:7])
        quat2 = tfq.Quaternion(object_state_t_1[...,3:7])
        delta_quat = quat2 * quat1.inverse()
        rot_mat = delta_quat.as_rotation_matrix()
        # forward flow: need both
        flow = tf.matmul(rot_mat, grid_center_at_object_t) + trans - grid_center_at_object_t
        flow = tf.reshape(flow, [bs, nobjs, 3, voxel_size, voxel_size, voxel_size])
        flow = tf.transpose(flow, [0, 1, 3, 4, 5, 2])
        masked_flow = flow * mask_t

        # put mask on top
        grid_center_at_object_t_1 = tf.tile(tf.expand_dims(tf.expand_dims(grid, 0), 0), [bs, nobjs, 1, 1]) - tf.expand_dims(object_state_t_1[..., :3], -1)


        trans_inv = -trans
        delta_quat_inv = quat1 * quat2.inverse()
        rot_mat_inv = delta_quat_inv.as_rotation_matrix()
        flow_inv = tf.matmul(rot_mat_inv, grid_center_at_object_t_1) + trans_inv - grid_center_at_object_t_1
        flow_inv = tf.reshape(flow_inv, [bs, nobjs, 3, voxel_size, voxel_size, voxel_size])
        flow_inv = tf.transpose(flow_inv, [0, 1, 3, 4, 5, 2])
        masked_flow_inv = flow_inv * mask_t_1

        # inverse flow
        return masked_flow, masked_flow_inv


def rotate_voxel2(voxel, rotmat, out_size=None, is_rot_inverse=False, debug=False):
    """
    rotate voxel using its center.
    rotmat does not depend on focal length
    voxel: batch_size x s x s x s x c
    rotmat: batch_size x 16
    """
    def meshgrid_origin_centered(depth, height, width):
        with tf.variable_scope('_meshgrid'):
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

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            d_t_flat = tf.reshape(z_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat([d_t_flat, y_t_flat, x_t_flat, ones], 0)
            return grid

    with tf.variable_scope('rotate'):
        num_batch = voxel.get_shape().as_list()[0]
        num_channels = voxel.get_shape().as_list()[4]
        theta = tf.reshape(rotmat, (-1, 4, 4))
        theta_origin = theta
        theta = tf.cast(theta, 'float32')
        if not is_rot_inverse:
            #import tfquaternion as tfq
            #theta_quat = tfq.vector3d_to_quaternion(theta[:, :3, :3])

            #theta = tf.linalg.inv(theta)
            theta = tf.map_fn(lambda x: tf.matrix_inverse(x), theta[:, :3, :3])
            #theta = tf.matrix_inverse(theta[:, :3, :3])

            displacement = np.zeros((num_batch, 3, 1), dtype=np.float32)
            displacement = tf.constant(displacement, dtype = np.float32)
            bottom_row = np.zeros((num_batch, 1, 4), dtype = np.float32)
            bottom_row[:, 0,3] = 1.0
            bottom_row = tf.constant(bottom_row)

            theta = tf.concat([
                tf.concat([theta, -displacement], axis = 2),
                bottom_row
                ], axis = 1)

        if out_size:
            out_depth = out_size[0]
            out_height = out_size[1]
            out_width = out_size[2]
        else:
            out_depth = voxel.get_shape()[1].value
            out_height = voxel.get_shape()[2].value
            out_width = voxel.get_shape()[3].value
            out_size=[out_depth, out_height, out_width]
        grid = meshgrid_origin_centered(out_depth, out_height, out_width)

        grid_tmp = grid
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_batch]))
        grid = tf.reshape(grid, tf.stack([num_batch, 4, -1]))

        t_g = tf.matmul(theta, grid)
        z_s = t_g[:, 0, :]
        y_s = t_g[:, 1, :]
        x_s = t_g[:, 2, :]

        z_s_flat = tf.reshape(z_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])
        x_s_flat = tf.reshape(x_s, [-1])

        idxs, ws = interpolate_grid(voxel.get_shape()[:4], x_s_flat,
                                        y_s_flat, z_s_flat, out_size)

        output = interpolate_with_idx(voxel, idxs, ws)

        return output #, theta_origin, theta

def rotate_voxel(voxel, rotmat):
    """
    plan to deprecated. but this function is still used in many models.
    """
    BS = getbs(voxel)
    S = getsize(voxel)
    
    return transformer(
        voxel,
        tf.reshape(rotmat, (BS, 16)),
        (S, S, S),
        const.radius - const.boundary_to_center,#3.0,
        const.radius + const.boundary_to_center,#5.0,
        do_project = False,
    )


def project_voxel(voxel, cam_int=None, debug=False):
    BS = getbs(voxel)
    S = getsize(voxel)
    if cam_int is None:
        rotmat = tf.constant(get_transform_matrix(0.0, 0.0), dtype = tf.float32)
        rotmat = tf.reshape(rotmat, (1, 4, 4))
        rotmat = tf.tile(rotmat, (BS, 1, 1))
    else:
        rotmat = cam_int
    if debug:
        import ipdb; ipdb.set_trace()
    return transformer(
        voxel,
        tf.reshape(rotmat, (BS, 16)),
        (S, S, S),
        const.radius - const.boundary_to_center,#3.0,
        const.radius + const.boundary_to_center,#5.0,
        do_project = True,
    )

def unproject_voxel(voxel, cam_int=None, debug=False):
    BS = getbs(voxel)
    size = int(voxel.shape[1])
    if cam_int is None:
        rotmat = tf.constant(get_transform_matrix(0.0, 0.0, invert_focal = True, debug=debug), dtype = tf.float32)
        rotmat = tf.reshape(rotmat, (1, 4, 4))
        rotmat = tf.tile(rotmat, (BS, 1, 1))
    else:
        rotmat = cam_int
    voxel =  transformer(
        voxel,
        tf.reshape(rotmat, (BS, 16)),
        (size, size, size),
        const.radius - const.boundary_to_center,#3.0,
        const.radius + const.boundary_to_center,#5.0,
        do_project = 'invert',
    )

    voxel = tf.reverse(voxel, axis=[2, 3])
    return voxel

def unproject_image(img_, cam_int=None, debug=False):
    #step 1: convert to voxel
    #step 2: call unproject_voxel

    size = int(img_.shape[1])

    def unflatten(img):
        voxel = tf.expand_dims(img, 1)
        #now BS x 1 x H x W x C
        voxel = tf.tile(voxel, (1, size, 1, 1, 1))
        #BS x S x S x S x C hopefully
        return voxel

    projected_voxel = unflatten(img_)
    return unproject_voxel(projected_voxel, cam_int=cam_int, debug=debug)


def transformer_preprocess(voxel):
    return tf.reverse(voxel, axis=[1]) #z axis

def transformer_postprocess(voxel):
    #so, since the bilinear sampling screws up the precision, this adjustment is necessary:
    delta = 1E-4
    voxel = voxel * (1.0-delta) + delta/2.0
    voxel = tf.reverse(voxel, axis=[2, 3])
    #zxy -> xyz i think
    voxel = tf.transpose(voxel, (0, 2, 3, 1, 4))
    return voxel

def voxel2mask_aligned(voxel):
    return tf.reduce_max(voxel, axis=3)


def voxel2depth_aligned(voxel):
    BS = getbs(voxel)
    S = getsize(voxel)
    
    voxel = tf.squeeze(voxel, axis=4)

    costgrid = tf.cast(tf.tile(
        tf.reshape(tf.range(0, S), (1, 1, 1, S)),
        (BS, S, S, 1)
    ), tf.float32)

    invalid = 1000 * tf.cast(voxel < 0.5, dtype=tf.float32)
    invalid_mask = tf.tile(tf.reshape(tf.constant([1.0] * (S - 1) + [0.0], tf.float32),
                                      (1, 1, 1, S)),
                           (BS, S, S, 1))

    costgrid = costgrid + invalid * invalid_mask

    depth = tf.expand_dims(tf.argmin(costgrid, axis=3), axis=3)

    #convert back to (3,5)
    depth = tf.cast(depth, tf.float32)

    #apparently don't do this...
    #depth += 0.5 #0.5 to S-0.5
    
    depth /= S #almost 0.0 to 1.0
    depth *= 2 #almost 0.0 to 2.0
    depth += const.radius-1 #about 3.0 to 5.0

    return depth

def features_from_projected(voxel1, voxel2, factor):
    raise Exception('this function is out of date')
    #voxel 1 is used to compute occupancy
    #voxel 2 is where the features come from
    #factor is the downsampling factor bt voxel2 and voxel1
    
    voxel1 = tf.squeeze(voxel1, axis=4)

    costgrid = tf.cast(tf.tile(
        tf.reshape(tf.range(0, S), (1, 1, 1, const.S)),
        (const.BS, const.S, const.S, 1)
    ), tf.float32)

    invalid = 1000 * tf.cast(voxel1 < 0.5, dtype=tf.float32)
    invalid_mask = tf.tile(tf.reshape(tf.constant([1.0] * (const.S - 1) + [0.0], tf.float32),
                                      (1, 1, 1, const.S)),
                           (const.BS, const.S, const.S, 1))

    costgrid = costgrid + invalid * invalid_mask

    ###############
    #using Q for the size of voxel2
    Q = const.S / factor

    depth = tf.expand_dims(tf.argmin(costgrid, axis=3), axis = 3) #B x S x S x 1
    depth = tf.squeeze(tf.image.resize_images(depth, (Q, Q)), axis = 3) #B x Q x Q
    depth = tf.cast(tf.round(depth), dtype = tf.int32)

    depth = tf.one_hot(depth, depth = Q, dtype = tf.float32) #B x Q x Q x Q

    #we want to conv on last axis to spread things out a bit

    #first move some stuff to the batch axis to make things easier
    depth = tf.reshape(depth, (const.BS * Q * Q, Q, 1)) #BQQ x Q x 1

    filt = tf.constant([0.1, 0.2, 0.4, 0.2, 0.1], dtype = tf.float32)
    filt = tf.reshape(filt, (5, 1, 1))

    #BQQ x Q x 1
    depth = tf.nn.conv1d(depth, filt, stride = 1, padding = 'SAME', use_cudnn_on_gpu = True) 

    depth = tf.reshape(depth, (const.BS, Q, Q, Q, 1))
    
    depth_gated_feats = tf.reduce_sum(voxel2 * depth, axis = 3)

    return depth_gated_feats

def loss_against_constraint_l2(constraint, voxels):
    return tf.reduce_mean(constraint * voxels)    

def loss_against_constraint_ce(constraint, voxels):
    #we want voxels to be 1, so that the log tends to 0
    #0 so that smaller voxel -> bigger error
    return tf.reduce_mean(-tf.log(voxels + const.eps) * constraint)

def translate_given_angles(dtheta, phi1, phi2, vox):
    """
    plan to deprecated: please use translate_given_angles2
    dtheta, phi1, phi2: [batch_size]
    vox: batch_size x w x w x w x c
    """

    # dtheta: [batch_size]
    #first kill elevation
    BS = phi1.get_shape()[0].value
    rot_mat_1 = get_transform_matrix_easy_tf([0.0] * BS, -phi1)
    rot_mat_2 = get_transform_matrix_easy_tf(dtheta, phi2)
    rot_mat = tf.linalg.matmul(rot_mat_1, rot_mat_2)

    #remember to postprocess after this?
    foo = rotate_voxel2(vox, rot_mat, is_rot_inverse=True)
    return foo

def translate_given_angles_deprecated(dtheta, phi1, phi2, vox):
    """
    plan to deprecated: please use translate_given_angles2
    dtheta, phi1, phi2: [batch_size]
    vox: batch_size x w x w x w x c
    """

    # dtheta: [batch_size]
    #first kill elevation
    BS = phi1.get_shape()[0].value
    rot_mat_1 = get_transform_matrix_tf([0.0] * BS, -phi1)
    rot_mat_2 = get_transform_matrix_tf(dtheta, phi2)

    #remember to postprocess after this?
    foo = rotate_voxel(vox, rot_mat_1)
    foo = rotate_voxel(foo, rot_mat_2)
    return foo
