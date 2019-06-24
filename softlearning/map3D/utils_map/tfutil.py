import tensorflow as tf
import constants as const


def box_refinement_graph(positive_rois, roi_gt_boxes):
    gt_center = tf.reduce_mean(roi_gt_boxes, 2)
    pd_center = tf.reduce_mean(positive_rois, 2)
    delta_zyx = gt_center-pd_center

    len_gt = roi_gt_boxes[:,:,1] - roi_gt_boxes[:,:,0]
    len_pd = positive_rois[:,:,1] - positive_rois[:,:,0]
    delta_len = len_gt - len_pd
    return tf.concat([delta_zyx, delta_len], 1)

def image_crop_and_resize(image,boxes,box_ind, crop_size):
    """
    image: batch_size x image_height x image_width x channel
    boxes: num_boxes x 4, float32, [0-1]
    box_ind: num_boxes, int32, [y1, x1, y2, x2]
    crop_size: crop_height x crop_width, int32
    """
    # create grid
    y_axis = tf.tile(tf.reshape(tf.range(tf.cast(crop_size[0], dtype=tf.float32), dtype=tf.float32), [-1, 1]), [1, crop_size[1]])
    x_axis = tf.tile(tf.reshape(tf.range(tf.cast(crop_size[1], dtype=tf.float32), dtype=tf.float32), [1, -1]), [crop_size[0], 1])
    grid_center = 0.5 * tf.cast(crop_size, dtype=tf.float32)

    grid = tf.stack([y_axis, x_axis], 2) + 0.5 - tf.reshape(grid_center, [1,1,2])
    # output_height x output_width x 2
    grid = grid / tf.reshape(tf.cast(crop_size, dtype=tf.float32), [1,1,2])
    # calculate box center and width and height on the images
    image_height = tf.shape(image)[1]
    image_width = tf.shape(image)[2]
    image_c = tf.shape(image)[3]
    image_height_f = tf.cast(image_height, dtype=tf.float32)
    image_width_f = tf.cast(image_width, dtype=tf.float32)

    box_center_y = 0.5 * (boxes[:,0] + boxes[:,2]) * image_height_f
    box_center_x = 0.5 * (boxes[:,1] + boxes[:,3]) * image_width_f
    box_height = (boxes[:,2] - boxes[:,0]) * image_height_f
    box_width = (boxes[:,3] - boxes[:,1]) * image_width_f
    #nbox x 1 x 1 x 2
    box_scale = tf.expand_dims(tf.expand_dims(tf.stack([box_height, box_width], 1), 1), 2)
    box_trans = tf.expand_dims(tf.expand_dims(tf.stack([box_center_y, box_center_x], 1), 1), 2)
    #nbox x output_height x output_width x 2
    index = tf.expand_dims(grid, 0) * box_scale + box_trans

    y_f = index[:, :, :,0] - 0.5
    x_f = index[:, :, :,1] - 0.5

    y1_int = tf.clip_by_value(tf.cast(y_f, dtype=tf.int32), 0, image_height - 1)
    y2_int = tf.clip_by_value(y1_int + 1, 0, image_height - 1)
    x1_int = tf.clip_by_value(tf.cast(x_f, dtype=tf.int32), 0, image_width - 1)
    x2_int = tf.clip_by_value(x1_int + 1, 0, image_width - 1)

    image_nd = tf.gather_nd(image, tf.expand_dims(box_ind, 1))
    # num_box x output_h x output_w x image_h x image_w x c
    # using tile is creating image that is too big -> process image one by one
    f = lambda x: image_sample_four_corners(*x)
    pixel_y1x1, pixel_y1x2, pixel_y2x1, pixel_y2x2 = tf.map_fn(f, [image_nd, y1_int, y2_int, x1_int, x2_int], dtype=(tf.float32, tf.float32, tf.float32, tf.float32))

    y1_int_f = tf.cast(y1_int, dtype=tf.float32)
    y2_int_f = tf.cast(y2_int, dtype=tf.float32)
    x1_int_f = tf.cast(x1_int, dtype=tf.float32)
    x2_int_f = tf.cast(x2_int, dtype=tf.float32)
    w_y1x1 = (y2_int_f - y_f) * (x2_int_f - x_f)
    w_y1x2 = (y2_int_f - y_f) * (x_f - x1_int_f)
    w_y2x1 = (y_f - y1_int_f) * (x2_int_f - x_f)
    w_y2x2 = (y_f - y1_int_f) * (x_f - x1_int_f)

    pixel = tf.expand_dims(w_y1x1, 3) * pixel_y1x1 +\
            tf.expand_dims(w_y1x2, 3) * pixel_y1x2 +\
            tf.expand_dims(w_y2x1, 3) * pixel_y2x1 +\
            tf.expand_dims(w_y2x2, 3) * pixel_y2x2
    final_image = tf.reshape(pixel, [-1, crop_size[0], crop_size[1], image_c])
    return final_image

def image_sample_four_corners(image, y1_int, y2_int, x1_int, x2_int):
    height = tf.shape(y1_int)[0]
    width = tf.shape(y1_int)[1]
    channel = tf.shape(image)[2]
    y1_int = tf.reshape(y1_int, [-1])
    y2_int = tf.reshape(y2_int, [-1])
    x1_int = tf.reshape(x1_int, [-1])
    x2_int = tf.reshape(x2_int, [-1])

    pixel_y1x1 = tf.reshape(tf.gather_nd(image, tf.stack([y1_int, x1_int], 1)), [height, width, channel])
    pixel_y1x2 = tf.reshape(tf.gather_nd(image, tf.stack([y1_int, x2_int], 1)), [height, width, channel])
    pixel_y2x1 = tf.reshape(tf.gather_nd(image, tf.stack([y2_int, x1_int], 1)), [height, width, channel])
    pixel_y2x2 = tf.reshape(tf.gather_nd(image, tf.stack([y2_int, x2_int], 1)), [height, width, channel])

    return pixel_y1x1, pixel_y1x2, pixel_y2x1, pixel_y2x2
    """

    image_nd = tf.tile(tf.expand_dims(tf.expand_dims(image_nd, 1), 2), [1, crop_size[0], crop_size[1], 1, 1, 1])
    image_nd = tf.reshape(image_nd, [-1, image_height, image_width, image_c])
    image_id = tf.range(tf.shape(image_nd)[0])
    pixel_y1x1 = tf.gather_nd(image_nd, tf.stack([image_id, y1_int, x1_int], 1))
    pixel_y1x2 = tf.gather_nd(image_nd, tf.stack([image_id, y1_int, x2_int], 1))
    pixel_y2x1 = tf.gather_nd(image_nd, tf.stack([image_id, y2_int, x1_int], 1))
    pixel_y2x2 = tf.gather_nd(image_nd, tf.stack([image_id, y2_int, x2_int], 1))

    """
def tensor3d_crop_and_resize(image, boxes,box_ind, crop_size):
    """
    image: batch_size x image_depth x image_height x image_width x channel
    boxes: num_boxes x 6, float32, [0-1], [z1, y1, x1, z2, y2, x2]
    box_ind: num_boxes, int32
    crop_size: crop_depth x crop_height x crop_width, int32
    """
    # create grid
    z_axis = tf.tile(tf.reshape(
                 tf.range(tf.cast(crop_size[0], dtype=tf.float32), dtype=tf.float32)
                 , [-1, 1, 1]), [1, crop_size[1], crop_size[2]])
    y_axis = tf.tile(tf.reshape(
                 tf.range(tf.cast(crop_size[1], dtype=tf.float32), dtype=tf.float32)
                 , [1, -1, 1]), [crop_size[0], 1, crop_size[2]])
    x_axis = tf.tile(tf.reshape(
                 tf.range(tf.cast(crop_size[2], dtype=tf.float32), dtype=tf.float32)
                 , [1, 1, -1]), [crop_size[0], crop_size[1], 1])
    grid_center = 0.5 * tf.cast(crop_size, dtype=tf.float32)

    grid = tf.stack([z_axis, y_axis, x_axis], 3) + 0.5 - tf.reshape(grid_center, [1,1,1,3])


    # output_depth x output_height x output_width x 2
    grid = grid / tf.reshape(tf.cast(crop_size, dtype=tf.float32), [1,1,1,3])
    # calculate box center and width and height on the images
    image_depth = tf.shape(image)[1]
    image_height = tf.shape(image)[2]
    image_width = tf.shape(image)[3]
    image_c = tf.shape(image)[4]

    image_depth_f = tf.cast(image_depth, dtype=tf.float32)
    image_height_f = tf.cast(image_height, dtype=tf.float32)
    image_width_f = tf.cast(image_width, dtype=tf.float32)

    box_center_z = 0.5 * (boxes[:,0] + boxes[:,3]) * image_depth_f
    box_center_y = 0.5 * (boxes[:,1] + boxes[:,4]) * image_height_f
    box_center_x = 0.5 * (boxes[:,2] + boxes[:,5]) * image_width_f

    box_depth = (boxes[:,3] - boxes[:,0]) * image_depth_f
    box_height = (boxes[:,4] - boxes[:,1]) * image_height_f
    box_width = (boxes[:,5] - boxes[:,2]) * image_width_f
    #nbox x 1 x 1 x1 x 3
    box_scale = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.stack([box_depth, box_height, box_width], 1), 1), 2), 3)
    box_trans = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.stack([box_center_z, box_center_y, box_center_x], 1), 1), 2), 3)
    #nbox x output_depth x output_height x output_width x 3
    index = tf.expand_dims(grid, 0) * box_scale + box_trans

    z_f = index[:, :, :, :,0] - 0.5
    y_f = index[:, :, :, :,1] - 0.5
    x_f = index[:, :, :, :,2] - 0.5

    #nbox x output_depth x output_height x output_width
    z1_int = tf.clip_by_value(tf.cast(z_f, dtype=tf.int32), 0, image_depth - 1)
    z2_int = tf.clip_by_value(z1_int + 1, 0, image_depth - 1)
    y1_int = tf.clip_by_value(tf.cast(y_f, dtype=tf.int32), 0, image_height - 1)
    y2_int = tf.clip_by_value(y1_int + 1, 0, image_height - 1)
    x1_int = tf.clip_by_value(tf.cast(x_f, dtype=tf.int32), 0, image_width - 1)
    x2_int = tf.clip_by_value(x1_int + 1, 0, image_width - 1)

    image_nd = tf.gather_nd(image, tf.expand_dims(box_ind, 1))
    # using tile is creating image that is too big -> process image one by one

    nbox = tf.shape(z1_int)[0]
    depth = tf.shape(z1_int)[1]
    height = tf.shape(z1_int)[2]
    width = tf.shape(z1_int)[3]

    query_idx = tf.tile(tf.reshape(tf.range(nbox), [-1, 1, 1, 1]), [1, depth, height, width])
    query_idx = tf.reshape(query_idx, [-1])
    z1_int_flat = tf.reshape(z1_int, [-1])
    z2_int_flat = tf.reshape(z2_int, [-1])
    y1_int_flat = tf.reshape(y1_int, [-1])
    y2_int_flat = tf.reshape(y2_int, [-1])
    x1_int_flat = tf.reshape(x1_int, [-1])
    x2_int_flat = tf.reshape(x2_int, [-1])
    new_size = [nbox, depth, height, width, image_c]
    pixel_z1y1x1 = tf.reshape(tf.gather_nd(image_nd,\
        tf.stack([query_idx, z1_int_flat, y1_int_flat, x1_int_flat], 1)), new_size)
    pixel_z1y1x2 = tf.reshape(tf.gather_nd(image_nd,\
        tf.stack([query_idx, z1_int_flat, y1_int_flat, x2_int_flat], 1)), new_size)
    pixel_z1y2x1 = tf.reshape(tf.gather_nd(image_nd,\
        tf.stack([query_idx, z1_int_flat, y2_int_flat, x1_int_flat], 1)), new_size)
    pixel_z1y2x2 = tf.reshape(tf.gather_nd(image_nd,\
        tf.stack([query_idx, z1_int_flat, y2_int_flat, x2_int_flat], 1)), new_size)
    pixel_z2y1x1 = tf.reshape(tf.gather_nd(image_nd,\
        tf.stack([query_idx, z2_int_flat, y1_int_flat, x1_int_flat], 1)), new_size)
    pixel_z2y1x2 = tf.reshape(tf.gather_nd(image_nd,\
        tf.stack([query_idx, z2_int_flat, y1_int_flat, x2_int_flat], 1)), new_size)
    pixel_z2y2x1 = tf.reshape(tf.gather_nd(image_nd,\
        tf.stack([query_idx, z2_int_flat, y2_int_flat, x1_int_flat], 1)), new_size)
    pixel_z2y2x2 = tf.reshape(tf.gather_nd(image_nd,\
        tf.stack([query_idx, z2_int_flat, y2_int_flat, x2_int_flat], 1)), new_size)


    z1_int_f = tf.cast(z1_int, dtype=tf.float32)
    z2_int_f = tf.cast(z2_int, dtype=tf.float32)
    y1_int_f = tf.cast(y1_int, dtype=tf.float32)
    y2_int_f = tf.cast(y2_int, dtype=tf.float32)
    x1_int_f = tf.cast(x1_int, dtype=tf.float32)
    x2_int_f = tf.cast(x2_int, dtype=tf.float32)


    w_z1y1x1 = (z2_int_f - z_f) * (y2_int_f - y_f) * (x2_int_f - x_f)
    w_z1y1x2 = (z2_int_f - z_f) * (y2_int_f - y_f) * (x_f - x1_int_f)
    w_z1y2x1 = (z2_int_f - z_f) * (y_f - y1_int_f) * (x2_int_f - x_f)
    w_z1y2x2 = (z2_int_f - z_f) * (y_f - y1_int_f) * (x_f - x1_int_f)
    w_z2y1x1 = (z_f - z1_int_f) * (y2_int_f - y_f) * (x2_int_f - x_f)
    w_z2y1x2 = (z_f - z1_int_f) * (y2_int_f - y_f) * (x_f - x1_int_f)
    w_z2y2x1 = (z_f - z1_int_f) * (y_f - y1_int_f) * (x2_int_f - x_f)
    w_z2y2x2 = (z_f - z1_int_f) * (y_f - y1_int_f) * (x_f - x1_int_f)

    pixel = tf.expand_dims(w_z1y1x1, 4) * pixel_z1y1x1 +\
            tf.expand_dims(w_z1y1x2, 4) * pixel_z1y1x2 +\
            tf.expand_dims(w_z1y2x1, 4) * pixel_z1y2x1 +\
            tf.expand_dims(w_z1y2x2, 4) * pixel_z1y2x2 +\
            tf.expand_dims(w_z2y1x1, 4) * pixel_z2y1x1 +\
            tf.expand_dims(w_z2y1x2, 4) * pixel_z2y1x2 +\
            tf.expand_dims(w_z2y2x1, 4) * pixel_z2y2x1 +\
            tf.expand_dims(w_z2y2x2, 4) * pixel_z2y2x2
    final_image = tf.reshape(pixel, [-1, crop_size[0], crop_size[1], crop_size[2], image_c])
    return final_image


#def apply_refinement():
def overlap_mask_graph(mask1, mask2):
    #mask: batch x s x s x s x 1
    bs1 = tf.shape(mask1)[0]
    bs2 = tf.shape(mask2)[0]
    mask1 = tf.reshape(mask1, [bs1, -1])
    mask2 = tf.reshape(mask2, [bs2, -1])

    intersection = tf.matmul(mask1, tf.transpose(mask2, [1, 0]))
    union = tf.reshape(tf.reduce_sum(mask1,1), [-1, 1]) + tf.reshape(tf.reduce_sum(mask2, 1), [1,-1])\
        -intersection

    iou = intersection / union
    return iou

def overlap_graph(boxes1, boxes2):
    # boxes1: batch x 3 x 2(z1,z2,y1,y2,x1,x2)
    b1_bs = tf.shape(boxes1)[0]
    b2_bs = tf.shape(boxes2)[0]

    boxes1 = tf.reshape(boxes1, [-1, 6])
    boxes2 = tf.reshape(boxes2, [-1, 6])
    b1 = tf.reshape(tf.tile(boxes1, [1, b2_bs]), [-1, 6])
    b2 = tf.tile(boxes2, [b1_bs, 1])

    b1_z1, b1_z2, b1_y1, b1_y2, b1_x1, b1_x2 = tf.split(b1, 6, axis=1)
    b2_z1, b2_z2, b2_y1, b2_y2, b2_x1, b2_x2 = tf.split(b2, 6, axis=1)
    z1 = tf.maximum(b1_z1, b2_z1)
    z2 = tf.minimum(b1_z2, b2_z2)
    y1 = tf.maximum(b1_y1, b2_y1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x1 = tf.maximum(b1_x1, b2_x1)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(z2 - z1, 0) * tf.maximum(y2 - y1, 0) * tf.maximum(x2 - x1, 0)
    b1_area = (b1_z2 - b1_z1) * (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_z2 - b2_z1) * (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    iou = intersection / union
    overlaps = tf.reshape(iou, [b1_bs, b2_bs])

    return overlaps

def anchors_to_bboxes(anchors, indices, scale):
    grid_center = (tf.cast(indices, dtype=tf.float32) + 0.5)* (1/scale)
    object_center = grid_center + anchors[:,:3] * const.anchor_size
    object_min = tf.clip_by_value(object_center - 0.5 * tf.clip_by_value(tf.exp(anchors[:,3:]) * const.anchor_size, 0, 1), 0, 1)
    object_max = tf.clip_by_value(object_center + 0.5 * tf.clip_by_value(tf.exp(anchors[:,3:]) * const.anchor_size, 0, 1), 0, 1)
    return tf.stack([object_min, object_max], 2)

def draw_bbox(anchor, center_idx, image_front_single, image_top_single):
    # anchor: bs x s x s x s x 6, anchor values for each little grids
    # center_idx: bs x nobj x 4 (batch_id, z, y, x)
    # images to draw boxes on: bs x s x s x 3
    bs, s, s, s, _ = anchor.get_shape()
    bs, nobj, _ = center_idx.get_shape()

    batch_id = tf.tile(tf.reshape(tf.range(bs.value), [1, -1, 1]),
        [nobj.value, 1, 1])
    center_idx_tmp = (tf.cast(center_idx, dtype=tf.float32)+ 0.5) * (1/s.value)
    center_idx = tf.concat([batch_id, tf.transpose(center_idx, [1, 0, 2])], 2)

    selected_anchors = tf.transpose(tf.gather_nd(anchor, center_idx), [1, 0, 2])
    zyxmin = 1 - (center_idx_tmp + selected_anchors[:,:,:3] * const.anchor_size\
             -  tf.exp(selected_anchors[:,:,3:]) * const.anchor_size * 0.5)
    zyxmax = 1 - (center_idx_tmp + selected_anchors[:,:,:3] * const.anchor_size\
             + tf.exp(selected_anchors[:,:,3:]) * const.anchor_size * 0.5)

    bbox_image_front_gt = tf.image.draw_bounding_boxes(
        image_front_single,
        tf.concat([zyxmax[:,:,1:],zyxmin[:,:,1:]], 2)
    )
    bbox_image_top_gt = tf.image.draw_bounding_boxes(
        image_top_single,
        tf.concat([zyxmax[:,:,0:1], zyxmax[:,:,2:3],zyxmin[:,:,0:1],
                   zyxmin[:,:,2:3]], 2)
    )
    return bbox_image_front_gt, bbox_image_top_gt


def poolorunpool(input_, targetsize):
    inputsize = input_.shape.as_list()[1]
    if inputsize == targetsize:
        return input_
    elif inputsize > targetsize:
        ratio = inputsize // targetsize
        return tf.nn.pool(
            input_,
            window_shape = [ratio, ratio],
            padding = 'SAME',
            pooling_type = 'AVG',
            strides = [ratio, ratio]
        )
    else: #inputsize < targetsize:
        ratio = targetsize // inputsize
        return tf.image.resize_nearest_neighbor(
            input_,
            tf.stack([inputsize * ratio]*2)
        )


def current_scope():
    return tf.get_variable_scope().name


def current_scope_and_vars():
    scope = current_scope()

    collections = [tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES]
    #DONT USE the below, because it leaves out the moving avg variables
    #collections = [tf.GraphKeys.TRAINABLE_VARIABLES]

    vars_ = []
    for collection in collections:
        vars_.extend(tf.get_collection(collection, scope))

    vars_ = list(set(vars_))

    #for z in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope):
    #    assert z in vars_
    return (scope, vars_)


def add_scope_to_dct(dct, name):
    dct[name] = current_scope_and_vars()


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64s_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def leaky_relu(alpha):
    return lambda x: tf.nn.relu(x) + tf.nn.relu(-x) * alpha


def upscale(feats):
    _shape = feats.get_shape()
    h = _shape[1]
    w = _shape[2]
    return tf.image.resize_nearest_neighbor(feats, tf.stack([h * 2, w * 2]))

def mv_unstack(x):
    return tf.reshape(x, [const.BS, const.H, const.W, const.V, -1])


def mv_stack(x):
    return tf.reshape(x, [const.BS, const.H, const.W, -1])


def mv_shape4(chans=1):
    return (const.BS, const.H, const.W, const.V * chans)


def mv_shape5(chans=1):
    return (const.BS, const.H, const.W, const.V, chans)


def rand_elem(tensor, axis):
    n = int(tensor.get_shape()[axis])
    r = tf.rank(tensor)
    idx = tf.multinomial([[1.0] * n], 1)[0, 0]
    idx_ = [0] * axis + idx + [0] * r - axis - 1
    size = [-1] * axis + 1 + [-1] * r - axis - 1
    return tf.slice(tensor, tf.stack(idx_), size)


def tf_random_bg(N, darker = False):
    color = tf.random_uniform((3,))
    if darker:
        color /= 2.0
    color = tf.tile(tf.reshape(color, (1, 1, 1, 3)), (N, const.Hdata, const.Wdata, 1))
    return color

def add_feat_to_img(img, feat):
    # img is BS x H x W x C
    # feat is BS x D
    # output is BS x H x W x (C+D)

    bs, h, w, _ = list(map(int, img.get_shape()))
    feat = tf.reshape(feat, (bs, 1, 1, -1))
    tilefeat = tf.tile(feat, (1, h, w, 1))
    return tf.concat([img, tilefeat], axis=3)


def cycle(tensor, idx, axis):
    r = len(tensor.get_shape())
    n = tensor.get_shape()[axis]
    # move idx elements from the front to the back on axis
    start_idx = [0] * r

    head_size = [-1]*r
    head_size[axis] = idx
    head_size = tf.stack(head_size)

    mid_idx = [0]*r
    mid_idx[axis] = idx
    mid_idx = tf.stack(mid_idx)

    tail_size = [-1] * r

    head = tf.slice(tensor, start_idx, head_size)
    tail = tf.slice(tensor, mid_idx, tail_size)
    return tf.concat([tail, head], axis=axis)


def meshgrid2D(bs, height, width):
    with tf.variable_scope("meshgrid2D"):
        grid_x = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(tf.expand_dims(tf.linspace(0.0, width - 1, width), 1), [1, 0])
        )

        grid_y = tf.matmul(tf.expand_dims(tf.linspace(0.0, height - 1, height), 1),
                           tf.ones(shape=tf.stack([1, width])))
        grid_x = tf.tile(tf.expand_dims(grid_x, 0), [bs, 1, 1], name="grid_x")
        grid_y = tf.tile(tf.expand_dims(grid_y, 0), [bs, 1, 1], name="grid_y")
        return grid_x, grid_y


def batch(tensor):
    return tf.expand_dims(tensor, axis=0)


def interleave(x1, x2, axis):
    x1s = tf.unstack(x1, axis=axis)
    x2s = tf.unstack(x2, axis=axis)
    outstack = []
    for (x1_, x2_) in zip(x1s, x2s):
        outstack.append(x1_)
        outstack.append(x2_)
    return tf.stack(outstack, axis=axis)


def bin_indices(Z, num_bins):
    bin_size = 100.0 / num_bins
    percentiles = tf.stack([tf.contrib.distributions.percentile(Z, bin_size * p)
                            for p in range(num_bins)])
    Z_ = tf.expand_dims(Z, 1)
    in_bin = tf.cast(Z_ > percentiles, tf.float32)
    in_bin *= tf.constant(list(range(1, num_bins + 1)), dtype=tf.float32)
    bin_idx = tf.cast(tf.argmax(in_bin, axis=1), tf.int32)
    return bin_idx


def prob_round(z):
    ''' probabilistically rounds z to floor(z) or ceil(z) '''

    zf = tf.floor(z)
    p = z - zf
    zf = tf.cast(zf, tf.int32)
    zc = zf + 1
    #if p ~= 0, then condition ~= 0 -> zf selected
    #if p ~= 1, then condition ~= 1 -> zc selected
    return tf.where(tf.random_uniform(tf.shape(p)) < p, zc, zf)


def select_p(tensor, p):
    ''' select entries of tensor with probability p'''
    d1 = tf.expand_dims(tf.shape(tensor)[0], axis=0)  # a weird approach
    keep = tf.random_uniform(d1) < p
    return tf.boolean_mask(tensor, keep), keep


def extract_axis3(tensor, index):
    tensor_t = tf.transpose(mv_unstack(tensor), (3, 0, 1, 2, 4))
    base = tf.gather(tensor_t, index)
    base = tf.squeeze(base, axis=1)  # 0 is batch axis
    return base


def rank(tensor):
    return len(tensor.get_shape())

def variable_in_shape(shape, name = 'variable'):
    return tf.get_variable(name, shape)

def norm01(t):
    t -= tf.reduce_min(t)
    t /= tf.reduce_max(t)
    return t

def round2int(t):
    return tf.cast(tf.round(t), tf.int32)

def randidx(N):
    return tf.cast(tf.multinomial([[1.0] * N], 1)[0, 0], tf.int32)

def match_placeholders_to_inputs(phs, inps):

    listortuple = lambda x: isinstance(x, list) or isinstance(x, tuple)
    
    if isinstance(phs, dict) and isinstance(inps, dict):
        rval = {}
        for name in phs:
            rval.update(match_placeholders_to_inputs(phs[name], inps[name]))
        return rval
    elif listortuple(phs) and listortuple(inps):
        rval = {}
        for (ph, inp) in zip(phs, inps):
            rval.update(match_placeholders_to_inputs(ph, inp))
        return rval
    elif 'tensorflow' in phs.__class__.__module__ and 'numpy' in inps.__class__.__module__:
        return {phs:inps}
    else:
        raise Exception('unsupported type...')

def pool3d(x, factor = 2, rank4 = False):
    if rank4:
        assert rank(x) == 4
        x = tf.expand_dims(x, axis = 4)
    return tf.nn.max_pool3d(
        x,
        ksize = [1, factor, factor, factor, 1],
        strides = [1, factor, factor, factor, 1],
        padding = 'VALID'
    )

def unitize(tensor, axis = -1):
    norm = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis = axis, keep_dims = True) + const.eps)
    return tensor / norm

def set_batch_size(tensor, bs = const.BS):
    sizelist = list(tensor.shape)
    sizelist[0] = bs
    tensor.set_shape(sizelist)

def make_opt_op(optimizer, fn):
    if const.eager:
        return optimizer.minimize(fn)
    else:
        if const.summ_grads:
            x = fn()
            grads = tf.gradients(x, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = optimizer.minimize(x)
            return train_step
        else:
            x = fn()

            train_list = tf.contrib.framework.filter_variables(
                tf.trainable_variables())

            grads = tf.gradients(x, train_list)
            grads = list(zip(grads, tf.trainable_variables()))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            global_step = tf.Variable(0, trainable=False, name='global_step')
            with tf.control_dependencies(update_ops):
                train_step =  optimizer.apply_gradients(grads_and_vars=grads, global_step=global_step)
            return train_step
    
def read_eager_tensor(x):
    if x is not None:
        return x.numpy()
    else:
        return None

def concat_apply_split(inputs, fn):
    num_split = len(inputs)

    inputs = tf.concat(inputs, axis = 0)
    outputs = fn(inputs)

    if isinstance(outputs, list):
        return [tf.split(output, num_split, axis = 0) for output in outputs]
    else:
        return tf.split(outputs, num_split, axis = 0)

def tanh01(x):
    return (tf.nn.tanh(x)+1)/2
