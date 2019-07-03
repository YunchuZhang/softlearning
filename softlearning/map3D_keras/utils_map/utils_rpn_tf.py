import tensorflow as tf
import numpy as np
from easydict import EasyDict as edict
import pyximport
pyximport.install()

#from box_overlaps import *
#from utils_basic import *
#import utils_rpn
#import hyperparams as hyp
import sys

"""
B = hyp.B
H = hyp.H
W = hyp.W
N = hyp.N

__C = edict()
cfg = __C

__C.Y_MIN = -40
__C.Y_MAX = 40
__C.X_MIN = 0
__C.X_MAX = 70.4
__C.VOXEL_X_SIZE = 0.2
__C.VOXEL_Y_SIZE = 0.2
__C.VOXEL_POINT_COUNT = 35
__C.INPUT_WIDTH = int((__C.X_MAX - __C.X_MIN) / __C.VOXEL_X_SIZE)
__C.INPUT_HEIGHT = int((__C.Y_MAX - __C.Y_MIN) / __C.VOXEL_Y_SIZE)
__C.FEATURE_RATIO = 2
__C.FEATURE_WIDTH = int(__C.INPUT_WIDTH / __C.FEATURE_RATIO)
__C.FEATURE_HEIGHT = int(__C.INPUT_HEIGHT / __C.FEATURE_RATIO)

# car anchor
__C.ANCHOR_L = 3.9
__C.ANCHOR_W = 1.6
__C.ANCHOR_H = 1.56
__C.ANCHOR_Z = -1.0 - cfg.ANCHOR_H/2
__C.RPN_POS_IOU = 0.6
__C.RPN_NEG_IOU = 0.45

# cal mean from train set
__C.MATRIX_pix_T_rect = ([[719.787081,    0.,            608.463003, 44.9538775],
                  [0.,            719.787081,    174.545111, 0.1066855],
                  [0.,            0.,            1.,         3.0106472e-03],
                  [0.,            0.,            0.,         0]])

# cal mean from train set
__C.MATRIX_cam_T_velo = ([
    [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
    [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
    [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
    [0, 0, 0, 1]
])
# cal mean from train set
__C.MATRIX_rect_T_cam = ([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]
])

def delta_to_boxes3d(deltas, anchors):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)
    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    anchors_reshaped = tf.reshape(anchors, [-1, 7])
    B = int(deltas.get_shape()[0])
    deltas = tf.reshape(deltas, [B, -1, 7])
    anchors_diag = tf.sqrt(tf.square(anchors_reshaped[:, 4]) + tf.square(anchors_reshaped[:, 5]))
    box0 = deltas[:,:,0] * anchors_diag + anchors_reshaped[:,0]
    box1 = deltas[:,:,1] * anchors_diag + anchors_reshaped[:,1]
    box2 = deltas[:,:,2] * cfg.ANCHOR_H + anchors_reshaped[:,2]
    box3 = tf.exp(deltas[:,:,3]) * anchors_reshaped[:,3]
    box4 = tf.exp(deltas[:,:,4]) * anchors_reshaped[:,4]
    box5 = tf.exp(deltas[:,:,5]) * anchors_reshaped[:,5]
    box6 = deltas[:,:,6] + anchors_reshaped[:,6]
    boxes3d = tf.stack([box0, box1, box2, box3, box4, box5, box6], axis=2)
    return boxes3d
    
def corner_to_standup_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    box0 = tf.reduce_min(boxes_corner[:, :, 0], axis=1)
    box1 = tf.reduce_min(boxes_corner[:, :, 1], axis=1)
    box2 = tf.reduce_max(boxes_corner[:, :, 0], axis=1)
    box3 = tf.reduce_max(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d = tf.stack([box0, box1, box2, box3], axis=1)
    return standup_boxes2d

def center_to_corner_box2d(boxes_center, coordinate='lidar', rect_T_cam=None, cam_T_velo=None):
    return tf.py_func(utils_rpn.center_to_corner_box2d, [boxes_center], tf.float32)

def trim_gt_boxes(gt_boxes3d):
    # gt_boxes3d is N x 7
    sums = tf.reduce_sum(tf.abs(gt_boxes3d), axis=1)
    inds = tf.where(tf.greater(sums, 0.0))
    gt_boxes3d = tf.gather(gt_boxes3d, inds)
    gt_boxes3d = tf.reshape(gt_boxes3d, [-1, 7])
    return gt_boxes3d


def draw_lidar_box3d_on_image(img, boxes3d, scores, gt_boxes3d, pix_T_rect, rect_T_cam, cam_T_velo, draw_dets=True,
                              color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1):
    # first we need to get rid of invalid gt boxes
    gt_boxes3d = trim_gt_boxes(gt_boxes3d)
    return tf.py_func(utils_rpn.draw_lidar_box3d_on_image, [img, boxes3d, scores, gt_boxes3d,
                                                            pix_T_rect, rect_T_cam, cam_T_velo,
                                                            draw_dets], tf.uint8)

def draw_lidar_box3d_on_birdview(img, boxes3d, scores, gt_boxes3d, pix_T_rect, rect_T_cam, cam_T_velo, draw_dets=True,
                                 color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1, factor=1):
    # first we need to get rid of invalid gt boxes
    gt_boxes3d = trim_gt_boxes(gt_boxes3d)
    return tf.py_func(utils_rpn.draw_lidar_box3d_on_birdview, [img, boxes3d, scores, gt_boxes3d,
                                                               pix_T_rect, rect_T_cam, cam_T_velo,
                                                               draw_dets], tf.uint8)

def center_to_corner_box3d(boxes_center, rect_T_cam, cam_T_velo):
    # assume we are in lidar coords
    # boxes_center is B x N x 7
    B = int(boxes_center.get_shape()[0])
    N = int(boxes_center.get_shape()[1])
    corners = tf.map_fn(center_to_corner_box3d_single, [boxes_center,
                                                        rect_T_cam, cam_T_velo], tf.float32)
    # assert shape
    corners = tf.reshape(corners, [B, N, 8, 3])
    return corners

def center_to_corner_box3d_single((boxes_center, rect_T_cam, cam_T_velo)):
    # assume we are in lidar coords
    return tf.py_func(utils_rpn.center_to_corner_box3d, [boxes_center,
                                                         rect_T_cam, cam_T_velo], tf.float32)
def corner_to_center_box3d(boxes_corner, rect_T_cam, cam_T_velo):
    # assume we are in lidar coords
    # boxes_corner is B x N x 8 x 3
    B = int(boxes_corner.get_shape()[0])
    N = int(boxes_corner.get_shape()[1])
    centers = tf.map_fn(corner_to_center_box3d_single, [boxes_corner,
                                                        rect_T_cam, cam_T_velo], tf.float32)
    centers = tf.reshape(centers, [B, N, 7])
    return centers
def corner_to_center_box3d_single((boxes_corner, rect_T_cam, cam_T_velo)):
    # assume we are in lidar coords
    return tf.py_func(utils_rpn.corner_to_center_box3d, [boxes_corner,
                                                         rect_T_cam, cam_T_velo], tf.float32)

def lidar_to_camera_point((points, rect_T_cam, cam_T_velo)):
    return tf.py_func(utils_rpn.lidar_to_camera_point, [points,
                                                        rect_T_cam, cam_T_velo], tf.float32)
def cal_rpn_target2(boxes, rect_T_cam, cam_T_velo):
    # assume we are dealing with Car class and want the targets in lidar coords
    return tf.map_fn(cal_rpn_target2_single, [boxes,
                                              rect_T_cam, cam_T_velo],
                     [tf.float32, tf.float32, tf.float32])




def cal_rpn_target2_single((boxes, rect_T_cam, cam_T_velo)):
    # assume we are dealing with Car class and want the targets in lidar coords

    feature_map_shape = [hyp.BY/2, hyp.BX/2]
    pos, neg, tar = tf.py_func(utils_rpn.cal_rpn_target2,
                               [feature_map_shape,
                                boxes,
                                rect_T_cam, cam_T_velo],
                               [tf.float32, tf.float32, tf.float32])
    ## assert shapes
    # 2 anchor sizes at each position
    pos = tf.reshape(pos, [hyp.BY/2, hyp.BX/2, 2])
    neg = tf.reshape(neg, [hyp.BY/2, hyp.BX/2, 2])
    # 7 regression targets
    tar = tf.reshape(tar, [hyp.BY/2, hyp.BX/2, 14])
    return [pos, neg, tar]

def cal_ap(pred_boxes2d_bird,
           gt_boxes2d_bird,
           pred_probs):

    ap = tf.py_func(utils_rpn.cal_ap, [pred_boxes2d_bird, \
        gt_boxes2d_bird, pred_probs], tf.float32)

    return ap

def get_box_label(pred_boxes2d_bird,
                  gt_boxes2d_bird):

    labels = tf.py_func(utils_rpn.get_box_label, [pred_boxes2d_bird, \
        gt_boxes2d_bird], tf.int32)

    return labels

"""
def summarize_AP_on_bird_bbox(prob_output,
                              delta_output,
                              boxes3D,
                              ):
    #anchors = utils_rpn.cal_anchors()
    #anchors = tf.constant(anchors.astype(np.float32))
    #batch_boxes3d = delta_to_boxes3d(delta_output, anchors)
    
    import ipdb; ipdb.set_trace()
 
    #h, w, l, x, y, z, ry
    a, b, c, d, e, f, g = tf.unstack(batch_boxes3d, axis=2)
    batch_boxes2d = tf.stack([a, b, d, e, f], axis=2)
    ## get bt boxes2d in bird view from boxes3D
    a_, b_, c_, d_, e_, f_, g_ = tf.unstack(boxes3D, axis=2)
    batch_gt_boxes2d = tf.stack([a_, b_, d_, e_, f_], axis=2)
    batch_probs = tf.reshape(prob_output, [B, -1])

    #boxes3d = batch_boxes3d[0]
    #batch_id = 0
    #tmp_gt_boxes2d = batch_gt_boxes2d[batch_id]
    #tmp_boxes2d = batch_boxes2d[batch_id]
    #probs = batch_probs[batch_id]
    #boxes2d = utils_rpn_tf.corner_to_standup_box2d(
    #    utils_rpn_tf.center_to_corner_box2d(tmp_boxes2d))
    #gt_boxes2d = utils_rpn_tf.corner_to_standup_box2d(
    #    utils_rpn_tf.center_to_corner_box2d(tmp_gt_boxes2d))

    #ap = utils_rpn_tf.cal_ap(boxes2d, gt_boxes2d, probs, \
    #    rect_T_cam, cam_T_velo) 
    #print(prob_output.get_shape())
    #print(batch_boxes2d.get_shape())
    #sys.exit()
    aps = tf.map_fn(summarize_warp_ap, (batch_boxes2d, batch_gt_boxes2d,
        batch_probs), tf.float32)
    ap = tf.reduce_mean(aps)
    tf.summary.scalar('AP', ap)

"""
def summarize_AP_on_bird_bbox2(prob_output,
                              delta_output,
                              boxes3D,
                              ):
    anchors = utils_rpn.cal_anchors()
    anchors = tf.constant(anchors.astype(np.float32))
    batch_boxes3d = delta_to_boxes3d(delta_output, anchors)
    a, b, c, d, e, f, g = tf.unstack(batch_boxes3d, axis=2)
    batch_boxes2d = tf.stack([a, b, d, e, f], axis=2)
    ## get bt boxes2d in bird view from boxes3D
    a_, b_, c_, d_, e_, f_, g_ = tf.unstack(boxes3D, axis=2)
    batch_gt_boxes2d = tf.stack([a_, b_, d_, e_, f_], axis=2)
    batch_probs = tf.reshape(prob_output, [B, -1])

    #boxes3d = batch_boxes3d[0]
    #batch_id = 0
    #tmp_gt_boxes2d = batch_gt_boxes2d[batch_id]
    #tmp_boxes2d = batch_boxes2d[batch_id]
    #probs = batch_probs[batch_id]
    #boxes2d = utils_rpn_tf.corner_to_standup_box2d(
    #    utils_rpn_tf.center_to_corner_box2d(tmp_boxes2d))
    #gt_boxes2d = utils_rpn_tf.corner_to_standup_box2d(
    #    utils_rpn_tf.center_to_corner_box2d(tmp_gt_boxes2d))

    #ap = utils_rpn_tf.cal_ap(boxes2d, gt_boxes2d, probs, \
    #    rect_T_cam, cam_T_velo) 
    labels, tmp_probs = tf.map_fn(summarize_warp_ap2, (batch_boxes2d, batch_gt_boxes2d,
        batch_probs), (tf.int32, tf.float32))
    #labels, tmp_probs = summarize_warp_ap2((batch_boxes2d[0], batch_gt_boxes2d[0],
    #    batch_probs[0]))

    ap = tf.py_func(utils_rpn.ap_from_label_prob, [labels, tmp_probs], tf.float32)
    tf.summary.scalar('AP2', ap)

def summarize_AP_on_bird_bbox3(prob_output,
                              delta_output,
                              boxes3D,
                              ):
    anchors = utils_rpn.cal_anchors()
    anchors = tf.constant(anchors.astype(np.float32))
    batch_boxes3d = delta_to_boxes3d(delta_output, anchors)
    a, b, c, d, e, f, g = tf.unstack(batch_boxes3d, axis=2)
    batch_boxes2d = tf.stack([a, b, d, e, f], axis=2)
    ## get bt boxes2d in bird view from boxes3D
    a_, b_, c_, d_, e_, f_, g_ = tf.unstack(boxes3D, axis=2)
    batch_gt_boxes2d = tf.stack([a_, b_, d_, e_, f_], axis=2)
    batch_probs = tf.reshape(prob_output, [B, -1])
    #print(batch_probs.get_shape().as_list())
    #print(batch_gt_boxes2d.get_shape().as_list())
    #print(batch_boxes2d.get_shape().as_list())
    #sys.exit()
    
    ap = tf.py_func(utils_rpn.compute_ap_batch, (batch_boxes2d, batch_gt_boxes2d,
        batch_probs), (tf.float32))

    tf.summary.scalar('AP3', ap)

def summarize_warp_ap((tmp_boxes2d, tmp_gt_boxes2d, probs)):
    boxes2d = corner_to_standup_box2d(
        center_to_corner_box2d(tmp_boxes2d))
    gt_boxes2d = corner_to_standup_box2d(
        center_to_corner_box2d(tmp_gt_boxes2d))
    
    ## remove NMS currently, since we want to keep tensor shape consistent and 
    ## doding NMS doesn't change a lot to AP currently
    #TOPK = 200
    #ind = tf.image.non_max_suppression(
    #    boxes2d,
    #    probs,
    #    max_output_size=TOPK,
    #    iou_threshold=0.1)
    #tmp_probs = tf.reshape(tf.gather(probs, ind), [-1])
    #tmp_boxes_2d = tf.reshape(tf.gather(boxes2d, ind), [-1, 5])

    #ap = cal_ap(tmp_boxes_2d, gt_boxes2d, probs) 
    #ap = cal_ap(tmp_boxes_2d, gt_boxes2d, tmp_probs) 
    ap = cal_ap(boxes2d, gt_boxes2d, probs) 

    return ap

def summarize_warp_ap2((tmp_boxes2d, tmp_gt_boxes2d, probs)):
    boxes2d = corner_to_standup_box2d(
        center_to_corner_box2d(tmp_boxes2d))
    gt_boxes2d = corner_to_standup_box2d(
        center_to_corner_box2d(tmp_gt_boxes2d))
    
    ## remove NMS currently, since we want to keep tensor shape consistent and 
    ## doding NMS doesn't change a lot to AP currently
    #TOPK = 200
    #ind = tf.image.non_max_suppression(
    #    boxes2d,
    #    probs,
    #    max_output_size=TOPK,
    #    iou_threshold=0.1)
    #tmp_probs = tf.reshape(tf.gather(probs, ind), [-1])
    #tmp_boxes_2d = tf.reshape(tf.gather(boxes2d, ind), [-1, 5])

    #labels = get_box_label(tmp_boxes_2d, gt_boxes2d)
    tmp_probs = probs
    labels = get_box_label(boxes2d, gt_boxes2d)

    return (labels, tmp_probs)
"""
