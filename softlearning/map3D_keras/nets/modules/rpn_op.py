



def generate_pos_anchor_gt(self, memory):
    bs, nobj, S, _, _ = self.voxel_gt.get_shape()
    if len(memory.get_shape()) == 5:
        _, S_d, S_h, S_w, _ = memory.get_shape()
    elif len(memory.get_shape()) == 4:
        _, S_h, S_w, _ = memory.get_shape()
        S_d = S_h
    self.nobj = nobj.value
    self.voxel_size = S_d.value
    self.batch_size = bs.value
    grid = self.meshgrid(S_d.value, S_h.value, S_w.value)
    grid = (grid + 0.5 * tf.ones_like(grid, dtype=tf.float32)) * (1/S_d.value)
    delta_pos =tf.expand_dims(tf.expand_dims(tf.expand_dims(self.object_bbox_center, 2), 3), 4) - tf.expand_dims(tf.expand_dims(grid, 0), 1)

    delta_len = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(self.object_bbox[:,:,:,1] - self.object_bbox[:,:,:,0],2),3), 4), [1,1,32,32,32,1])
        # batch_size x nobj x S x S x S x 6 (dz, dy, dx, d, h, w)
    delta_gt = tf.concat([delta_pos, delta_len], 5)
    self.delta_gt = delta_gt
    resized_voxel_gt = utils.tfutil.pool3d(tf.transpose(self.voxel_gt, [0, 2, 3, 4, 1]), factor = int(S.value/S_d.value))
    self.voxel_occ_bin = utils.utils.binarize(resized_voxel_gt, 0.5)
    #object_pos_mask = tf.expand_dims(tf.transpose(pos_equal_one, [0, 4, 1, 2, 3]), 5)
    object_dist = tf.reduce_max(tf.divide(tf.abs(delta_pos), delta_len * 0.5), 5)
    object_dist_mask = tf.expand_dims(tf.ones_like(object_dist)\
        - utils.utils.binarize(object_dist, 0.5), 5)
    object_neg_dist_mask = tf.ones_like(object_dist) - utils.utils.binarize(object_dist, 0.8)
    anchors_gt = None





def rpn_proposal_and_gt(self, memory):
    """
    input:
        memory: [(Txbatch_size, 32, 32, 32, 8)]
    outputs: 
        pos_equal_one, neg_equal_one, predict_pos_one: [T x batch_size, 32, 32, 32]
        anchors_gt, predict_anchors: (T x batch_size, 32 ,32 ,32, 6)

    """
    # build p map
    pos_equal_one, neg_equal_one, anchors_gt = self.generate_pos_anchor_gt(memory[0])
    # copy for several timesteps
    pos_equal_one = tf.tile(pos_equal_one, [self.nviews, 1, 1, 1]) 
    neg_equal_one = tf.tile(neg_equal_one, [self.nviews, 1, 1, 1]) 
    anchors_gt = tf.tile(anchors_gt, [self.nviews, 1, 1, 1, 1]) 

    with tf.variable_scope('RPN'):
        net_out =  utils.nets.encoder_decoder3D(memory)[-1]
    net = utils.nets.slim2_conv3d(net_out, 7, 3, stride=1, padding="SAME")
    bs, s_d, s_h, s_w, c = net.get_shape()

    predict_anchors = tf.slice(net, [0, 0, 0, 0, 0], [bs, s_d, s_h, s_w, c-1])
    predict_pos_one = tf.squeeze(tf.slice(net, [0, 0, 0, 0, c.value-1], [bs, s_d, s_h, s_w, 1]), [-1])
    return pos_equal_one, neg_equal_one, anchors_gt, tf.nn.sigmoid(predict_pos_one), predict_anchors






def rpn_proposal_graph(memory, predict_pos_one, predict_anchors, predict=True):
    ######################## ROI generation ####################
    # self.object_bbox: batch_size x nobj x 3 x 2
    object_bbox_gt = tf.tile(self.object_bbox, [self.nviews, 1, 1, 1]) 
    # resize_gt_first
    bs, nobj, s, s, s = self.voxel_gt.get_shape()
    voxel_gt_tmp = tf.reshape(self.voxel_gt, [bs * nobj, s, s, s, 1]) 
    bbox = np.zeros((bs.value * nobj.value, 1, 3, 2), dtype=np.float32)
    bbox[:,:,:,1] = 1 
    bbox = tf.constant(bbox, dtype=tf.float32)
    voxel_gt = utils.utils.binarize(utils.voxel.crop_and_resize_3d_box2(voxel_gt_tmp, bbox, 32), const.BIN_THRES)
    voxel_gt = tf.reshape(voxel_gt, [bs, nobj, 32, 32, 32])

    object_mask_gt = tf.expand_dims(tf.tile(voxel_gt, [self.nviews, 1, 1, 1, 1]), -1) 
    #object_centered_voxel_gt = tf.tile(self.centered_voxels_gt, [self.nviews, 1, 1, 1, 1, 1])
    high_prob_indices = tf.where(predict_pos_one > const.P_THRES)

    # build prediction target
    self.high_prob_indices = high_prob_indices
    T_bs = self.predict_pos_one.get_shape()[0]
    bs_rois = []
    bs_deltas = []
    bs_masks = []
    bs_class_labels = []
    bs_scores = []
    self.bs_selected_boxes = []
    self.bs_selected_scores = []
    self.bs_overlaps = []
    for i in range(T_bs):
        # high_prob_indices
        rois, deltas, masks, class_labels, selected_boxes, selected_boxes_scores, overlaps = self.detection_target_graph(i, high_prob_indices, object_bbox_gt, object_mask_gt, predict_pos_one, predict_anchors)
        bs_rois.append(rois)
        bs_deltas.append(deltas)
        bs_masks.append(masks)
        bs_class_labels.append(class_labels)
        self.bs_selected_boxes.append(selected_boxes)
        self.bs_selected_scores.append(selected_boxes_scores)
        self.bs_overlaps.append(overlaps)

    self.bs_rois = tf.stack(bs_rois)
    self.bs_deltas = tf.stack(bs_deltas)
    self.bs_masks = tf.stack(bs_masks)
    self.bs_class_labels = tf.stack(bs_class_labels)
"""
