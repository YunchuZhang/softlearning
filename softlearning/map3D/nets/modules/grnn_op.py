import utils_map as utils
import tensorflow as tf
from . import summary_op as summ_op
from . import loss as loss
def get_outputs2Denc(inputs, is_training, is_not_bn=False):
    """
    inputs: [images] * num_views
    images: batch_size x img_h x img_w x img_c
    
    """
    fn = lambda x: utils.nets.encoder2D(x, is_training, is_not_bn=is_not_bn)
    with tf.variable_scope('2Dencoder'): 
        return utils.tfutil.concat_apply_split(
            inputs,
            fn  
        )   

def get_outputs2Ddec(inputs, is_training, fs_2D=32, is_not_bn=False):
    """   
    inputs: [[images]* num_views] * num_features
    images: batch_size x img_h x img_w x img_c
  
    out:[[image]*num_views] * num_features
    (but only take the last one, so num_feaures=1)
    """
    nlayers = len(inputs)
    nviews = len(inputs[0])
    input = [tf.concat([inputs[nl][n] for n in range(nviews)], 0) for nl in range(nlayers)]

    with tf.variable_scope('2Ddecoder'):
        out = utils.nets.decoder2D(input[::-1], is_training, apply_3to2=False, dim_base=int(fs_2D/2.0), 
            out_dim=fs_2D, last_stride=2, is_not_bn = is_not_bn)
        return [tf.split(out, nviews)]


def unproject_depth_and_get_outline(inputs):
    # [(32 x 64 x 64 x 34), (32 x 64 x 64 x 34), (32 x 64 x 64 x 34)]

    _inputs = tf.stack(inputs, axis = 0)
    ones = tf.ones_like(_inputs)
    _inputs = tf.concat([ones, _inputs, ones], axis=-1)
    _inputs = tf.map_fn(
            lambda x: utils.nets.unproject2(x, use_outline=True, resize=False),
            _inputs, parallel_iterations = 1
    )

    _inputs =tf.unstack(_inputs, axis = 0)
    return _inputs



def unproject_inputs(inputs, use_outline=False, debug_unproject=False):
    def stack_unproject_unstack(_inputs):
        # [(32 x 64 x 64 x 34), (32 x 64 x 64 x 34), (32 x 64 x 64 x 34)]
        print("_inputs", _inputs)

        _inputs = tf.stack(_inputs, axis = 0)
        _inputs = tf.map_fn(
            lambda x: utils.nets.unproject2(x, use_outline=use_outline, debug_unproject=debug_unproject, resize=False),
            _inputs, parallel_iterations = 1 
        )

        _inputs =tf.unstack(_inputs, axis = 0)
        return _inputs
    return [stack_unproject_unstack(inp) for inp in inputs]

def get_refined3D(unprojected_features, is_training, scope="3DED", is_summ_feat=False, summ_inputs=None, is_not_bn=False):

    nviews = len(unprojected_features[0])
    stacked_features = [tf.concat(feature, 0) for feature in unprojected_features]

    with tf.variable_scope(scope):
        output = utils.nets.encoder_decoder3D(stacked_features, is_training, is_not_bn=is_not_bn)[-1:]
    unprojected_features_refined = [tf.split(feat, nviews) for feat in output]
    if is_summ_feat:
        # visualize feature after unprojection and after refinement after unprojection
        summ_op.summ_refined3D(unprojected_features, unprojected_features_refined, summ_inputs=summ_inputs)
        
    return unprojected_features_refined


def aggregate_multi_steps(features, aggre="gru"):
    if aggre == "gru":
        hidden_states = []
        for f_id, feature in enumerate(features):
            #output: [Txbatch_size, s, s, s, c]
            output = utils.nets.gru_aggregator(feature, f"f{f_id}", is_multi_t_output=True)
            hidden_states.append(output)
        return hidden_states
    elif aggre == "average":
        nsteps = len(features[0])
        aggre_t = []
        for t in range(nsteps):
            n = 1.0/float(t + 1)
            aggre_t.append([sum(feat[:t+1])*n for feat in features])
        # bind different time steps
        combined_feat_across_t = []
        for feat_id in range(len(aggre_t[0])):
            combined_feat_across_t.append(tf.concat([aggre_t[t][feat_id] for t in range(nsteps)], 0))
        return combined_feat_across_t
    else:
        raise Exception('unknown aggregation method')

def translate_multiple(dthetas, phi1s, phi2s, voxs):
    dthetas = tf.stack(dthetas, axis = 0)
    phi1s = tf.stack(phi1s, 0)
    phi2s = tf.stack(phi2s, 0)
    voxs = tf.stack(voxs, 0)

    f = lambda x: utils.voxel.translate_given_angles(*x)
    out = tf.map_fn(f, [dthetas, phi1s, phi2s, voxs], dtype = tf.float32)
    return tf.unstack(out, axis = 0)


def align_to_base_single(feature, thetas, phis, base_thetas, base_phis):
    dthetas = [base_thetas - theta for theta in thetas]
    phi1s = phis
    phi2s = [base_phis for _ in phis]
    return translate_multiple(dthetas, phi1s, phi2s, feature)

def align_to_base(features, thetas, phis, base_thetas, base_phis):
    return [align_to_base_single(feature, thetas, phis, base_thetas, base_phis) for feature in features]

def project_inputs(inputs):
    #[(8x4x4x4x256), (8x8x8x8x128), (8x16x16x16x64), (8x32x32x32x32)]
    # reutrn [(batch_sizexheightxwidthxdepthxc)]
    return [
        utils.voxel.transformer_postprocess(
            utils.voxel.project_voxel(feature)
        )
        for feature in inputs
    ]


def pass_rotate_to_base(inputs, thetas, phis, base_theta, base_phi, aggre="gru",\
                                      query=None, is_summ_feat=False):

    aligned_features = align_to_base(inputs, thetas, phis, base_theta, base_phi)

    return aligned_features

def pass_aggregate(aligned_features, aggre="gru", query=None, is_summ_feat=False):
    extra_outputs=dict()
    memory = aggregate_multi_steps(aligned_features, aggre=aggre)
    if query:
        query_features_concat = [tf.concat(feat, 0) for feat in query]
        logits, prod, _ = loss.match_memory(memory, query_features_concat)
        extra_outputs["logits"] = logits

    return memory, extra_outputs


def pass_rotate_to_base_and_aggregate(inputs, thetas, phis, base_theta, base_phi, aggre="gru",\
                                      query=None, is_summ_feat=False):
    """
    inputs = [[image] * num_views]
    thetas:[batch_size)] * num_views, [240, 200] 
    phis: [(batch_size)] * num_views, [40, 60]
    base_thetas: [batch_size]
    base_thetas: [batch_size]

    """
    extra_outputs=dict()

    """
    # inputs: [[]_(num_views)]_(num_feat_layers)
    if const.USE_PREDICTED_EGO:
        num_views = len(inputs[0])
        gru_cells = []
        memorys = []
        weighted_inputs = []
        logits = []
        input_at_time_0 = []
        for t in range(num_views):
            memory_time_t = []
            feat_at_time_t = [feat[t] for feat in inputs]
            query_at_time_t = [feat[t] for feat in query]
            # update memory first
            for feat_id, feat in enumerate(feat_at_time_t):
                bs, h, w, d, c = feat.get_shape()
                if t == 0:
                    aligned_first_view = self.align_to_base([[feat]], self.thetas[t:t+1], self.phis[t:t+1])
                    gru_cell = utils.nets.convgru_cell([h,w,d], filters=c)
                    gru_cells.append(gru_cell)
                    hidden = tf.zeros_like(feat)
                    input = aligned_first_view[0][0]
                    input_at_time_0.append(input)
                    # feature at t=0 aligned to base
                else:
                    hidden = memorys[t - 1][feat_id]
                    input = weighted_inputs[t - 1][feat_id]
                # input: [[batch_size x s x s x s x c]x3]
                # hidden: batch_size x s x s x s x c
                memory_out = gru_cell(input, hidden, scope=f"gru_{feat_id}", reuse=t>0)
                memory_time_t.append(memory_out)
                #query feature
 

            memorys.append(memory_time_t)
            score, prob, weight_features = self.match_memory(memory_time_t, query_at_time_t)
            logits.append(score)
            weighted_inputs.append(weight_features)

        aligned_features = []
        memory = []
        for feat_id in range(len(input_at_time_0)):
            features = [input_at_time_0[feat_id]] + [input[feat_id] for input in weighted_inputs]
            # throw away the last input since it is not integrated
            aligned_features.append(features[:-1])
            memory_to_concat = tf.concat([mem[feat_id] for mem in memorys] , 0)
            memory.append(memory_to_concat)
        logits = tf.concat(logits, 0)
    """
      
    if True:
        #[[(2x32x32x32x8)]_(num_views)]_(num_features)
        aligned_features = align_to_base(inputs, thetas, phis, base_theta, base_phi)
        """
        from utils.vis_np import save_voxel
        for batch_id in range(2):
            save_voxel(aligned_features[0][0][batch_id * 10, :, :, :, 2], f"dump/rotated_voxel_b{batch_id}.binvox")
        import ipdb; ipdb.set_trace()
        """
        #[<6 x 32 x 32 x 32 x 8>]_(num_features)
        memory = aggregate_multi_steps(aligned_features, aggre=aggre)
        if query: 
            query_features_concat = [tf.concat(feat, 0) for feat in query]
            logits, prod, _ = loss.match_memory(memory, query_features_concat)
            extra_outputs["logits"] = logits
    # visualize rotated and 3D memory features (they are aligned)
    """
    if is_summ_feat:
        voxel_scene = utils.voxel.resize_voxel(tf.expand_dims(tf.reduce_sum(self.voxel_gt, 1), -1), scale=0.25)
        for f_id, feat in enumerate(aligned_features):
            s = feat[0].get_shape()[1].value
            feat_to_print = tf.concat([feat[0][:,i ,:,:,2:5] for i in range(0, s, 2)], 1)
            max_value = tf.reshape(tf.reduce_max(feat_to_print, [1,2,3]), [-1, 1, 1, 1])
            gt_voxel_to_print = tf.concat([tf.tile(voxel_scene[:, i, :, :, :], [1,1,1,3]) for i in range(0, s, 2)], 1)
            summ.image(f"feat/step0_aligned_f{f_id}", tf.concat([gt_voxel_to_print * max_value, feat_to_print], 2))
            tf.summary.histogram(f"feat/step0_aligned_f{f_id}_summ", feat[0])

        memory_3D_split = [tf.split(feat, self.nviews, axis=0) for feat in memory]
        for f_id, feat in enumerate(memory_3D_split):
            s = feat[0].get_shape()[1].value
            gt_voxel_to_print = tf.concat([tf.tile(voxel_scene[:, i, :, :, :], [1,1,1,3]) for i in range(0, s, 2)], 1)
            for step_id in range(self.nviews):
                feat_to_print = tf.concat([feat[step_id][:,i ,:,:,2:5] for i in range(0, s, 2)], 1)
                max_value = tf.reshape(tf.reduce_max(feat_to_print, [1,2,3]), [-1, 1, 1, 1])
                summ.image(f"feat/step{step_id}_memory_f{f_id}", tf.concat([gt_voxel_to_print * max_value, feat_to_print], 2))
                tf.summary.histogram(f"feat/step{step_id}_memory_f{f_id}_summ", feat[step_id])
    """
    return memory, aligned_features, extra_outputs
    

