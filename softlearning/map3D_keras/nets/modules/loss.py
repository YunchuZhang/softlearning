import tensorflow as tf
import utils_map as utils

def set_up_rotate_helper(query_features):
    rotate_helper = []
    for feature in query_features:
        bs, depth, height, width, c = feature.get_shape().as_list()
        rh = utils.voxel.rotate_helper((depth, height, width), (depth, height, width),
            const.HV, const.VV, const.MINH, const.MINV, const.HDELTA, const.VDELTA)
        rotate_helper.append(rh)
    return rotate_helper

def rotate_features_to_multiple_angles(features):
    rotate_helper = set_up_rotate_helper(features)
    output_feat = []
    for feat_id, feature in enumerate(features):
        output_feat.append(rotate_helper[feat_id].transform(feature))
    return output_feat



def match_memory(memory_3D, query_features):
    # query_features: [6, 32, 32, 32, 8]
    # rotated_features: [6, 54, 32, 32, 32, 8]
    rotated_features = rotate_features_to_multiple_angles(query_features)

    similarity = []
    for feat_id, feature in enumerate(rotated_features):
        inner = tf.multiply(feature, tf.expand_dims(memory_3D[feat_id], 1))
        inner = tf.reduce_mean(inner, [2, 3, 4, 5])
        similarity.append(inner)
    scale = 1/(len(similarity))
    score = tf.add_n(similarity) * scale
    prob = tf.nn.softmax(score, 1)

    weighted_features = []
    for feat_id, feature in enumerate(rotated_features):
        bs = tf.shape(prob)[0]
        nangle = tf.shape(prob)[1]
        weighted_feature = tf.reduce_sum(tf.reshape(prob, [bs, nangle, 1,1,1,1]) * feature, 1)
        weighted_features.append(weighted_feature)
    return score, tf.nn.softmax(score, 1), weighted_features

