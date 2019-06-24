import tensorflow as tf
from tensorflow import summary as summ
import constants as const

def get_input_image(all_frames, step = 0, type="rgb", shape=64, reverse=False):
    if type == "rgb":
        img = all_frames[step][:,:,:,-3:]
    elif type == "depth":
        img = all_frames[step][:,:,:,1:2]
    elif type == "mask":
        img = all_frames[step][:,:,:,0:1]
    else:
        assert(1==2)
    img = tf.image.resize_images(img, [shape, shape])
    if reverse:
        img = tf.reverse(img, [1, 2])
    return img


def summ_refined3D(unprojected_features, unprojected_features_refined, summ_inputs):
    """
    unprojected_features_refined: [[image]*num_refined]
    """
    # visualize feature after unprojection and after refinement after unprojection
    for f_id, feat in enumerate(unprojected_features_refined):
        s = feat[0].get_shape()[1].value
        feat_to_print = tf.concat([feat[0][:,i ,:,:,2:5]\
                                   for i in range(0, s, 2)], 1)
        feat_stage1_to_print = tf.concat([\
            unprojected_features[f_id][0][:,i ,:,:,2:5]\
            for i in range(0, s, 2)], 1)

        max_value = tf.reshape(tf.reduce_max(feat_to_print, [1,2,3]),\
                                            [-1, 1, 1, 1]) 
        all_frames = summ_inputs["all_frames"] 
        image_to_print = tf.concat([\
            tf.concat([get_input_image(all_frames, 0, "rgb", s, reverse=True) * max_value, feat_stage1_to_print], 1),\
            tf.concat([get_input_image(all_frames, 0, "rgb", s, reverse=True) * max_value, feat_stage1_to_print * max_value], 1),\
            tf.concat([get_input_image(all_frames, 0, "rgb", s, reverse=True) * max_value, feat_to_print], 1)], 2)
        if not const.eager:
            summ.image(f"feat/step0_unprojected_refine_f{f_id}", image_to_print)
            summ.histogram(f"feat/step0_unprojected_refine_f{f_id}_summ", feat[0])
            summ.histogram(f"feat/step0_unprojected_f{f_id}_summ", unprojected_features[f_id][0])
