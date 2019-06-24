# import __init__path
import sys
import sys
# sys.path.append("/Users/mihirprabhudesai/Documents/projects/rl/softlearning/softlearning/map3D")
# print(sys.path)
# print("check")
import __init__path
from nets.Net import Net
import constants as const

import utils_map
from utils_map import voxel
from utils_map import vis_tf
# import 
import tensorflow as tf
import numpy as np
from munch import Munch
from tensorflow import summary as summ
from .modules.GraphNet import GraphNet
from .modules import grnn_op
from .modules import vis_op
import tfquaternion as tfq
import copy
import ipdb

st = ipdb.set_trace

class BulletPushBase(Net):
    
    def go_up_to_loss(self, index=None, is_training=None):
        if const.eager:
            self.is_training = tf.constant(is_training, tf.bool)
        self.orn_reset_base = tf.constant([1, 0, 0, 0], dtype=tf.float32)
        self.setup_data(index)
        self.extra_out = dict()
        with tf.variable_scope("main"):
            self.predict()
            self.add_weights("main_weights")
        self.loss()
        return self.loss_

    def setup_data(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def split_states_g(self, states, mode="obj_agent"):
        if mode == "obj_agent":
            return states[..., :self.nobjs, :], states[..., self.nobjs:, :]
        elif mode == "h13":
            return states[..., :3], states[..., 3:7], states[..., 7:]
        else:
            raise Exception(f"data format is not supported: {mode}")
    def loss(self):
        self.gt_delta = tf.concat([self.target.delta_obj_state, self.target.delta_agent_state], 2)[:, :self.T, ...]

        gt_xyz_delta, gt_orn_delta, gt_vel_delta = self.split_states_g(self.gt_delta, mode="h13")
        est_xyz_delta, est_orn_delta, est_vel_delta = self.split_states_g(self.est_dyn_states_delta, mode="h13")

        if const.IS_PREDICT_CONTACT:

            contact_logits = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gt_is_contact,\
              logits=self.pred_contact)
            contact_loss = tf.reduce_sum(contact_logits * self.target.all_class[:,:self.T, :self.nobjs])/tf.reduce_sum(self.target.all_class[:,:self.T, :self.nobjs])

            contact_acc = tf.reduce_sum(tf.dtypes.cast(tf.equal(tf.math.greater(self.gt_is_contact, 0.5),\
                   tf.math.greater(tf.nn.sigmoid(self.pred_contact), 0.5)), dtype=tf.float32)\
                   * self.target.all_class[:,:self.T, :self.nobjs])\
                  /tf.reduce_sum(self.target.all_class[:,:self.T, :self.nobjs])

        if const.IS_PREDICT_CONTACT and const.mode=="train":
            contact_mask = tf.concat([self.gt_is_contact, tf.ones((const.BS, self.T, 1), dtype=tf.float32)], -1)
        else:
            contact_mask = tf.ones_like(self.target.all_class[:,:self.T,:], dtype=tf.float32)
        angle_loss = utils.losses.quatloss(est_orn_delta, gt_orn_delta,
                                           mask=self.target.all_class[:,:self.T,:] * contact_mask)

        angle_l1_loss = utils.losses.l1loss(est_orn_delta, gt_orn_delta,
                                           mask=tf.expand_dims(self.target.all_class[:,:self.T,:] * contact_mask, -1))

        pos_loss = utils.losses.l2loss(est_xyz_delta, gt_xyz_delta,
                                       mask = tf.expand_dims(self.target.all_class[:,:self.T,:] * contact_mask, -1))
        vel_loss = utils.losses.l2loss(est_vel_delta, gt_vel_delta,
                                       mask = tf.expand_dims(self.target.all_class[:,:self.T,:] * contact_mask, -1))

        pos_l1_loss = utils.losses.l1loss(est_xyz_delta, gt_xyz_delta,
                                       mask = tf.expand_dims(self.target.all_class[:,:self.T,:] * contact_mask, -1))
        vel_l1_loss = utils.losses.l1loss(est_vel_delta, gt_vel_delta,
                                       mask = tf.expand_dims(self.target.all_class[:,:self.T,:] * contact_mask, -1))

        self.loss_ = pos_l1_loss + angle_loss + vel_l1_loss
        if const.IS_PREDICT_CONTACT:
            #if const.PRETRAIN_CONTACT:
            #    self.loss_ = contact_loss
            #else:
            self.loss_ += 0.01 * contact_loss
        pos_eud_loss = utils.losses.eudloss(est_xyz_delta, gt_xyz_delta,\
                                            mask = self.target.all_class[:,:self.T,:])

        # agent loss
        est_xyz_delta_object, est_xyz_delta_agent = self.split_states_g(est_xyz_delta)
        est_orn_delta_object, est_orn_delta_agent = self.split_states_g(est_orn_delta)
        gt_xyz_delta_object, gt_xyz_delta_agent = self.split_states_g(gt_xyz_delta)
        gt_orn_delta_object, gt_orn_delta_agent = self.split_states_g(gt_orn_delta)

        tf.summary.histogram("gt_xyz_delta_object", tf.linalg.norm(gt_xyz_delta_object[:,0,0,:], axis=-1),\
                             collections=["scalar", "all"])
        tf.summary.histogram("gt_xyz_delta_object_l1", tf.reduce_sum(tf.abs(gt_xyz_delta_object[:,0,0,:]), axis=-1),\
                             collections=["scalar", "all"])

        tf.summary.histogram("est_xyz_delta_object", tf.linalg.norm(est_xyz_delta_object[:,0,0,:], axis=-1),\
                             collections=["scalar", "all"])
        tf.summary.histogram("est_xyz_delta_object_l1", tf.reduce_sum(tf.abs(est_xyz_delta_object[:,0,0,:]), axis=-1),\
                             collections=["scalar", "all"])
        tf.summary.histogram("gt_orn_delta_object", tf.linalg.norm(\
                              tf.sign(gt_orn_delta_object[:, 0,0,:1]) * gt_orn_delta_object[:,0,0,:] \
                              - tf.expand_dims(self.orn_base, 0), axis=-1), collections=["scalar", "all"])
        tf.summary.histogram("gt_orn_delta_object_l1", tf.reduce_sum(tf.abs(\
                              tf.sign(gt_orn_delta_object[:, 0, 0, :1]) * gt_orn_delta_object[:,0,0,:] \
                              - tf.expand_dims(self.orn_base, 0)), axis=-1), collections=["scalar", "all"])

        mask_object, mask_agent = self.split_states_g(tf.expand_dims(self.target.all_class[:,:self.T,:], -1))


        pos_object_loss = utils.losses.l2loss(est_xyz_delta_object, gt_xyz_delta_object,
                                            mask=mask_object)
        self.pos_object_loss_dump = pos_object_loss
        pos_object_eud_loss = utils.losses.eudloss(est_xyz_delta_object, gt_xyz_delta_object,\
                                            mask = mask_object[...,0])
        pos_agent_loss = utils.losses.l2loss(est_xyz_delta_agent, gt_xyz_delta_agent,
                                            mask=mask_agent)
        pos_agent_eud_loss = utils.losses.eudloss(est_xyz_delta_agent, gt_xyz_delta_agent,\
                                            mask = mask_agent[...,0])

        orn_object_loss = utils.losses.quatloss(est_orn_delta_object, gt_orn_delta_object,
                                            mask=mask_object[...,0])
        orn_object_angle_loss = utils.losses.quat_angle_loss(est_orn_delta_object, gt_orn_delta_object,
                                            mask=mask_object[...,0])

        orn_agent_loss = utils.losses.quatloss(est_orn_delta_agent, gt_orn_delta_agent,
                                            mask=mask_agent[...,0])
        orn_agent_angle_loss = utils.losses.quat_angle_loss(est_orn_delta_agent, gt_orn_delta_agent,
                                            mask=mask_agent[...,0])

        if const.mode == "test":
            self.eval_pos_object_loss = pos_object_loss
            self.eval_orn_object_loss = orn_object_loss
            self.eval_pos_object_eud_loss = pos_object_eud_loss
            self.eval_orn_object_angle_loss = orn_object_angle_loss

        # object_loss

        self.loss_ = utils.tfpy.print_val(self.loss_, "loss")
        if const.DEBUG_LOSSES:
            angle_loss = utils.tfpy.print_val(angle_loss, "angle_loss")
            angle_l1_loss = utils.tfpy.print_val(angle_l1_loss, "angle_l1_loss")
            pos_eud_loss = utils.tfpy.print_val(pos_eud_loss, "pos_eud_loss")
            pos_loss = utils.tfpy.print_val(pos_loss, "pos_loss")
            vel_loss = utils.tfpy.print_val(vel_loss, "vel_loss")

            pos_object_loss = utils.tfpy.print_val(pos_object_loss, "pos_object_loss")
            pos_agent_loss = utils.tfpy.print_val(pos_agent_loss, "pos_agent_loss")
            orn_object_loss = utils.tfpy.print_val(orn_object_loss, "orn_object_loss")
            orn_agent_loss = utils.tfpy.print_val(orn_agent_loss, "orn_agent_loss")

            est_orn  = tf.concat([est_orn_delta_object[:,:,0,:], gt_orn_delta_object[:,:,0,:]], -1)
            est_orn = utils.tfpy.print_val(est_orn, "est_orn_delta_object")
            if const.IS_PREDICT_CONTACT:
                contact_loss = utils.tfpy.print_val(contact_loss, "contact_loss")
                contact_acc = utils.tfpy.print_val(contact_acc, "contact_acc")

            self.extra_out["est_orn_delta_object"] = est_orn
        summ.scalar("loss", self.loss_, collections=["scalar", "all"])
        summ.scalar("angle_loss", angle_loss, collections=["scalar", "all"])
        summ.scalar("angle_l1_loss", angle_l1_loss, collections=["scalar", "all"])
        summ.scalar("pos_loss", pos_loss, collections=["scalar", "all"])
        summ.scalar("pos_eud_loss", pos_eud_loss, collections=["scalar", "all"])
        summ.scalar("vel_loss", vel_loss, collections=["scalar", "all"])
        summ.scalar("pos_object_loss", pos_object_loss, collections=["scalar", "all"])
        summ.scalar("pos_agent_loss", pos_agent_loss, collections=["scalar", "all"])
        summ.scalar("orn_object_loss", orn_object_loss, collections=["scalar", "all"])
        summ.scalar("orn_agent_loss", orn_agent_loss, collections=["scalar", "all"])

        summ.scalar("pos_object_eud_loss", pos_object_eud_loss, collections=["scalar", "all"])
        summ.scalar("pos_agent_eud_oss", pos_agent_eud_loss, collections=["scalar", "all"])
        summ.scalar("orn_object_angle_loss", orn_object_angle_loss, collections=["scalar", "all"])
        summ.scalar("orn_agent_angle_loss", orn_agent_angle_loss, collections=["scalar", "all"])

        if const.IS_PREDICT_CONTACT:
            summ.scalar("contact_loss", contact_loss, collections=["scalar", "all"])
            summ.scalar("contact_acc", contact_acc, collections=["scalar", "all"])

    def build_vis(self):
        if const.mode == "test" and const.IS_DUMP_VIS:
            out_size = const.VIS_OUT_SIZE
            self.vis_input_top_gt = tf.stack([tf.reshape(image, [const.BS, self.nobjs, out_size, out_size, 3])[:, 0, :, :, :]\
                                              for image in self.vis_input_top_gt], 1)
            self.vis_output_top_gt = tf.stack([tf.reshape(image, [const.BS, self.nobjs, out_size, out_size, 3])[:, 0, :, :, :]\
                                              for image in self.vis_output_top_gt], 1)
            self.vis_output_top_pred = tf.stack([tf.reshape(image, [const.BS, self.nobjs, out_size, out_size, 3])[:, 0, :, :, :]\
                                              for image in self.vis_output_top_pred], 1)
            self.vis_output_top_diff = tf.stack([tf.reshape(image, [const.BS, self.nobjs, out_size, out_size, 3])[:, 0, :, :, :]\
                                              for image in self.vis_output_top_diff], 1)

            self.vis = {
                "rollout_images": tf.stack(self.rollout_images_vis, 1),
                "front_images": self.vis_front_images,
                "input_images": self.vis_input_images,
                "input_top_gt": self.vis_input_top_gt,
                "output_top_gt": self.vis_output_top_gt,
                "output_top_pred": self.vis_output_top_pred,
                "output_top_diff": self.vis_output_top_diff
            }
        else:
            self.vis = {"pred_views":self.predicted_view,"query_views":self.inputs.state.vp_frame}
            # self.vis = {}
    def build_evaluator(self):
        if const.mode == "test":
            self.evaluator = {
                "pos_object_loss": self.eval_pos_object_loss,
                "orn_object_loss": self.eval_orn_object_loss,
                "pos_object_eud_loss": self.eval_pos_object_eud_loss,
                "orn_object_angle_loss": self.eval_orn_object_angle_loss,
            }
        else:
            self.evaluator = Munch()

    def update_dyn_state_with_delta(self, state, delta_state, mode="xyz13"):
        """
        state: batch_size x T x nobjs x 13
        delta_state: batch_size x T x nobjs x 13
        """
        if mode == "xyz13":
            updated_pos = state[...,:3] + delta_state[...,:3]
            updated_vel = state[...,7:13] + delta_state[...,7:13]
            updated_quat = tfq.Quaternion(delta_state[...,3:7]) * tfq.Quaternion(state[...,3:7])
            return tf.concat([updated_pos, updated_quat, updated_vel], -1)
        elif mode == "orn":
            updated_quat = tfq.Quaternion(delta_state) * tfq.Quaternion(state)
            return updated_quat
        else:
            raise Exception(f"data format is not supported: {mode}")

    def reset_orn(self, states):
        """
        states: ... x 13
        """
        xyz = states[..., :3]
        vel = states[..., 7:13]
        states_shape = states.get_shape().as_list()
        states_shape_dims = len(states.get_shape())
        orn_reset_base = self.orn_reset_base
        for i in range(states_shape_dims - 1):
            orn_reset_base = tf.expand_dims(orn_reset_base, 0)

        states_shape_tmp = copy.deepcopy(states_shape)
        states_shape_tmp[-1] = 1
        orn_reset = tf.tile(orn_reset_base, states_shape_tmp)
        updated_states = tf.concat([xyz, orn_reset, vel], -1)
        return updated_states





    def multiply_orn_with_origin(self, predict_out, gt_out):
        """
        if len(predict_out.get_shape()) == 4:

            debug_out_orn = tfq.Quaternion(predict_out[:,:,:,3:7]) * tfq.Quaternion(gt_out[:,:,:,3:7])
            return tf.concat([predict_out[:,:,:, :3], debug_out_orn, predict_out[:,:,:, 7:]], 3)
        else:
            debug_out_orn = tfq.Quaternion(predict_out[:,:,3:7]) * tfq.Quaternion(gt_out[:,:,3:7])
            return tf.concat([predict_out[:,:,:3], debug_out_orn, predict_out[:,:, 7:]], 2)
        """
        debug_out_orn = tfq.Quaternion(predict_out[...,3:7]) * tfq.Quaternion(gt_out[...,3:7])
        return tf.concat([predict_out[...,:3], debug_out_orn, predict_out[..., 7:]], -1)



    ## for visulization
    def get_rgb_images_for_summ(self, images, out_size, nobjs):
        resize_images = tf.image.resize_images(images[:,0,:,:,:], [out_size, out_size])
        resize_images = tf.reshape(tf.tile(tf.expand_dims(resize_images, 1), [1, nobjs, 1, 1, 1]),\
                        [-1, out_size, out_size, 3])

        resize_images2 = tf.image.resize_images(images[:,1,:,:,:], [out_size, out_size])
        resize_images2 = tf.reshape(tf.tile(tf.expand_dims(resize_images, 1), [1, nobjs, 1, 1, 1]),\
                        [-1, out_size, out_size, 3])
        return resize_images, resize_images

    def get_image_given_object_states(self, object_state, out_size):
        """
        object_state: batch_size x nobj x 13
        object_size: for resizing objects in voxel
        return: [batch_size x nobj] x out_size x out_size x 3
        """
        voxel = self.inputs.state.voxels
        object_size = self.inputs.state.resize_factor

        object_in_3D = self.draw_object_in_3dtensor(object_state, object_size, voxel, out_size)

        object_in_3D_image_xz, object_in_3D_image_xy = vis_op.top_side_images_from_voxel(object_in_3D)

        return object_in_3D_image_xz, object_in_3D_image_xy

    def draw_object_in_3dtensor(self, object_state, object_size, voxel, voxel_size, object_class=None, output_format="bs_nobjs", with_greedy=False, debug=False, debug_msg=""):
        """
        given unnormalized xyz location and orientation, this function put the object to
        the scene with the specified object size
        object_state: batch_size x nobjs x 7 (quaterion)
        object_size: batch_size x nobjs x 3 (size)
        voxels: batch_size x nobjs x Sd x Sh x Sw
        """
        extra_out = dict()
        bs, nobjs, Sd, Sh, Sw = voxel.shape
        # convert from world coordiante to tensorflow coordinate [-1.5, 1.5] -> [0, 1]
        rois_center = (object_state[:,:,:3] * (1/const.boundary_to_center) + 1) * 0.5
        hwd = object_size * (1 / (2*const.boundary_to_center))
        if debug_msg == "shift":
            shift = tf.constant([[[0.4, 0, 0]]], dtype=tf.float32)
            rois = tf.stack([rois_center-hwd + shift, rois_center + hwd+shift], 3)

        else:
            rois = tf.stack([rois_center-hwd, rois_center + hwd], 3)
        rois = tf.reshape(rois, [-1, 3, 2])

        class_labels = tf.ones(tf.shape(rois)[0])

        obj_quat = object_state[:, :, 3:7]
        obj_rotmat = tfq.Quaternion(obj_quat).as_rotation_matrix()

        displacement = tf.constant(np.zeros((bs, nobjs, 3, 1), dtype=np.float32), dtype=tf.float32)
        bottom_row = np.zeros((bs, nobjs, 1, 4), dtype=np.float32)
        bottom_row[:,:,0,3] = 1.0
        bottom_row = tf.constant(bottom_row)
        pad_matrix = tf.concat([
            tf.concat([obj_rotmat, -displacement], axis = 3),
            bottom_row
        ], axis = 2)
        pad_matrix = tf.reshape(pad_matrix, [bs * nobjs, 4, 4])
        mask = tf.reshape(voxel, [bs * nobjs, Sd, Sh, Sw, 1])
        rotated_mask = utils.voxel.rotate_voxel2(mask, pad_matrix)
        extra_out["rotated_mask"] = tf.reshape(rotated_mask, [bs, nobjs, Sd, Sh, Sw])
        f = lambda x: utils.voxel.crop_mask_to_full_mask_precise(*x)
        pad_out, roi  = tf.map_fn(f,\
            [rois, class_labels, rotated_mask, voxel_size * tf.ones((tf.shape(rois)[0]), dtype=tf.int32)],\
            dtype = (tf.float32, tf.int32))

        pad_out = tf.reshape(pad_out, [tf.shape(rois)[0], voxel_size, voxel_size, voxel_size, 1])
        if object_class is not None:
            object_class = tf.reshape(object_class, [-1, 1, 1, 1, 1])
            pad_out = pad_out * object_class
        #pad_out = utils.voxel.resize_voxel(pad_out, 0.5)
        extra_out["reshape_pad_out"] = tf.reshape(pad_out, [bs, nobjs, voxel_size, voxel_size, voxel_size, 1])


        if with_greedy:
            f = lambda x: utils.voxel.crop_mask_to_full_mask_precise(*x, greedy=True)
            pad_out_greedy, roi_greedy  = tf.map_fn(f,\
                [rois, class_labels, rotated_mask, voxel_size * tf.ones((tf.shape(rois)[0]), dtype=tf.int32)],\
                dtype = (tf.float32, tf.int32))

            if object_class is not None:
                object_class = tf.reshape(object_class, [-1, 1, 1, 1, 1])
                pad_out = pad_out_greedy * object_class
            extra_out["greedy_pad"] = pad_out_greedy
            extra_out["reshape_greedy_pad"] = tf.reshape(pad_out_greedy, [bs, nobjs, voxel_size, voxel_size, voxel_size, 1])

        #if const.DEBUG_UNPROJECT:
        #    from utils.vis_np import save_voxel
        #    for batch_id in range(2):
        #        save_voxel(pad_out[batch_id*2, :, :, :, 0], f"dump/pad_" + debug_msg + f"b{batch_id}.binvox")
        if debug:
            return pad_out, extra_out
        else:
            return pad_out

    def make_dot_figure(self, point, image_template, nobjs, sigma=const.sigma):
        image_top, image_side = vis_op.make_dot_figure(point, image_template, nobjs, sigma=sigma, \
                               boundary_to_center=const.boundary_to_center)
        return image_top, image_side

    def compute_contact(self):
        norm = tf.linalg.norm(self.target.obj_state[...,:3] - self.inputs.state.obj_state[..., :3], axis=-1)
        norm_angle = tf.linalg.norm(self.target.obj_state[...,3:7] - self.inputs.state.obj_state[..., 3:7], axis=-1)

        is_contact = tf.dtypes.cast(tf.math.greater(norm + norm_angle, 0.0001), tf.float32)
        return is_contact

    def debug_output(self):
        out_size = 64
        self.input__ = self.inputs.state.obj_state[:,0,:,:]
        gt_in_object_image_xz, gt_object_image_xy = self.get_image_given_object_states(self.inputs.state.obj_state[:,0,:,:], out_size)
        self.output__ = gt_in_object_image_xz
        gt_out_object_image_xz, gt_object_image_xy = self.get_image_given_object_states(self.target.obj_state[:,0,:,:], out_size)
        import scipy.misc
        diff_image = vis_tf.create_diff_images(gt_in_object_image_xz, gt_out_object_image_xz)

        robot_hand_image_out_xz, robot_hand_image_xy = self.make_dot_figure(self.target.agent_state[:, 0,0,:3], diff_image, self.nobjs)
        robot_hand_image_in_xz, robot_hand_image_xy = self.make_dot_figure(self.inputs.state.agent_state[:, 0,0,:3], diff_image, self.nobjs)

        zero_pad= tf.zeros((64, 64, 1), dtype=tf.float32)
        for batch_id in range(const.BS):
            for t in range(2):
                scipy.misc.imsave(f"dump/b{batch_id}_t{t}.png", self.inputs.state.frames[batch_id, t,0,:,:,2:])
            #scipy.misc.imsave(f"dump/b{batch_id}_in_xz.png", gt_in_object_image_xz[batch_id*2, ...])
            #scipy.misc.imsave(f"dump/b{batch_id}_out_xz.png", gt_out_object_image_xz[batch_id*2, ...])
            scipy.misc.imsave(f"dump/b{batch_id}_diff.png", diff_image[batch_id*2, ...])
            scipy.misc.imsave(f"dump/b{batch_id}_diff_agent.png", gt_in_object_image_xz[batch_id*2,...] + tf.concat([robot_hand_image_out_xz[batch_id*2, ...], robot_hand_image_in_xz[batch_id*2, ...],zero_pad], 2) )


    def rollout_summary(self, basic_info, est_states, est_states_rollout, start_t, T=5):
        """
        est_states: batch_size x max_T x (nobjs + njoints) x 13
        est_states_rollout: batch_size x T X (nobjs + njoints) x 13
        """
        bs, _, _, _ = est_states.get_shape()
        object_size = basic_info["object_size"] #self.inputs.state.resize_factor
        voxel = basic_info["voxel"] #self.inputs.state.voxels
        voxel_size = int(const.S)
        out_size = const.VIS_OUT_SIZE #int(0.5 * voxel_size)
        rollout_images = []

        # first image
        self.vis_front_images = self.target.images_front[:, start_t:start_t + T + 1, :, :, :, :]
        self.vis_input_images = self.inputs.state.frames[:, start_t, :, :, :, 2:5]
        self.vis_input_top_gt = []
        self.vis_output_top_gt = []
        self.vis_output_top_pred = []
        self.vis_output_top_diff = []


        object_state = self.inputs.state.obj_state[:, start_t, :, :]
        action_state = self.inputs.action.actions
        agent_state_t = self.inputs.state.agent_state[:, start_t, 0, 0:3]

        _, nobjs, _ = object_state.get_shape()
        gt_object_image_xz, gt_object_image_xy = self.get_image_given_object_states(object_state, out_size)
        robot_hand_image_xz, robot_hand_image_xy = self.make_dot_figure(agent_state_t, gt_object_image_xz, nobjs)
        zero_pad = tf.zeros_like(robot_hand_image_xz)
        zero_pad3 = tf.tile(zero_pad, [1,1,1,3])

        hand_color = tf.reshape(tf.constant([0, 1, 0], dtype=tf.float32), [1,1,1,3])
        pred_hand_color = tf.reshape(tf.constant([1, 0, 1], dtype=tf.float32), [1,1,1,3])
        action_color = tf.reshape(tf.constant([1, 0, 0], dtype=tf.float32), [1,1,1,3])
        gt_object_image_xz = gt_object_image_xz + hand_color * robot_hand_image_xz
        gt_object_image_xy = gt_object_image_xy + hand_color * robot_hand_image_xy

        #resize_images, resize_images2 = self.get_rgb_images_for_summ(self.inputs.state.frames[:, start_t, :, :, :, 2:5], out_size, nobjs)
        resize_images, resize_images2 = self.get_rgb_images_for_summ(self.target.images_front[:, start_t, :, :, :, :] + 0.5, out_size, nobjs)

        # show input image and input state
        final_image = vis_op.merge_image([[resize_images,  gt_object_image_xz, zero_pad3, zero_pad3 , zero_pad3, zero_pad3],
                                        [resize_images2, gt_object_image_xy, zero_pad3, zero_pad3]], bs, nobjs)

        rollout_images.append(final_image[:, 0, :, :, :])

        # copy paste intial state to prediction block
        final_image = vis_op.merge_image([[resize_images,  gt_object_image_xz, gt_object_image_xz, zero_pad3, gt_object_image_xz, zero_pad3],
                                        [resize_images2, gt_object_image_xy, gt_object_image_xy, gt_object_image_xy]], bs, nobjs)
        rollout_images.append(final_image[:, 0, :, :, :])
        previous_agent_state = agent_state_t
        previous_gt_object_image_xz = gt_object_image_xz
        previous_gt_object_image_xy = gt_object_image_xy
        previous_images1 = resize_images
        previous_images2 = resize_images2

        for t in range(T): #T.value):
            # get action image first
            action_image_xz, action_image_xy = self.make_dot_figure(\
                previous_agent_state + 2.0 * tf.stack([(-1) * action_state[:, start_t + t, 1],\
                tf.zeros_like(action_state[:, start_t, 1]), action_state[:, start_t + t, 0]], 1),\
                previous_gt_object_image_xz, nobjs, const.sigma)
            gt_object_image_xz_with_action = previous_gt_object_image_xz + action_color * action_image_xz
            gt_object_image_xy_with_action = previous_gt_object_image_xy + action_color * action_image_xy
            if t == 0:
                image3_1 = gt_object_image_xz_with_action
                image3_2 = gt_object_image_xy_with_action
                image4_1 = gt_object_image_xz_with_action
                image4_2 = gt_object_image_xy_with_action
                diff3_1 = zero_pad3
                diff4_1 = zero_pad3

                self.vis_input_top_gt = []
            if t > 0:
                image3_1 = gt_object_image_xz_with_action
                image3_2 = gt_object_image_xy_with_action
                image4_1 = previous_rollout_pred_object_image_xz + action_color * action_image_xz
                image4_2 = previous_rollout_pred_object_image_xy + action_color * action_image_xy

                diff3_1 = zero_pad3
                diff4_1 = vis_tf.create_diff_images(previous_gt_object_image_xz, previous_rollout_pred_object_image_xz)

            self.vis_input_top_gt.append(image3_1)

            final_image = vis_op.merge_image([[previous_images1,  gt_object_image_xz_with_action, image3_1, diff3_1, image4_1, diff4_1],
                                            [previous_images2, gt_object_image_xy_with_action, image3_2, image4_2]], bs, nobjs)

            rollout_images.append(final_image[:, 0, :, :, :])

            # get prediction images
            pred_object_state = est_states[:, start_t + t, :nobjs.value, :]
            rollout_pred_object_state = est_states_rollout[:, t, :nobjs.value,:]
            agent_state_t = self.target.agent_state[:, start_t + t, 0, 0:3]
            object_state = self.target.obj_state[:, start_t + t, :, :]

            pred_agent_state_t = est_states[:, start_t + t, nobjs.value, 0:3]
            rollout_pred_agent_state_t = est_states_rollout[:, t, nobjs.value, 0:3]

            #self.update_with_delta()

            # batch_size x S x S x Si
            bs, nobjs, _ = object_size.get_shape()
            gt_object_image_xz, gt_object_image_xy = self.get_image_given_object_states(object_state, out_size)

            robot_hand_image_xz, robot_hand_image_xy = self.make_dot_figure(agent_state_t, gt_object_image_xz, nobjs)

            # making a gaussian map

            gt_object_image_xz_tmp = gt_object_image_xz
            gt_object_image_xz = gt_object_image_xz + hand_color * robot_hand_image_xz
            gt_object_image_xy = gt_object_image_xy + hand_color * robot_hand_image_xy

            pred_object_image_xz, pred_object_image_xy = self.get_image_given_object_states(pred_object_state, out_size)
            pred_robot_hand_image_xz, pred_robot_hand_image_xy = self.make_dot_figure(pred_agent_state_t, gt_object_image_xz, nobjs)
            diff_image_xz = vis_tf.create_diff_images(gt_object_image_xz_tmp, pred_object_image_xz)

            pred_object_image_xz = vis_op.image_sum_by_overlap(pred_object_image_xz +\
                                   hand_color * robot_hand_image_xz,\
                                   pred_hand_color * pred_robot_hand_image_xz)
            pred_object_image_xy = vis_op.image_sum_by_overlap(pred_object_image_xy +\
                                   hand_color * robot_hand_image_xy,\
                                   pred_hand_color * pred_robot_hand_image_xy)

            rollout_pred_object_image_xz, rollout_pred_object_image_xy = self.get_image_given_object_states(rollout_pred_object_state, out_size)
            rollout_pred_robot_hand_image_xz, rollout_pred_robot_hand_image_xy = self.make_dot_figure(rollout_pred_agent_state_t, gt_object_image_xz, nobjs)
            diff_rollout_image_xz = vis_tf.create_diff_images(gt_object_image_xz_tmp, rollout_pred_object_image_xz)
            rollout_pred_object_image_xz = vis_op.image_sum_by_overlap(rollout_pred_object_image_xz +\
                                           hand_color * robot_hand_image_xz, \
                                           pred_hand_color * rollout_pred_robot_hand_image_xz)
            rollout_pred_object_image_xy = vis_op.image_sum_by_overlap(rollout_pred_object_image_xy +\
                                           hand_color * robot_hand_image_xy,\
                                           pred_hand_color * rollout_pred_robot_hand_image_xy)

            #resize_images, resize_images2 = self.get_rgb_images_for_summ(self.inputs.state.frames[:, start_t + t + 1, :, :, :, 2:5], out_size, nobjs)

            resize_images, resize_images2 = self.get_rgb_images_for_summ(\
                self.target.images_front[:, start_t + t + 1, :, :, :, :] + 0.5, out_size, nobjs)
            #diff_image_xz = vis_tf.create_diff_images(gt_object_image_xz, pred_object_image_xz)

            self.vis_output_top_gt.append(gt_object_image_xz)
            self.vis_output_top_pred.append(pred_object_image_xz)
            self.vis_output_top_diff.append(diff_image_xz)
            final_image = vis_op.merge_image([[resize_images,  gt_object_image_xz, pred_object_image_xz, diff_image_xz,
                                             rollout_pred_object_image_xz, diff_rollout_image_xz],
                                            [resize_images2, gt_object_image_xy, pred_object_image_xy,
                                             rollout_pred_object_image_xy]], bs, nobjs)

            #final_image = tf.concat([final_image1, final_image2], 2)
            #final_image = tf.reshape(final_image, [bs, nobjs, out_size, out_size * 8, 3])
            rollout_images.append(final_image[:, 0, :, :, :])
            previous_agent_state = agent_state_t
            previous_gt_object_image_xz = gt_object_image_xz
            previous_gt_object_image_xy = gt_object_image_xy
            previous_pred_object_image_xz = pred_object_image_xz
            previous_pred_object_image_xy = pred_object_image_xy
            previous_rollout_pred_object_image_xz = rollout_pred_object_image_xz
            previous_rollout_pred_object_image_xy = rollout_pred_object_image_xy

            previous_images1 = resize_images
            previous_images2 = resize_images2
        self.rollout_images_vis = rollout_images