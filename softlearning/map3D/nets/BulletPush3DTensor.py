#from nets.Net import Net
from .BulletPushBase import BulletPushBase
import constants as const
const.set_experiment("rl_new")

import utils_map as utils
from utils_map import voxel
import tensorflow as tf
import numpy as np
from munch import Munch
from tensorflow import summary as summ
#from .modules.GraphNet import GraphNet
from .modules.ConvfcNet import ConvfcNet
from .modules import grnn_op
from .modules import vis_op
from .modules import bulletpush3DTensor_utils as class_utils
import copy
import tfquaternion as tfq
import ipdb
st = ipdb.set_trace
import __init__path
"""
* agent state prediction is separated from object prediction

"""



class BulletPush3DTensor4_cotrain(BulletPushBase):



    def loss(self):
        if const.loss=="l1":
            vp_loss = utils.losses.l1loss(self.predicted_view, self.inputs.state.vp_frame)            
        else:
            vp_loss = utils.losses.binary_ce_loss(self.predicted_view, self.inputs.state.vp_frame)

        self.loss_ = vp_loss
        vp_loss = utils.tfpy.print_val(vp_loss, "vp_loss")
        summ.scalar("viewpred_loss", vp_loss, collections=["scalar", "all"])
        # if const.run_full:
        #     self.gt_delta = tf.concat([self.target.delta_obj_state, self.target.delta_agent_state], 2)[:, :self.T, ...]

        #     gt_xyz_delta, gt_orn_delta, gt_vel_delta = self.split_states_g(self.gt_delta, mode="h13")
        #     est_xyz_delta, est_orn_delta, est_vel_delta = self.split_states_g(self.est_dyn_states_delta, mode="h13")

        #     if const.IS_PREDICT_CONTACT:

        #         contact_logits = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gt_is_contact,\
        #           logits=self.pred_contact)
        #         contact_loss = tf.reduce_sum(contact_logits * self.target.all_class[:,:self.T, :self.nobjs])/tf.reduce_sum(self.target.all_class[:,:self.T, :self.nobjs])

        #         contact_acc = tf.reduce_sum(tf.dtypes.cast(tf.equal(tf.math.greater(self.gt_is_contact, 0.5),\
        #                tf.math.greater(tf.nn.sigmoid(self.pred_contact), 0.5)), dtype=tf.float32)\
        #                * self.target.all_class[:,:self.T, :self.nobjs])\
        #               /tf.reduce_sum(self.target.all_class[:,:self.T, :self.nobjs])


        #     angle_loss = utils.losses.quatloss(est_orn_delta, gt_orn_delta,
        #                                        mask=self.target.all_class[:,:self.T,:])

        #     angle_l1_loss = utils.losses.l1loss(est_orn_delta, gt_orn_delta,
        #                                        mask=tf.expand_dims(self.target.all_class[:,:self.T,:], -1))

        #     pos_loss = utils.losses.l2loss(est_xyz_delta, gt_xyz_delta,
        #                                    mask = tf.expand_dims(self.target.all_class[:,:self.T,:], -1))
        #     vel_loss = utils.losses.l2loss(est_vel_delta, gt_vel_delta,
        #                                    mask = tf.expand_dims(self.target.all_class[:,:self.T,:], -1))

        #     pos_l1_loss = utils.losses.l1loss(est_xyz_delta, gt_xyz_delta,
        #                                    mask = tf.expand_dims(self.target.all_class[:,:self.T,:], -1))
        #     vel_l1_loss = utils.losses.l1loss(est_vel_delta, gt_vel_delta,
        #                                    mask = tf.expand_dims(self.target.all_class[:,:self.T,:], -1))

        #     ### view pred loss ###
        #     if const.train_vp:
        #         vp_loss = utils.losses.binary_ce_loss(self.predicted_view, self.inputs.state.vp_frame)
        #     else:
        #         vp_loss = tf.constant(0.0)
        #     is_trainset_coeff = tf.cond(tf.equal(self.input.q_ph, 0), true_fn=lambda:1.0, false_fn=lambda:0.0)

        #     tf.summary.image('pred_view', self.predicted_view, max_outputs=1, collections=["scalar"])
        #     tf.summary.image('gt_view', self.inputs.state.vp_frame, max_outputs=1, collections=["scalar"])

        #     self.dyn_loss = pos_l1_loss + angle_l1_loss + vel_l1_loss

        #     if const.IS_PREDICT_CONTACT:
        #         #if const.PRETRAIN_CONTACT:
        #         #    self.dyn_loss = contact_loss
        #         #else:
        #         self.dyn_loss += 0.01 * contact_loss

        #     # do not train dynamics in trainv mode
        #     self.real_total_loss = self.dyn_loss + vp_loss
        #     self.loss_ = self.dyn_loss*is_trainset_coeff + vp_loss

        #     pos_eud_loss = utils.losses.eudloss(est_xyz_delta, gt_xyz_delta,\
        #                                         mask = self.target.all_class[:,:self.T,:])
        #     # agent loss
        #     est_xyz_delta_object, est_xyz_delta_agent = self.split_states_g(est_xyz_delta)
        #     est_orn_delta_object, est_orn_delta_agent = self.split_states_g(est_orn_delta)
        #     gt_xyz_delta_object, gt_xyz_delta_agent = self.split_states_g(gt_xyz_delta)
        #     gt_orn_delta_object, gt_orn_delta_agent = self.split_states_g(gt_orn_delta)

        #     tf.summary.histogram("gt_xyz_delta_object", tf.linalg.norm(gt_xyz_delta_object[:,0,0,:], axis=-1),\
        #                          collections=["scalar", "all"])
        #     tf.summary.histogram("gt_xyz_delta_object_l1", tf.reduce_sum(tf.abs(gt_xyz_delta_object[:,0,0,:]), axis=-1),\
        #                          collections=["scalar", "all"])

        #     tf.summary.histogram("est_xyz_delta_object", tf.linalg.norm(est_xyz_delta_object[:,0,0,:], axis=-1),\
        #                          collections=["scalar", "all"])
        #     tf.summary.histogram("est_xyz_delta_object_l1", tf.reduce_sum(tf.abs(est_xyz_delta_object[:,0,0,:]), axis=-1),\
        #                          collections=["scalar", "all"])
        #     tf.summary.histogram("gt_orn_delta_object", tf.linalg.norm(\
        #                           tf.sign(gt_orn_delta_object[:, 0,0,:1]) * gt_orn_delta_object[:,0,0,:] \
        #                           - tf.expand_dims(self.orn_base, 0), axis=-1), collections=["scalar", "all"])
        #     tf.summary.histogram("gt_orn_delta_object_l1", tf.reduce_sum(tf.abs(\
        #                           tf.sign(gt_orn_delta_object[:, 0, 0, :1]) * gt_orn_delta_object[:,0,0,:] \
        #                           - tf.expand_dims(self.orn_base, 0)), axis=-1), collections=["scalar", "all"])

        #     mask_object, mask_agent = self.split_states_g(tf.expand_dims(self.target.all_class[:,:self.T,:], -1))

        #     pos_object_loss = utils.losses.l2loss(est_xyz_delta_object, gt_xyz_delta_object,
        #                                         mask=mask_object)

        #     pos_object_eud_loss = utils.losses.eudloss(est_xyz_delta_object, gt_xyz_delta_object,\
        #                                                 mask = mask_object[...,0])


        #     pos_agent_loss = utils.losses.l2loss(est_xyz_delta_agent, gt_xyz_delta_agent,
        #                                         mask=mask_agent)

            

        #     orn_object_loss = utils.losses.quatloss(est_orn_delta_object, gt_orn_delta_object,
        #                                         mask=mask_object[...,0])

        #     orn_object_angle_loss = utils.losses.quat_angle_loss(est_orn_delta_object, gt_orn_delta_object, mask=mask_object[...,0])


        #     orn_agent_loss = utils.losses.quatloss(est_orn_delta_agent, gt_orn_delta_agent,
        #                                         mask=mask_agent[...,0])

        #     # object_loss

        #     self.loss_ = utils.tfpy.print_val(self.loss_, "loss")
        #     self.real_total_loss = utils.tfpy.print_val(self.real_total_loss, "real_total_loss")
        #     if const.DEBUG_LOSSES:
        #         vp_loss = utils.tfpy.print_val(vp_loss, "vp_loss")
        #         angle_loss = utils.tfpy.print_val(angle_loss, "angle_loss")
        #         angle_l1_loss = utils.tfpy.print_val(angle_loss, "angle_l1_loss")
        #         pos_eud_loss = utils.tfpy.print_val(pos_eud_loss, "pos_eud_loss")
        #         pos_loss = utils.tfpy.print_val(pos_loss, "pos_loss")
        #         vel_loss = utils.tfpy.print_val(vel_loss, "vel_loss")

        #         pos_object_loss = utils.tfpy.print_val(pos_object_loss, "pos_object_loss")
        #         pos_agent_loss = utils.tfpy.print_val(pos_agent_loss, "pos_agent_loss")
        #         orn_object_loss = utils.tfpy.print_val(orn_object_loss, "orn_object_loss")
        #         orn_agent_loss = utils.tfpy.print_val(orn_agent_loss, "orn_agent_loss")

        #         est_orn  = tf.concat([est_orn_delta_object[:,:,0,:], gt_orn_delta_object[:,:,0,:]], -1)
        #         est_orn = utils.tfpy.print_val(est_orn, "est_orn_delta_object")
        #         if const.IS_PREDICT_CONTACT:
        #             contact_loss = utils.tfpy.print_val(contact_loss, "contact_loss")
        #             contact_acc = utils.tfpy.print_val(contact_acc, "contact_acc")

        #         self.extra_out["est_orn_delta_object"] = est_orn
        #     summ.scalar("loss", self.loss_, collections=["scalar", "all"])
        #     summ.scalar("real_total_loss", self.real_total_loss, collections=["scalar", "all"])
        #     summ.scalar("angle_loss", angle_loss, collections=["scalar", "all"])
        #     summ.scalar("angle_l1_loss", angle_l1_loss, collections=["scalar", "all"])
        #     summ.scalar("pos_loss", pos_loss, collections=["scalar", "all"])
        #     summ.scalar("pos_eud_loss", pos_eud_loss, collections=["scalar", "all"])
        #     summ.scalar("vel_loss", vel_loss, collections=["scalar", "all"])
        #     summ.scalar("pos_object_loss", pos_object_loss, collections=["scalar", "all"])
        #     summ.scalar("pos_agent_loss", pos_agent_loss, collections=["scalar", "all"])
        #     summ.scalar("orn_object_loss", orn_object_loss, collections=["scalar", "all"])
        #     summ.scalar("orn_agent_loss", orn_agent_loss, collections=["scalar", "all"])

        #     summ.scalar("pos_object_eud_loss", pos_object_eud_loss, collections=["scalar", "all"])
        #     summ.scalar("orn_object_angle_loss", orn_object_angle_loss, collections=["scalar", "all"])

        #     if const.IS_PREDICT_CONTACT:
        #         summ.scalar("contact_loss", contact_loss, collections=["scalar", "all"])
        #         summ.scalar("contact_acc", contact_acc, collections=["scalar", "all"])

    def set_batchSize(self,bs):
        const.BS = bs
    def setup_data(self, data):
        # st()
        self.__dict__.update(data)

        # create ground truth voxel at the first step

        self.T = const.max_T +1
        # self.crop_size=const.crop_size
        self.bs = const.BS
        # calculate the rois
        # batch_size x nrois x 3, from [-1.5, 1.5] -> [0,1]
        # self.input_images = self.inputs.state.frames[:, 0, :, :, :, :]
        if const.run_full:
            self.rollout_T = np.minimum(self.T, 5)
            self.nobjs = 2
            self.njoints = 1
            self.object_static_dim = 4

            self.orn_base = tf.constant([1, 0, 0, 0], dtype=tf.float32)
            voxel_size = int(const.S)
            self.object_first_state = self.inputs.state.obj_state[:, 0, :, :]

            object_state = self.inputs.state.obj_state[:, 0, :, :]
            object_size = self.inputs.state.resize_factor
            self.gt_object_size = object_size
            self.object_state = object_state
            voxel = self.inputs.state.voxels
            if const.IS_PREDICT_CONTACT:
                self.gt_is_contact = self.compute_contact()

            #self.debug_output()
            #import ipdb; ipdb.set_trace()
            """
            this is not working
            import imageio
            #import ipdb; ipdb.set_trace()
            for batch_id in range(2):
                imageio.mimsave(f"dump/input_b{batch_id}.gif",\
                   [img.numpy() for img in tf.unstack(self.inputs.state.frames[batch_id, :, 0, :, :, 2:], axis=0)], 'GIF', duration=5)
            self.inputs.state.frames
            """
            self.pad_out, extra_out = self.draw_object_in_3dtensor(object_state, object_size, voxel, voxel_size, debug=True)

            if const.MASK_AGENT:
                hand_mask = np.zeros((64, 64, 64, 1))
                hand_mask[:, 10:, :, :] = 1
                hand_mask[26:38, :10, 26:38] = 1
                hand_mask = tf.constant(hand_mask, dtype=tf.float32)
                hand_mask = tf.tile(tf.expand_dims(hand_mask, 0), [const.BS * self.T, 1, 1, 1, 1])

                hand_class_labels = tf.ones(tf.shape(hand_mask)[0], dtype=tf.float32)
                hand_size = 32 * tf.ones(tf.shape(hand_mask)[0], dtype=tf.int32)

                agent_loc = tf.reshape(self.inputs.state.agent_state[:,:,:,:3], [const.BS * self.T, 3])
                ones = tf.ones_like(agent_loc[..., 0], dtype=tf.float32)
                agent_center = ((tf.stack([agent_loc[..., 0], 0.5*ones, \
                           agent_loc[..., 2]], -1)/const.boundary_to_center) + 1.0) * 0.5

                agent_hwd = tf.stack([ones * 0.35, ones, ones * 0.35], -1)/(2 * const.boundary_to_center)
                agent_rois = tf.stack([agent_center - agent_hwd, agent_center + agent_hwd], -1)
                #agent should be [z, 1, x], hwd = [d, 1, w]

                f = lambda x: utils.voxel.crop_mask_to_full_mask_precise(*x)

                agent_mask, agent_rois = tf.map_fn(f, [agent_rois, hand_class_labels, hand_mask, hand_size], dtype=(tf.float32, tf.int32))
                self.agent_mask = agent_mask
            if const.DEBUG_UNPROJECT: 
                from utils.vis_np import save_voxel
                for t in range(self.T):
                    pad_out, _ = self.draw_object_in_3dtensor(self.inputs.state.obj_state[:, t, :, :],\
                        object_size, voxel, voxel_size, debug=True)
            
                    for batch_id in range(const.BS):
                        save_voxel(pad_out[batch_id * self.nobjs, :, :, :, 0], f"dump/pad_out_t{t}_b{batch_id}.binvox")
                        #save_voxel(agent_mask[batch_id * self.T + t, :, :, :, 0], f"dump/agent_mask_t{t}_b{batch_id}.binvox")
            self.gt_mask = self.pad_out
 
    def normalize_quat(self, input):
        xyz, orn, vel = self.split_states(input, "h13")
        orn_n_dims = len(orn.get_shape())
        orn_base = tf.reshape(self.orn_base, [1] * (orn_n_dims - 1) + [4])

        orn = tf.nn.l2_normalize(orn + orn_base, -1)
        output = tf.concat([xyz, orn, vel], -1)
        return output


    class Dynamics:
        def __init__(self, BP, states, node_predict_dim=0, global_input=None):
            self.BP = BP
            object_states, agent_states = BP.split_states(states)
           
            #self.convfcnet = []
            #for object_id in range(self.BP.nobjs):
            object_id = 0
            convfcnet = ConvfcNet(object_states[:, object_id, :],\
                        predict_dim=node_predict_dim["object"],\
                        layers=[32, 32, 32, 32], scopename=f"object_conv_{object_id}")
            self.convfcnet = convfcnet
            if const.IS_PREDICT_CONTACT:
                self.convfcnet_contact = ConvfcNet(object_states[:, object_id, :],\
                        predict_dim=1,\
                        layers=[32, 32], scopename=f"object_contatc_conv_{object_id}", is_normalize=True)
                

            #self.convfcnet.append(convfcnet)
            #_, _, agent_xyzorn = BP.split_states(agent_states, 'h')
            self.convfcnet_agent = []
            for agent_id in range(self.BP.njoints):
                convfcnet_agent = ConvfcNet(agent_states, \
                            predict_dim=node_predict_dim["agent"], \
                            layers=[32, 32], scopename=f"agent_conv_{agent_id}")
                self.convfcnet_agent.append(convfcnet_agent)
        def predict_one_step(self, node_input, global_input= None, gt_contact=None, reuse=True):
            object_state, agent_state = self.BP.split_states(node_input)
            #_, _, agent_xyzorn = self.BP.split_states(agent_state, 'h')
           
            objects_delta = []
            if const.IS_PREDICT_CONTACT:
                objects_contact = []
            obj_reuse = reuse
            for object_id in range(self.BP.nobjs):
                if object_id > 0:
                    obj_reuse=True
                obj_delta = self.convfcnet.predict_one_step(object_state[:, object_id, :], \
                    global_input=None, reuse=obj_reuse)
                objects_delta.append(obj_delta)
                if const.IS_PREDICT_CONTACT:
                    is_contact = self.convfcnet_contact.predict_one_step(object_state[:, object_id,:], global_input=None, reuse=obj_reuse)
                    #is_contact = tf.nn.sigmoid(is_contact)
                    objects_contact.append(is_contact)

            agents_delta = []
            for joint_id in range(self.BP.njoints): 
                agent_delta = self.convfcnet_agent[joint_id].predict_one_step(agent_state[:, joint_id, :], global_input=None if const.AGENT_WITHOUT_GLOBAL else global_input, reuse=reuse)
                agents_delta.append(agent_delta)

            object_delta = tf.stack(objects_delta, 1)
            agent_delta = tf.stack(agents_delta, 1)

            delta = tf.concat([object_delta, agent_delta], 1)
            delta = self.BP.normalize_quat(delta)
            if const.IS_PREDICT_CONTACT:
                # pad agent contact with zero
                objects_contact_score = tf.stack([tf.nn.sigmoid(x) for x in objects_contact] + [tf.ones_like(objects_contact[0])] * self.BP.njoints, 1)
                if const.PRETRAIN_CONTACT and const.mode=="train":
                    # if it is training, use gt contact as mask
                    objects_contact_score = tf.concat([tf.expand_dims(gt_contact, -1)] +\
                        [tf.ones_like(tf.expand_dims(objects_contact[0], 1))] * self.BP.njoints, 1)
                objects_contact_bin = tf.math.greater(objects_contact_score, 0.5)

                pose_base = tf.concat([tf.zeros((3), dtype=tf.float32), self.BP.orn_base, tf.zeros((6), dtype=tf.float32)], 0)
                pose_n_dims = len(delta.get_shape())
                bs, nobjs, _ = delta.get_shape()
                pose_base = tf.tile(tf.reshape(pose_base, [1] * (pose_n_dims - 1)\
                                + [13]), [bs, nobjs, 1])
                filtered_delta = tf.where(tf.tile(objects_contact_bin, [1,1,13]),
                    delta, pose_base)

                return filtered_delta, tf.stack(objects_contact, 1)[...,0]
            return delta


    def get_inputs2Ddec_gqn3d(self, inputs):
        aligned_inputs = self.align_to_query_gqn3d(inputs) #4 scales
        projected_inputs = grnn_op.project_inputs(aligned_inputs)

        with tf.variable_scope('depthchannel_net'):
            return [utils.nets.depth_channel_net_v2(feat)
                    for feat in projected_inputs]


    def align_to_query_gqn3d(self, features):
        return [self.align_to_query_single_gqn3d(feature) for feature in features]
    
    def align_to_query_single_gqn3d(self, feature):
        #a single feature from view 0
        dthetas = [self.inputs.state.vp_theta - self.base_theta]
        dphis = [self.inputs.state.vp_phi - self.base_phi]
        return self.translate_multiple2(dthetas, dphis, [feature])[0]

    def translate_multiple2(self, dthetas, dphis, voxs):
        dthetas = tf.stack(dthetas, axis = 0)
        dphis = tf.stack(dphis, 0)
        voxs = tf.stack(voxs, 0)

        f = lambda x: utils.voxel.translate_given_angles2(*x)
        out = tf.map_fn(f, [dthetas, dphis, voxs], dtype = tf.float32)
        return tf.unstack(out, axis = 0)

    def get_outputs2Ddec_gqn3d(self, inputs):
        with tf.variable_scope('gqn3d_2Ddecoder'):
            return self.convlstm_decoder(inputs)

    def convlstm_decoder(self, inputs):
        #we get feature maps of different resolution as input
        #downscale last and concat with second last


        inputs = [utils.tfutil.poolorunpool(x, 16) for x in inputs]
        net = tf.concat(inputs, axis = -1)
        #net = slim.conv2d(net, 256, [3, 3])
        net = utils.nets.slim2_conv2d(net, 256, 3, 1)

        out, extra = utils.fish_network.make_lstmConv(
            net,
            None,
            self.inputs.state.vp_frame,
            [['convLSTM', const.CONVLSTM_DIM, 3, const.CONVLSTM_STEPS, const.CONVLSTM_DIM]],
            stochastic = const.GQN3D_CONVLSTM_STOCHASTIC,
            weight_decay = 1E-5,
            is_training = True,
            reuse = False,
            output_debug = False,
        )

        out = utils.tfutil.tanh01(out)
        out.loss = extra['kl_loss']

        return out



     
    def predict(self):
        # st()
        self.prepare_inputs()

        # (batch_size x T) x 32 x 32 x 32 x dim
        memory_3D = self.building_3D_tensor()
        
        self.memory_3D = memory_3D
        ####################33 view pred here ###################
        inputs2Ddec = self.get_inputs2Ddec_gqn3d([memory_3D])
        outputs2Ddec = self.get_outputs2Ddec_gqn3d(inputs2Ddec)
        self.predicted_view = outputs2Ddec
        # st()

        if const.run_full:

            #memory_3D = self.debug_3d_tensor
            ## todo: if we have rpn here, we want it to convert rpn into raw_object_state
            raw_object_state = self.gt_raw_object_states
            object_size = tf.tile(tf.expand_dims(self.gt_object_size, 1), [1, self.T, 1, 1])
            raw_agent_state = self.gt_raw_agent_states


            # frozen state cannot be access by the prediction network, but can be updated based on the
            # the prediction, for example, accumulated delta_orn
            frozen_states=dict()
            object_states, frozen_object_states = self.raw_object_state_to_full_states(raw_object_state,raw_agent_state, self.inputs.action.actions,  object_size, memory_3D)
            agent_states, _ = self.raw_agent_state_to_full_states(raw_agent_state)
            frozen_states.update(frozen_object_states)

            # batch_size x T x nobjs (including agent hand) x dim
            #states = tf.concat([object_states, agent_states], 2)
            states = dict()
            states["object_states"] = object_states
            states["agent_states"] = agent_states
            node_predict_dim = dict()
            node_predict_dim["object"] = 13
            node_predict_dim["agent"] = 13

            actions = self.inputs.action.actions
            
            
            t = 0
            dynamics = self.Dynamics(self, class_utils.get_dict_state_t(states, t),\
                node_predict_dim=node_predict_dim)
            
            # one step froward
            est_dyn_states_delta = []
            for t in range(self.T):
                reuse=True
                if t == 0:
                    reuse=False
                if const.PRETRAIN_CONTACT:
                    gt_contact = self.gt_is_contact[:, t, :]
                else:
                    gt_contact = None
                est_dyn_states_delta.append(dynamics.predict_one_step(\
                    class_utils.get_dict_state_t(states, t), actions[:,t,:],\
                    gt_contact=gt_contact, reuse=reuse))

            if const.IS_PREDICT_CONTACT:
                pred_contact = [item[1] for item in est_dyn_states_delta]
                self.pred_contact = tf.stack(pred_contact, 1)
                est_dyn_states_delta = [item[0] for item in est_dyn_states_delta]

            #import ipdb; ipdb.set_trace()
            self.est_dyn_states_delta = tf.stack(est_dyn_states_delta, 1)

            est_states, est_frozen_states = self.update_state_with_delta(states, frozen_states,self.est_dyn_states_delta)
            #est_states, est_frozen_states = self.update_state_with_delta(states, frozen_states,gt_delta)

            self.est_states = est_states
            start_t = tf.cast(tf.random.uniform([1], minval=-0.499999, maxval=self.T - 0.000001 - self.rollout_T + 1, dtype=tf.float32)[0], tf.int32)

            # 5 steps rollout
            est_states_rollout = []
            est_frozen_states_rollout = []
            start_state = class_utils.get_dict_state_t(states, start_t) #states[:, start_t, :, :]
            frozen_state = class_utils.get_dict_state_t(frozen_states, start_t)
            current_state = start_state
            current_frozen_state = frozen_state
            for t in range(self.rollout_T):
                #delta_state = dynamics.predict_one_step(current_state, actions[:, start_t + t, :], reuse=reuse)
                if const.PRETRAIN_CONTACT:
                    gt_contact = self.gt_is_contact[:, start_t + t, :]
                else:
                    gt_contact = None
                delta_state = dynamics.predict_one_step(current_state, actions[:, start_t + t, :], gt_contact=gt_contact, reuse=True)
                if const.IS_PREDICT_CONTACT:
                    delta_state, pred_contact = delta_state
                ## for debugging
                #delta_state = gt_delta[:, start_t + t, :, :]

                if const.USE_AGENT_GT:
                    est_obj = delta_state[..., :self.nobjs, :]
                    gt_agent =  self.target.delta_agent_state[:, start_t + t, :, :]
                    delta_state = tf.concat([est_obj, gt_agent], axis=1)

                est_state, est_frozen_state = self.update_state_with_delta(current_state, current_frozen_state, delta_state, actions = actions[:, tf.minimum(start_t + t + 1, self.T - 1), :])

                est_states_rollout.append(est_state)
                est_frozen_states_rollout.append(est_frozen_state)
                current_frozen_state = est_frozen_state
                current_state = est_state
            est_states_rollout = class_utils.concat_dict_states(est_states_rollout)
            est_frozen_states_rollout = class_utils.concat_dict_states(est_frozen_states_rollout)
            
            if const.DEBUG_UNPROJECT:
                from utils.vis_np import save_voxel
                for t in range(5):
                    for batch_id in range(2):
                        save_voxel(est_frozen_states_rollout["object_3d_tensor"][batch_id, t, 0, :, :, :, 2], f"dump/frozen_t{t}_b{batch_id}.binvox")
            
            self.est_states_rollout = est_states_rollout
            if const.mode=="test":
                self.rollout_summary_(est_states, est_frozen_states, est_states_rollout, est_frozen_states_rollout, start_t, self.rollout_T)

    #############  3D tenor #################
    def building_3D_tensor(self):
        # inputs2Denc: [[(32x64x64x32), (32x64x64x32), (32x64x64x32)]]
        if const.DEBUG_UNPROJECT: 
            import scipy.misc
            for batch_id in range(const.BS):
                for view_id in range(3):
                    scipy.misc.imsave(f"dump/input_image_b{batch_id}_v{view_id}.png", 
                              self.inputs.state.frames[batch_id, 0, view_id, :, :, 2:])
                    scipy.misc.imsave(f"dump/input_depth_b{batch_id}_v{view_id}.png", self.inputs.state.frames[batch_id, 0, view_id, :, :,1])                    
                #view_id = 0
                #for t in range(self.T + 1):
                #    scipy.misc.imsave(f"dump/input_image_b{batch_id}_t{t}_v{view_id}.png",
                #              self.inputs.state.frames[batch_id, t, view_id, :, :, 2:])

        inputs2Denc = self.pass_2Denc()
        self.inputs2Denc = inputs2Denc
        precomputed_outline = None
        if const.OUTLINE_PRECOMPUTED:
            depth_ori = tf.unstack(self.inputs.state.depth_ori, axis=2)
            _, _, h, w, c = depth_ori[0].shape
            depth_ori2 = [tf.reshape(depth[:, :self.T, :, :, :], [-1, h, w, c]) for depth in depth_ori]
            unproject_depth_ori = grnn_op.unproject_depth_and_get_outline(depth_ori2)
            precomputed_outline = unproject_depth_ori
            output = [utils.voxel.resize_voxel(voxel, 0.25) for voxel in unproject_depth_ori]
        # import pickle
        # inputs2Denc = pickle.load(open("input.p","rb"))
        # st()
        unprojected_features = grnn_op.unproject_inputs(inputs2Denc, use_outline=const.USE_OUTLINE,\
                                                              debug_unproject=const.DEBUG_UNPROJECT)
        self.unprojected_features_check = unprojected_features
        self.unprojected_features = unprojected_features
        def merge_feat(feat_, output_):
            final_feat = []
            for view_id, feat_view in enumerate(feat_):
                final_feat.append(tf.concat([feat_view, output_[view_id]], axis=-1))
            return final_feat

        #self.unprojected_features = [merge_feat(feat, output) for feat in self.unprojected_features]

        if const.DEBUG_UNPROJECT:
            from utils.vis_np import save_voxel
            for batch_id in range(const.BS):
                for view_id in range(3):
                    save_voxel(self.unprojected_features[0][view_id][batch_id * self.T, :, :, :, 2],
                           f"dump/unprojected_depth_b{batch_id}_{view_id}.binvox")

                    if const.OUTLINE_PRECOMPUTED:
                        precompute =precomputed_outline[view_id][batch_id * self.T, :, :, :, :]
                        save_voxel(precompute[..., 0],
                           f"dump/precomputed_depth_b{batch_id}_{view_id}.binvox")

                        precompute = utils.voxel.resize_voxel(precomputed_outline[view_id], 0.25)
                        save_voxel(precompute[batch_id * self.T, ..., 0],
                           f"dump/precomputed_32_depth_b{batch_id}_{view_id}.binvox")

        to_base_features = grnn_op.pass_rotate_to_base(self.unprojected_features, self.thetas, self.phis,\
                                    self.base_theta, self.base_phi, aggre=const.AGGREGATION_METHOD)

        if precomputed_outline:
            to_base_depth = grnn_op.pass_rotate_to_base([precomputed_outline], self.thetas, self.phis,\
                                    self.base_theta, self.base_phi, aggre=const.AGGREGATION_METHOD)[0]

            # apply the mask on the features
            from utils.vis_np import save_voxel
            to_base_features_out = []
            for feat_id, feat in enumerate(to_base_features):
                _, sd, sh, sw, sc = feat[0].shape
                downsample_depth = [utils.utils.binarize(utils.voxel.resize_voxel(depth, scale = sd.value/128.0), 0.1) for depth in to_base_depth]
                nviews = len(feat)

                """
                for batch_id in range(const.BS):
                    for view_id in range(3):
                        save_voxel(feat[view_id][batch_id, :, :, :, 4], f"dump/aligned_b{batch_id}_v{view_id}.binvox")
                        save_voxel(feat[view_id][batch_id, :, :, :, 2], f"dump/aligned_depth_b{batch_id}_v{view_id}.binvox")
                        save_voxel(downsample_depth[view_id][batch_id, :, :, :, 0],\
                                    f"dump/aligned_precompute_b{batch_id}_v{view_id}.binvox")
                """
                if const.DEBUG_UNPROJECT:
                    to_base_features_out.append([downsample_depth[vid] for vid in range(nviews)])
                else:
                    to_base_features_out.append([feat[vid] * downsample_depth[vid] for vid in range(nviews)])
            to_base_features = to_base_features_out

        if const.DEBUG_UNPROJECT:
          ch = 2
          if const.OUTLINE_PRECOMPUTED:
                ch = 0
          for batch_id in range(const.BS):
            for view_id in range(3):
                save_voxel(to_base_features[0][view_id][batch_id * self.T, :, :, :, ch],
                           f"dump/aligned_depth_b{batch_id}_{view_id}.binvox")
                if const.OUTLINE_PRECOMPUTED:
                    precompute = to_base_depth[view_id][batch_id * self.T, :, :, :, :]
                    save_voxel(precompute[..., 0],
                           f"dump/aligned_precomputed_depth_b{batch_id}_{view_id}.binvox")
                    precompute = utils.voxel.resize_voxel(to_base_depth[view_id], 0.25)
                    """
                    save_voxel(precompute[batch_id * self.T, ..., 0],
                           f"dump/aligned_precomputed_32_depth_b{batch_id}_{view_id}.binvox")

                    save_voxel(precompute[batch_id * self.T, ..., 0],
                           f"dump/aligned_precomputed_32_0.3_depth_b{batch_id}_{view_id}.binvox", 0.3)
                    save_voxel(precompute[batch_id * self.T, ..., 0],
                           f"dump/aligned_precomputed_32_0.2_depth_b{batch_id}_{view_id}.binvox", 0.2)
                    """
                    save_voxel(precompute[batch_id * self.T, ..., 0],
                           f"dump/aligned_precomputed_32_0.1_depth_b{batch_id}_{view_id}.binvox", 0.1)

        if const.DEBUG_UNPROJECT:
            to_base_features_refined = to_base_features
        else:
            to_base_features_refined = grnn_op.get_refined3D(to_base_features,\
                is_training=self.is_training,\
                is_summ_feat=True, summ_inputs=self.summ_inputs, is_not_bn=const.IS_NOT_BN_IN_3D)

        memory_3D, _ = grnn_op.pass_aggregate(to_base_features_refined, aggre=const.AGGREGATION_METHOD)
        #memory_3D, aligned_features, _ = grnn_op.pass_rotate_to_base_and_aggregate(\
        #    unprojected_features_refined,\
        #    self.thetas, self.phis, self.base_theta, self.base_phi,\
        #    aggre=const.AGGREGATION_METHOD, is_summ_feat=False)

        memory_3D = tf.split(memory_3D[0], self.nviews)
        memory_3D_last = memory_3D[-1]

        # build agent mask
        if const.MASK_AGENT:
            memory_3D_last = memory_3D_last * (1 - self.agent_mask)
        if const.DEBUG_UNPROJECT:
            ch = 2
            if const.OUTLINE_PRECOMPUTED:
                ch = 0
            from utils.vis_np import save_voxel
            for batch_id in range(const.BS):
                for view_id in range(3):
                    #save_voxel(aligned_features[0][view_id][batch_id, :, :, :, 2],\
                    #       f"dump/aligned_b{batch_id}_v{view_id}.binvox")
                    save_voxel(to_base_features[0][view_id][batch_id, :, :, :, ch],\
                           f"dump/aligned_features_b{batch_id}_v{view_id}.binvox")
                if const.MASK_AGENT:
                    mask = self.agent_mask

                    save_voxel(memory_3D_last[batch_id, :, :, :, ch],\
                           f"dump/memory_b{batch_id}.binvox")
                    save_voxel(mask[batch_id * self.T, :, :, :, 0],\
                           f"dump/mask_b{batch_id}.binvox", 0.1)
                    save_voxel((1 - mask[batch_id * self.T, :, :, :, 0]) * memory_3D_last[batch_id, :, :, :, ch],\
                           f"dump/mask_memory_b{batch_id}.binvox")

        return memory_3D_last

    def pass_2Denc(self, is_summ_feat=False):
        all_frames = self.all_frames
        if const.USE_OUTLINE:
            # [bs x h x w x 3] * nviews
            all_frames = [frame[:,:,:,2:5] for frame in self.all_frames]


        output = grnn_op.get_outputs2Denc(all_frames, is_training=self.is_training, is_not_bn=const.IS_NOT_BN_IN_2D)
        self.output_check = output

        output = grnn_op.get_outputs2Ddec(output, is_training=self.is_training, fs_2D=const.fs_2D, is_not_bn=const.IS_NOT_BN_IN_2D)

        #if const.DEBUG_UNPROJECT:
        #    output = [all_frames]

        if const.USE_OUTLINE: # and is not const.OUTLINE_PRECOMPUTED:
            output_tmp = []
            for feats_id, feats in enumerate(output):
                feat_tmp = []
                for feat_id, feat in enumerate(feats):
                    s_h = tf.shape(feat)[1]
                    s_w = tf.shape(feat)[2]
                    resize_mask_depth = tf.image.resize_images(self.all_frames[feat_id][:,:,:,:2], [s_h, s_w])
                    feat_tmp.append(tf.concat([resize_mask_depth, feats[feat_id]], 3))
                output_tmp.append(feat_tmp)
            output = output_tmp
        return output

    #############  building features #################
    def tensor3d_to_feat(self, crop_features, scopename="tensor3d_to_feat", reuse=True):
        """
        feat: batch_size x nobjs x Sd x Sh x Sw x Sc
        """
        with tf.variable_scope(scopename, reuse=reuse):
            bs, nobjs, Sd, Sh, Sw, Sc = crop_features.get_shape()

            out = utils.nets.slim2_conv3d(tf.reshape(crop_features, [-1, Sd, Sh, Sw, Sc]), 16, 3, 2)
            #out = utils.nets.batch_norm(out, self.is_training, "layer1")
            out = utils.nets.slim2_conv3d(out, 32, 3, 2)
            #out = utils.nets.batch_norm(out, self.is_training, "layer2")
            out = utils.nets.slim2_conv3d(out, 32, 3, 2)
            #out = utils.nets.batch_norm(out, self.is_training, "layer3")
            out = utils.nets.slim2_conv3d(out, 64, 3, 2)

            return tf.reshape(out, [bs, nobjs, -1])

    def update_state_with_delta(self, states, frozen_states, dyn_state_delta, actions=None):
        """
        states: batch_size x T x nobjs x 81 or batch_size x nobjs x 81
        dyn_state_delta: batch_size x T x nobjs x 13 or batch_size x nobj x 13

        """
        updated_frozen_states = dict()
        object_states, agent_states = self.split_states(states)
         
        dyn_state_delta_object = dyn_state_delta[..., :self.nobjs, :]
        dyn_state_delta_agent = dyn_state_delta[..., self.nobjs:, :]
        _, dyn_states_delta_orn_objects, _ = self.split_states(dyn_state_delta_object, "h13")
        updated_object_dyn_state = self.update_dyn_state_with_delta(frozen_states["object_state"], dyn_state_delta_object)
        updated_agent_dyn_state = self.update_dyn_state_with_delta(agent_states, dyn_state_delta_agent)


        # batch_size x T x nobjs(3) x 3 x 3
        is_with_T_dim = (len(object_states.get_shape()) == 4)
        if is_with_T_dim:
            _, _, nobjs, _ = object_states.get_shape()
            bs_, T, _, c = dyn_states_delta_orn_objects.get_shape()
            bs = bs_ * T
            _, _, _, vd, vh, vw, vc = frozen_states["object_3d_tensor"].get_shape()


            dyn_states_delta_orn_objects = tf.reshape(dyn_states_delta_orn_objects, [bs, nobjs, c])
            object_3d_tensor = tf.reshape(frozen_states["object_3d_tensor"], [bs, nobjs, vd, vh, vw, vc])
        else:
            _, nobjs, _ = object_states.get_shape()
            bs, _, c = dyn_states_delta_orn_objects.get_shape()
            object_3d_tensor = frozen_states["object_3d_tensor"]

            _, _, vd, vh, vw, vc = object_3d_tensor.get_shape()

        object_rotmat = tfq.Quaternion(dyn_states_delta_orn_objects).as_rotation_matrix()
        displacement = tf.constant(np.zeros((bs, nobjs, 3, 1), dtype=np.float32), dtype=tf.float32)
        bottom_row = np.zeros((bs, nobjs, 1, 4), dtype=np.float32)
        bottom_row[:,:,0,3] = 1.0
        bottom_row = tf.constant(bottom_row)
        pad_matrix = tf.concat([
            tf.concat([object_rotmat, -displacement], axis = 3),
                    bottom_row], axis=2)
        pad_matrix = tf.reshape(pad_matrix, [bs * nobjs, 16])

        object_3d_tensor = tf.reshape(object_3d_tensor, [bs * nobjs, vd, vh, vw, vc])
        rotated_object_3d_tensor = tf.reshape(utils.voxel.rotate_voxel2(object_3d_tensor, pad_matrix),
                                              [bs, nobjs, vd, vh, vw, vc])

        # merge batch_size and T
        agent_state = tf.reshape(updated_agent_dyn_state[...,0, :3], [-1, 3])
        if actions is None:
            actions = tf.zeros_like(agent_state[:,:2])
        agent_3D = self.build_agent_feat(agent_state, actions, vd)

        # keep the first 2 dimension (bsxT) x nobjs
        object_size = tf.reshape(frozen_states["object_size"], [-1, self.nobjs, 3])
        hwd =  object_size* (1/(2 * const.boundary_to_center))
        updated_object_xyz = tf.reshape(updated_object_dyn_state[..., :3], [-1, self.nobjs, 3])
        rois_center = (updated_object_xyz * (1/const.boundary_to_center) + 1) * 0.5

        crop_hwd = 0.35 * tf.ones_like(hwd)
        rois = tf.stack([rois_center-crop_hwd, rois_center + crop_hwd], 3)
        rois = tf.reshape(rois, [-1, self.nobjs, 3, 2])

        crop_features_agent3D = utils.voxel.crop_and_resize_3d_box2_pad(agent_3D, rois, self.crop_size)
        crop_features = tf.concat([rotated_object_3d_tensor, crop_features_agent3D], axis=-1)
        # add the updated agent delta

        updated_object_states_feat = self.tensor3d_to_feat(crop_features, reuse=True)

        if is_with_T_dim:
            _, _, c = updated_object_states_feat.get_shape()
            updated_object_states_feat = tf.reshape(updated_object_states_feat, [bs_, T, nobjs, c])
            rotated_object_3d_tensor = tf.reshape(rotated_object_3d_tensor, [bs_, T, nobjs, vd, vh, vw, vc])
        updated_frozen_states = dict()
        updated_frozen_states["object_3d_tensor"] = rotated_object_3d_tensor
        updated_frozen_states["object_state"] = updated_object_dyn_state
        updated_frozen_states["object_size"] = frozen_states["object_size"]
        
        updated_states = dict()
        updated_states["agent_states"] = updated_agent_dyn_state
        updated_states["object_states"] = updated_object_states_feat
       
        return updated_states, updated_frozen_states

    def split_states(self, states, mode="obj_agent"):
        if mode == "obj_agent":
            return states["object_states"], states["agent_states"]
        elif mode == "h":
            # static dimension, feat dimension, dyn dimesntion
            return states[..., :self.object_static_dim], \
                   states[..., self.object_static_dim: self.object_static_dim + self.tensor3d_feat_dim:],\
                   states[..., self.object_static_dim + self.tensor3d_feat_dim:]
        elif mode == "h13":
            return states[..., :3], states[..., 3:7], states[..., 7:]
        else:
            raise Exception(f"data format is not supported: {mode}")

    def replace_object_orn_from_frozen(self, states, frozen_states):
        # replace object orn from frozen (because object orn is reset)
        object_states, agent_states = self.split_states(states)
        _, _, object_states_dyn = self.split_states(object_states, mode = "h")
        _, _, agent_states_dyn = self.split_states(agent_states, mode = "h")
        object_states_xyz, object_states_dyn_orn, object_states_dyn_vel = \
            self.split_states(object_states_dyn, mode = "h13")
        object_states_2 = tf.concat([object_states_xyz, frozen_states["object_orn"], object_states_dyn_vel], -1)
        return object_states_2, agent_states_dyn

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

    def prepare_inputs(self):
        self.summ_inputs = dict()
        self.bs, T, self.nviews, img_h, img_w, img_c= self.inputs.state.frames.shape
        # about the image
        # [(bs x T)] * n_views
        self.all_frames = [tf.reshape(tf.squeeze(img, 2), [-1, img_h, img_w, img_c])\
            for img in tf.split(self.inputs.state.frames[:,:self.T, :, :, :, :],\
            self.nviews.value, axis=2)]
        self.summ_inputs["all_frames"] = self.all_frames

        bs, T, nviews = self.inputs.state.phis.shape
        # [(bs x T)] * n_views
        thetas = [tf.squeeze(x, axis=1) for x in tf.split(tf.reshape(self.inputs.state.thetas[:,:self.T, :],\
                 [self.bs*self.T, nviews]), nviews.value, axis = 1)]

        phis = [tf.squeeze(x, axis=1) for x in tf.split(tf.reshape(self.inputs.state.phis[:, :self.T, :],\
                 [self.bs*self.T, nviews]), nviews.value, axis = 1)]

        self.thetas = thetas
        self.phis = phis
        self.base_theta = tf.ones_like(thetas[0], dtype=tf.float32) * 180.0
        self.base_phi = tf.zeros_like(phis[0], dtype=tf.float32) # want everything to lie on ground
        # st()
        if const.run_full:
            self.orn_reset_base = tf.constant(np.array([1, 0, 0, 0], dtype=np.float32))
            gt_raw_object_states = self.inputs.state.obj_state[:, :self.T, :, :]
            self.gt_raw_object_states = self.reset_orn(gt_raw_object_states)
            self.gt_raw_agent_states = self.inputs.state.agent_state[:, :self.T, :, :]

    def raw_agent_state_to_full_states(self, raw_agent_state):
        # add 0 in the first dimension as object code
        # object_code (1) + hwd (3) + features(64) + xyzorn(13) -> 81
        input_agent_hwd = 0.2 * tf.ones_like(raw_agent_state[:,:,:,:3])
        bs, T, njoints, dim = raw_agent_state.get_shape()
        #zero_feat = tf.zeros([bs, T, njoints, self.tensor3d_feat_dim], dtype=tf.float32)
        # add 1 in the first dimension as object code
        #full_agent_state = tf.concat([tf.ones((bs, T, njoints, 1)), input_agent_hwd, zero_feat, raw_agent_state], 3)
        full_agent_state = raw_agent_state
        return full_agent_state, None

    def build_agent_feat(self, agent_state, actions, tensor_size):
        """
        agent_State: batch_size x 3, agent_xyz
        actions: batch_size x 2
        """
        agent_state= agent_state * (1/const.boundary_to_center)

        agent_3D = vis_op.make_dot_3D(agent_state, tensor_size)
        agent_3D = tf.reshape(agent_3D, [-1, tensor_size, tensor_size, tensor_size, 1])
        actions = tf.reshape(actions, [-1, 1, 1, 1, 2])
        GAUSS_THRES= 3.0
        if const.DEBUG_UNPROJECT:
            agent_mask = utils.utils.binarize(agent_3D, GAUSS_THRES)
        else:
            agent_mask = actions * tf.tile(utils.utils.binarize(agent_3D, GAUSS_THRES), [1, 1, 1, 1, 2])
        return agent_mask

    def raw_object_state_to_full_states(self, raw_obj_state, raw_agent_state, actions, object_size, tensor3D):
        """
        This function converts raw_state (xyzorn) into full_state
        # object/agent code(1) + hwd(3)  + feat(64) + xyzornvel(13) -> 17
        raw_object_state: batch_size x T x nobjs x 13
        note: from object proposal, it should provide xyz, object_size; orn and vel is set to 0
        object_size: batch_size x T x 2 x 3
        tensor3D: (batch_size x T) x 32 x 32 x 32 x 8
        """
        # normalize raw statos
        hwd = object_size * (1 / (2*const.boundary_to_center))
        # [-1.5, 1.5] - > [0, 1]
        rois_center = (raw_obj_state[:, :, :, :3] * (1/const.boundary_to_center) + 1) * 0.5

        if const.BBOX_RANDOMIZATION:
            rois_center = tf.where(self.is_training,
                rois_center + 0.06 * tf.random.uniform(rois_center.shape, -1, 1),
                rois_center)
        # data augmentation here?

        tensor_size  = tensor3D.get_shape()[1] 
        bs, T, _, _ = raw_agent_state.get_shape()
        actions = tf.reshape(actions, [-1, 2])
        agent_state = tf.reshape(raw_agent_state[:,:,0, :3], [-1, 3])
        #agent_3D = vis_op.make_dot_3D(agent_state, tensor_size)
        agent_3D = self.build_agent_feat(agent_state, actions, tensor_size)

        #import ipdb; ipdb.set_trace()
        # mask for xy

        #tensor3D = tf.concat([tensor3D, agent_3D, agent_mask], -1)

        if const.DEBUG_UNPROJECT: 
            import scipy.misc
            from utils.vis_np import save_voxel

            agent_3D_flat = tf.reduce_mean(agent_mask, 2)
            for batch_id in range(const.BS):

                save_voxel(agent_3D[batch_id, :, :, :, 0], f"dump/agent_dot_b{batch_id}.binvox")
                #for t in range(self.T):
                #    scipy.misc.imsave(f"dump/flat_agent_b{batch_id}_t{t}.png",\
                #        tf.tile(tf.reverse(agent_3D_flat[batch_id * self.T + t, :, :,:], [0,1]), [1,1,3]).numpy())
        #import ipdb; ipdb.set_trace()

        # the crop should be larger
        crop_hwd = 0.35 * tf.ones_like(hwd)
        rois = tf.stack([rois_center-crop_hwd, rois_center + crop_hwd], 4)
        rois = tf.reshape(rois, [self.bs * self.T, self.nobjs, 3, 2])

        # 3d feature from scene
        crop_features_tensor3D = utils.voxel.crop_and_resize_3d_box2_pad(tensor3D, rois, self.crop_size)
        _, _, s, s, s, tensor_feat = crop_features_tensor3D.get_shape()
        crop_features_T = tf.reshape(crop_features_tensor3D, [self.bs, self.T, self.nobjs, s, s, s, tensor_feat])

        # agent action
        crop_features_agent3D = utils.voxel.crop_and_resize_3d_box2_pad(agent_3D, rois, self.crop_size)

        crop_features = tf.concat([crop_features_tensor3D, crop_features_agent3D], axis=-1)
        #tensor3D = tf.concat([tensor3D, agent_mask], -1)
        #object_tensor_feat, crop_features = self.get_tensor_feat_from_rois(rois, tensor3D)
        #crop_features = utils.voxel.crop_and_resize_3d_box2_pad(tensor3D, tf.reshape(rois, [self.bs*self.T, self.nobjs, 3, 2]), self.crop_size)

        object_tensor_feat = tf.reshape(self.tensor3d_to_feat(crop_features, reuse=False), [self.bs, self.T, self.nobjs, -1])
        if const.DEBUG_UNPROJECT:
            ch = 2
            if const.OUTLINE_PRECOMPUTED:
                ch = 0
            from utils.vis_np import save_voxel
            resize_pad_out = utils.voxel.resize_voxel(self.pad_out, 0.5)
            resize_pad_out_32 = utils.voxel.resize_voxel(self.pad_out, 0.25)
            resize_pad_out_16 = utils.voxel.resize_voxel(self.pad_out, 0.125)

            for batch_id in range(const.BS):
                save_voxel(self.pad_out[self.nobjs * batch_id, :, :, :, 0], f"dump/padpad_128_b{batch_id}.binvox")
                save_voxel(resize_pad_out_32[self.nobjs * batch_id, :, :, :, 0], f"dump/padpad_32_b{batch_id}.binvox")
                save_voxel(resize_pad_out_16[self.nobjs * batch_id, :, :, :, 0], f"dump/padpad_16_b{batch_id}.binvox")
                save_voxel(resize_pad_out[self.nobjs * batch_id, :, :, :, 0], f"dump/resize_padpad_64_b{batch_id}.binvox")
                save_voxel(tensor3D[batch_id, :, :, :, ch], f"dump/depth_b{batch_id}.binvox")
                save_voxel(tensor3D[batch_id, :, :, :, ch] + tensor3D[batch_id, :, :, :, -1], f"dump/depth_with_agent_b{batch_id}.binvox")
                save_voxel(crop_features[batch_id, 0, 0, :, :, :, ch] + crop_features[batch_id, 0, 0, :, :, :, -1], f"dump/crop_feat_with_action_b{batch_id}.binvox")
                save_voxel(crop_features[batch_id, 0, 0, :, :, :, ch], f"dump/crop_feat_b{batch_id}.binvox")
                #save_voxel(crop_features[batch_id, 0, 0, :, :, :, ch], f"dump/crop_action_b{batch_id}.binvox")

        self.tensor3d_feat_dim = object_tensor_feat.get_shape()[-1].value
        full_object_state = self.bind_obj_feat(object_tensor_feat, raw_obj_state, object_size)

        frozen_state = dict()
        #init_orn = tf.tile(tf.reshape(self.orn_reset_base, [1,1,1,4]), [self.bs, self.T, self.nobjs, 1])
        # batch_size x T x 2 x 13
        frozen_state["object_state"] = raw_obj_state
        # batch_size x T x 2 x 16 x 16 x 16 x 8
        frozen_state["object_3d_tensor"] = crop_features_T
        # batch_size x T x 2 x 3
        frozen_state["object_size"] = object_size
        return full_object_state, frozen_state



    def get_tensor_feat_from_rois(self, rois, tensor3D):
        """
        This function croppsed features
        rois: batch_size x T x nobjs x 3 x 2
        tensor3D: (batch_size x T) x
        """
        crop_features = utils.voxel.crop_and_resize_3d_box2_pad(tensor3D, tf.reshape(rois, [self.bs*self.T, self.nobjs, 3, 2]), self.crop_size)

        _, _, s, s, s, tensor_feat = crop_features.get_shape()
        crop_features_T = tf.reshape(crop_features, [self.bs, self.T, self.nobjs, s, s, s, tensor_feat])

        object_tensor_feat = tf.reshape(self.tensor3d_to_feat(crop_features, reuse=False), [self.bs, self.T, self.nobjs, -1])
        return object_tensor_feat, crop_features_T


    def bind_obj_feat(self, object_tensor, object_xyzorn, hwd):
        """
        object_tensor: bs x T x nobjs x 64
        hwd: bs x 1 x nobjs x 3
        obj_xyzorn: bs x T x nobjs x 13
        """
        bs, T, nobjs, _ = object_tensor.get_shape()
        # object/agent code(1) + hwd(3)  + feat(64) + xyzornvel(13) -> 81
        #input_object_state = tf.concat([hwd, object_tensor, object_xyzorn], -1)
        #input_object_state = tf.concat([tf.zeros((bs, T, nobjs, 1)), input_object_state], 3)
        input_object_state = object_tensor
        return input_object_state

    #############  visualization #################
    def rollout_summary_(self, est_states, est_frozen_states, est_states_rollout, est_frozen_states_rollout, start_t, T=5):
        basic_info = dict()
        basic_info["voxel"] = self.inputs.state.voxels
        basic_info["object_size"] = self.inputs.state.resize_factor

        _, est_agent_states = self.split_states(est_states)
        est_object_states = est_frozen_states["object_state"]
        #est_object_states_2, est_agent_states_dyn = self.replace_object_orn_from_frozen(est_states, est_frozen_states)
        # for visualization, multiply with object abosolute rotations
        est_object_states_2 = self.multiply_orn_with_origin(est_object_states, self.inputs.state.obj_state[:, :self.T, ...])
        est_states_to_print = tf.concat([est_object_states_2, est_agent_states], -2)

        _, est_agent_states_rollout = self.split_states(est_states_rollout)
        est_object_states_rollout = est_frozen_states_rollout["object_state"]
        #est_object_states_rollout2, est_agent_states_dyn_rollout = self.replace_object_orn_from_frozen(est_states_rollout, est_frozen_states_rollout)
        est_object_states_rollout2 = self.multiply_orn_with_origin(est_object_states_rollout, \
                           tf.tile(self.inputs.state.obj_state[:, start_t:start_t + 1, :, :], [1, T,  1, 1]))
        est_states_rollout_to_print = tf.concat([est_object_states_rollout2, est_agent_states_rollout], -2)
        self.rollout_summary(basic_info, est_states_to_print, est_states_rollout_to_print, start_t, T=T)
