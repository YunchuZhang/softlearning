import tensorflow as tf
import tensorflow.contrib.slim as slim
import constants as const
import utils
import os.path as path
import numpy as np
# import inputs
from tensorflow import summary as summ
from munch import Munch
import ipdb
import math
import tfquaternion as tfq

st = ipdb.set_trace

class Net:
    def __init__(self):
        super(Net, self).__init__()
        val = self.data()
        self.is_training =True
        self.weights = {}

        # self.setup_data(val)
        # self.predict()
        # self.extra_out = dict()
        # self.loss()
        self.optimize(lambda: self.go_up_to_loss(val,self.is_training))

        # st()
        # self.weights = {}
        # self.input = input_
        # self.extra_out = None
        # # st()
        # #this is used to select whether to pull data from train, val, or test set
        # # self.data_selector = input_.q_ph

        # if not const.eager:
        #     self.is_training = tf.placeholder(dtype = tf.bool, shape = (), name="is_training")
        # else:
        #     self.is_training = None
    def clip_data(self, item, start_t, len):
        item = tf.reshape(item[:, start_t:start_t + len, ...],
                      item.shape[0:1].as_list() + [len] + item.shape[2:].as_list())
        return item

    def init_placeholders(self):
        # basic_info = pickle.load(f)
        T = 2
        W = 128
        H = 128
        N =1
        self.max_T = 1
        self.image_size_h =64
        self.image_size_w =64
        print(const.mode)
        
        #TODO_MIHIR change these parameters and placeholders based on the data


        self.images = tf.placeholder(tf.float32, [const.BS, T, const.NUM_VIEWS, W, H, 4],"images")
        self.angles = tf.placeholder(tf.float32, [const.BS, T, const.NUM_VIEWS, 2],"angles")
        self.zmaps = tf.placeholder(tf.float32, [const.BS, T, const.NUM_VIEWS, W, H, 1],"zmapss")
        self.xyzorn_objects = tf.placeholder(tf.float32, [const.BS,T,2,13],"xyzorn_objects")
        self.object_class = tf.placeholder(tf.float32, [const.BS, T, 2],"object_class")
        self.xyzorn_agent = tf.placeholder(tf.float32, [const.BS,T,1,13],"xyzorn_agent")
        self.action = tf.placeholder(tf.float32, [const.BS,1,2],"action")
        self.voxels = tf.placeholder(tf.float32, [const.BS, T,W,H,W],"voxels")
        self.resize_factor = tf.placeholder(tf.float32, [const.BS, 2,3],"resize_images")
        self.raw_seq_filename = tf.placeholder(tf.float32, [const.BS],"raw_seq_filename")

        self.dictVal =  Munch(images=self.images,angles=self.angles,zmaps=self.zmaps,xyzorn_objects=self.xyzorn_objects,object_class=self.object_class,\
        xyzorn_agent=self.xyzorn_agent,action=self.action,voxels=self.voxels,resize_factor=self.resize_factor,\
        raw_seq_filename=self.raw_seq_filename)
        

        return self.dictVal

    def data(self):
        #data = self.child.data_for_selector(self.q_ph)
        data = self.init_placeholders()
        input_T = data.images.get_shape()[1].value
        start_t = tf.cast(tf.random.uniform([1], minval=0, maxval=input_T-0.000001 - self.max_T, dtype=tf.float32)[0], tf.int32)
        data.images = self.clip_data(data.images, start_t, self.max_T + 1)
        data.angles = self.clip_data(data.angles, start_t, self.max_T + 1)
        data.zmaps = self.clip_data(data.zmaps, start_t, self.max_T + 1)
        data.xyzorn_objects = self.clip_data(data.xyzorn_objects, start_t, self.max_T + 1)
        data.object_class = self.clip_data(data.object_class, start_t, self.max_T + 1)
        data.xyzorn_agent = self.clip_data(data.xyzorn_agent, start_t, self.max_T + 1)
        data.action = self.clip_data(data.action, start_t, self.max_T)

        data = Munch(inputs = self.make_inputs(data),
                     target = self.make_target(data)
                    )
         
        return data

    def get_delta(self, xyzorn, xyzorn_after):
        delta_pos = xyzorn_after[:,:,:,:3] - xyzorn[:,:,:,:3]
        #quat1 = tfq.Quaternion(tf.concat([xyzorn[:,:,:,6:7], xyzorn[:,:,:,3:6]], 3))
        #quat2 = tfq.Quaternion(tf.concat([xyzorn_after[:,:,:,6:7], xyzorn_after[:,:,:,3:6]], 3))
        quat1 = tfq.Quaternion(xyzorn[:,:,:,3:7])
        quat2 = tfq.Quaternion(xyzorn_after[:,:,:,3:7])

        delta_quat = quat2 * quat1.inverse()
        delta_vel =  xyzorn_after[:,:,:,7:13] - xyzorn[:,:,:,7:13]
        return tf.concat([delta_pos, delta_quat, delta_vel], 3)[:,:15, :, :]


    def make_inputs(self, data):
        return Munch(state = self.make_state(data),
                     action = self.make_action(data))

    def make_state(self, data):
        bs, T, nviews, image_h, image_w, image_c = data.zmaps.get_shape()
        zmaps_small = tf.image.resize_images(tf.reshape(data.zmaps, [bs * T * nviews, \
                     image_h, image_w, image_c]), [self.image_size_h, self.image_size_w])
        zmaps_small = tf.reshape(zmaps_small, [bs, T, nviews, self.image_size_h, self.image_size_w, image_c])

        bs, T, nviews, image_h, image_w, image_c = data.images.get_shape()
        images_small = tf.image.resize_images(tf.reshape(data.images, [bs * T * nviews, \
                     image_h, image_w, image_c]), [self.image_size_h, self.image_size_w])
        images_small = tf.reshape(images_small, [bs, T, nviews, self.image_size_h, self.image_size_w, image_c])

        mask_dump = tf.ones_like(zmaps_small)
        # mask, depth, rgb
        frames = tf.concat([mask_dump, zmaps_small,
                            images_small[:, :, :, :, :, :3]], 5)

        # batch_size x T x num_views x 2
        phis, thetas = tf.split(data.angles, 2, axis=3)
        phis = tf.squeeze(phis, 3)
        thetas = tf.squeeze(thetas, 3)

        return Munch(frames=frames[:, :, :const.NUM_VIEWS], depth_ori = data.zmaps, cameras=data.angles[:, :, :const.NUM_VIEWS], phis = phis[:, :, :const.NUM_VIEWS], thetas=thetas[:, :, :const.NUM_VIEWS],
                    vp_frame = frames[:, 0, -1, :, :, -3:], vp_phi = phis[:, 0, -1], vp_theta = thetas[:, 0, -1],
                     obj_state=data.xyzorn_objects[:, :-1, :, :],
                     object_class=data.object_class,
                     agent_state=data.xyzorn_agent[:, :-1, :, :],
                     voxels=data.voxels, resize_factor=data.resize_factor)

    def make_action(self, data):
        return Munch(actions=data.action)
     
    def make_target(self, data):
        delta_obj_state = self.get_delta(data.xyzorn_objects[:,:-1,:,:],\
                          data.xyzorn_objects[:,1:,:,:])
        delta_agent_state = self.get_delta(data.xyzorn_agent[:,:-1,:,:],\
                            data.xyzorn_agent[:,1:,:,:])
        Nagents = delta_agent_state.get_shape()[2]
        bs, T, dim = data.object_class.get_shape()
        agent_class = tf.ones((bs, T, Nagents)) # always exists
        all_class = tf.concat([data.object_class, agent_class], 2)

        out =  Munch(obj_state=data.xyzorn_objects[:,1:,:,:],
                     agent_state=data.xyzorn_agent[:,1:,:,:],
                     delta_obj_state=delta_obj_state,
                     delta_agent_state=delta_agent_state,
                     object_class = data.object_class,
                     all_class = all_class)
        if const.mode == "test":
            out["images_front"] = data.images_front
        return out

    def add_weights(self, name):
        st()
        self.weights[name] = utils.tfutil.current_scope_and_vars()

    def optimize(self, fn):
        self.optimizer = tf.train.AdamOptimizer(const.lr, const.mom)
        self.opt = utils.tfutil.make_opt_op(self.optimizer, fn)

    def call(self, is_training=None):
        #index is passed to the data_selector, to control which data is used
        #self.go_up_to_loss(index)
        self.optimize(lambda: self.go_up_to_loss(is_training))
        self.assemble()

    def go_up_to_loss(self, is_training = None):
        #should save the loss to self.loss_
        #and also return the loss
        raise NotImplementedError

    def build_vis(self):
        #should save a Munch dictionary of values to self.vis
        raise NotImplementedError

    def build_evaluator(self):
        #should save a Munch dictionary of values to self.vis
        raise NotImplementedError

    def assemble(self):
        #define all the summaries, visualizations, etc
        with tf.name_scope('summary'):
            summ.scalar('loss', self.loss_) #just to keep it consistent

        if not const.eager:
            if const.is_trainval_diff_summ:
                self.small_summary = tf.summary.merge_all('scalar')
                self.summary = tf.summary.merge_all("all")
            else:
                self.summary = tf.summary.merge_all()
                self.small_summary = tf.summary.merge_all()
        else:
            self.summary = None

        #self.evaluator = Munch()
        self.build_evaluator()
        self.build_vis()

        #these are the tensors which will be run for a single train, test, val step
        self.test_run = Munch(evaluator = self.evaluator, vis = self.vis, summary = self.summary)
        self.train_run = Munch(opt = self.opt, summary = self.small_summary) #, extra_out=self.extra_out)
        #self.train_run = Munch(loss = self.loss_, summary = self.summary)
        self.val_run = Munch(loss = self.loss_, #acc = self.acc_, acc_3 = self.acc_3_, acc_5 = self.acc_5acc_10 = self.acc_10_
                             summary = self.summary, vis = self.vis)
