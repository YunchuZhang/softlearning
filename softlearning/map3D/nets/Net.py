import tensorflow as tf
import tensorflow.contrib.slim as slim
import constants as const
import utils_map as utils
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
		self.is_training =True
		self.weights = {}
		self.opname = const.opname
		self.detector = False
		self.action_predictor = False
		# self.setup_data(val)
		# self.predict()
		# self.extra_out = dict()
		# self.loss()
		# self.optimize(lambda: self.go_up_to_loss(val,self.is_training))


	def clip_data(self, item, start_t, len):
		item = tf.reshape(item[:, start_t:start_t + len, ...],
					  item.shape[0:1].as_list() + [len] + item.shape[2:].as_list())
		return item

	def normalize_custom(self, images, angles, zmaps):
		images = tf.cast(images, dtype=tf.float32)
		# voxels = tf.cast(voxels, dtype=tf.float32)
		# N = self.basic_info["num_views"]
		N=54
		def gather_image_(images, idx):
			images_NTHWC = tf.transpose(images, [2,1, 0, 3, 4, 5])
			images_crop = tf.gather_nd(images_NTHWC, tf.expand_dims(idx, 1))
			return tf.transpose(images_crop, [2,1, 0, 3, 4, 5])

		def gather_angles_(angles, idx):
			angles_NTC = tf.transpose(angles, [2,1, 0, 3])
			angles_crop = tf.gather_nd(angles_NTC, tf.expand_dims(idx, 1))
			return tf.transpose(angles_crop, [2,1, 0, 3])

		def extract_rand_view(images, zmaps, angles, sample_size):
			# idx = tf.random_shuffle(tf.range(N), seed=0 if const.mode=="test" else None)[:sample_size]
			# idx = tf.convert_to_tensor([21, 46, 10, 17])
			idx = tf.convert_to_tensor([0,1,2,3])

			images = gather_image_(images, idx)
			zmaps = gather_image_(zmaps, idx)
			angles = gather_angles_(angles, idx)
			return images, zmaps, angles, idx
		# if const.mode == "test":
		#     images_front = images[:, 13, :, :, :3]
		#     images_side = images[:, 8, :, :, :3]
		#     images_front = tf.stack([images_front, images_side], 1)
		#     images_front = images_front/255.0 - 0.5

		nviews = const.NUM_VIEWS
		if const.IS_VIEW_PRED:
			nviews += const.NUM_PREDS

		images, zmaps, angles, idx = extract_rand_view(images, zmaps, angles, nviews)

		angles = tf.convert_to_tensor(angles)
		zmaps = tf.convert_to_tensor(zmaps)

		input_T = images.get_shape()[1].value
		# randomly select a chunk that has length of max_T
		
		start_t = tf.cast(tf.random.uniform([1], minval=0, maxval=input_T-0.000001 - self.max_T,dtype=tf.float32)[0], tf.int32)        

		images = self.clip_data(images, start_t, self.max_T+1)
		angles = self.clip_data(angles, start_t, self.max_T+1)
		zmaps = self.clip_data(zmaps, start_t, self.max_T+1)

		# xyzorn_objects = self.clip_data(xyzorn_objects, start_t, self.max_T+1)
		# object_class = self.clip_data(object_class, start_t, self.max_T + 1)
		# xyzorn_agent = self.clip_data(xyzorn_agent, start_t, self.max_T + 1)
		# actions = self.clip_data(actions, start_t, self.max_T)

		# images = images
		#tf.compat.as_text(raw_seq_filename, encoding='utf-8')
		#st()
		names = ['images', 'angles', 'zmaps']
		stuff = [images, angles, zmaps]
		if const.mode == "test":
			names += ["images_front"]
			stuff += [images_front]
		return Munch(zip(names, stuff))

	def init_placeholders(self):
		# basic_info = pickle.load(f)
		T = 1
		W = 84
		H = 84
		# N =1
		self.max_T = const.max_T

		# self.max_T = 0
		self.image_size_h =const.H
		self.image_size_w =const.W
		print(const.mode)

		#TODO_MIHIR change these parameters and placeholders based on the data

		self.images = tf.placeholder(tf.float32, [const.BS, T, const.NUM_VIEWS, W, H, 4],"images")
		self.angles = tf.placeholder(tf.float32, [const.BS, T, const.NUM_VIEWS, 2],"angles")
		self.zmaps = tf.placeholder(tf.float32, [const.BS, T, const.NUM_VIEWS, W, H, 1],"zmapss")


		self.dictVal =  Munch(images=self.images,angles=self.angles,zmaps=self.zmaps)
		

		return self.dictVal

	def init_Inputs(self,images,zmaps,angles):
		# basic_info = pickle.load(f)
		# T = 1
		# W = 128
		# H = 128
		# N =1
		T = 1
		W = 84
		H = 84
		# N =1
		self.max_T = const.max_T

		# self.max_T = 0
		self.image_size_h =const.H
		self.image_size_w =const.W

		dictVal = self.normalize_custom(images,angles,zmaps)

		return dictVal

	def data(self,images,zmaps,angles,goal,state_observation):

		#data = self.child.data_for_selector(self.q_ph)
		
		# self.images_val = images
		data = self.init_Inputs(images,zmaps,angles)
		#goal_ph = tf.placeholder(tf.float32, [const.BS, 5], "goal_centroid")
		# st()
		data.update({'goal':goal})
		data.goal = goal
		data.update({'state_observation':state_observation})
		data.state_observation = state_observation
		# data = self.init_placeholders()
		input_T = data.images.get_shape()[1].value
		start_t = tf.cast(tf.random.uniform([1], minval=0, maxval=input_T-0.000001 - self.max_T, dtype=tf.float32)[0], tf.int32)

		data.images = self.clip_data(data.images, start_t, self.max_T + 1)
		data.angles = self.clip_data(data.angles, start_t, self.max_T + 1)
		data.zmaps = self.clip_data(data.zmaps, start_t, self.max_T + 1)

		if const.run_full:
			data.xyzorn_objects = self.clip_data(data.xyzorn_objects, start_t, self.max_T + 1)
			data.object_class = self.clip_data(data.object_class, start_t, self.max_T + 1)
			data.xyzorn_agent = self.clip_data(data.xyzorn_agent, start_t, self.max_T + 1)
			data.action = self.clip_data(data.action, start_t, self.max_T)
			data = Munch(inputs = self.make_inputs(data),
						 target = self.make_target(data)
						)
		else:
			data = Munch(inputs = self.make_inputs(data))

		return data
	def child_normalize(self,data):
		images = tf.cast(data.images, tf.uint8)
		angles = tf.cast(data.angles, tf.float32)
		zmaps = tf.cast(data.zmaps, tf.float32)

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
		
		if const.run_full:
			return Munch(state = self.make_state(data),
						 action = self.make_action(data))
		else:
			return Munch(state = self.make_state(data))


 
	def make_state(self, data):
		bs, T, nviews, image_h, image_w, image_c = data.zmaps.get_shape()
		bs = const.BS
		zmaps_small = tf.image.resize_images(tf.reshape(data.zmaps, [bs * T * nviews, \
					 image_h, image_w, image_c]), [self.image_size_h, self.image_size_w])
		zmaps_small = tf.reshape(zmaps_small, [bs, T, nviews, self.image_size_h, self.image_size_w, image_c])

		bs, T, nviews, image_h, image_w, image_c = data.images.get_shape()
		# st()
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
		if const.run_full:
			return Munch(frames=frames[:, :, :const.NUM_VIEWS], depth_ori = data.zmaps, cameras=data.angles[:, :, :const.NUM_VIEWS], phis = phis[:, :, :const.NUM_VIEWS], thetas=thetas[:, :, :const.NUM_VIEWS],
						vp_frame = frames[:, 0, -1, :, :, -3:], vp_phi = phis[:, 0, -1], vp_theta = thetas[:, 0, -1],
						 obj_state=data.xyzorn_objects[:, :-1, :, :],
						 object_class=data.object_class,
						 agent_state=data.xyzorn_agent[:, :-1, :, :],
						 voxels=data.voxels, resize_factor=data.resize_factor)
		else:
			return Munch(frames=frames[:, :, :const.NUM_VIEWS], depth_ori = data.zmaps, cameras=data.angles[:, :, :const.NUM_VIEWS], phis = phis[:, :, :const.NUM_VIEWS], thetas=thetas[:, :, :const.NUM_VIEWS],
						vp_frame = frames[:, 0, -1, :, :, -3:], vp_phi = phis[:, 0, -1], vp_theta = thetas[:, 0, -1], goal = data.goal,state_observation = data.state_observation)


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
		self.weights[name] = utils.tfutil.current_scope_and_vars()


	def optimize(self, fn):
		global_step = tf.Variable(0, trainable=False)
		decay_steps = 2000
		lr = tf.train.exponential_decay(0.0002,
										global_step,
										decay_steps,
										0.1,
										staircase=True)
		self.optimizer = tf.train.AdamOptimizer(lr, const.mom)
		self.opt = utils.tfutil.make_opt_op(self.optimizer, fn)

	def __call__(self,images,angles,zmaps,goal,state_observation,batch_size=None,exp_name=None,is_training=None,reuse=False,eager=False,position=None):
		#index is passed to the data_selector, to control which data is used
		#self.go_up_to_loss(index)
		# st()
		# self.exp_name = exp_name
		# st()
		if exp_name:
			const.set_experiment(exp_name)
			self.__dict__.update(const.__dict__)
		
		if batch_size:
			const.BS = batch_size
		if eager:
			const.eager = eager
			const.DEBUG_UNPROJECT =True

		# self.detector = detector
		if position is not None:
			self.position = tf.layers.flatten(position)
		# st()
		const.fx = const.W / 2.0 * 1.0 / math.tan(const.fov * math.pi / 180 / 2)
		const.fy = const.fx
		const.focal_length = const.fx / (const.W / 2.0)

		const.x0 = const.W / 2.0
		const.y0 = const.H / 2.0
		with tf.compat.v1.variable_scope("Variables",reuse=reuse):
			val = self.data(images,zmaps,angles,goal,state_observation)
			self.optimize(lambda: self.go_up_to_loss(val,is_training))
			self.assemble()
		return self.memory_3D

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
