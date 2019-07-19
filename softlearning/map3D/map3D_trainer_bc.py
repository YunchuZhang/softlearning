from os import path as osp
import numpy as np
import tensorflow as tf
import math
import time
from numbers import Number
from collections import OrderedDict
from nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain
import ipdb
import time
import os 
import pickle

import utils_map as utils

st = ipdb.set_trace
def create_stats_ordered_dict(
		name,
		data,
		stat_prefix=None,
		always_show_all_stats=True,
		exclude_max_min=False,
):

	return stats


"""
Implementation of save_image() and make_grid() of PyTorch in numpy

Original Pytorch implementation can be found here:
https://github.com/pytorch/vision/blob/master/torchvision/utils.py
# """
#             map3D =bulledtPush,
#             training_environment=training_environment,
#             initial_exploration_policy=initial_exploration_policy,
#             pool=replay_pool,
#             observation_keys = observation_keys,
#             sampler=sampler,
#             session=self._session)


class MappingTrainer():
	def __init__(self,variant,
			map3D,
			training_environment,
			initial_exploration_policy,
			pool,
			batch_size,
			observation_keys,
			sampler,
			eager_enabled,
			exp_name,
			session):
		# session = tf.keras.backend.get_session()
		# st()
		self._n_initial_exploration_steps = variant['algorithm_params']['kwargs']["n_initial_exploration_steps"]

		self.eager_enabled = eager_enabled

		self._n_initial_exploration_steps 

		self.sampler = sampler

		path = "/projects/katefgroup/yunchu/expert_mug3"
		
		self.batch_size =batch_size
		self.observation_keys = observation_keys
		print(self.observation_keys)
		self.pool = pool
		self.log_interval = 10

		#self.action_predictior = action_predictior


		self.export_interval = 100
		self.model = map3D
		self.exp_name = exp_name
		if self.eager_enabled:
			self.forwardPass()
		else:
			self.initGraph()
		# st()
		if eager_enabled:
			self.debug_unproject = True
		else:
			self.debug_unproject = False
		self.train_writer = tf.summary.FileWriter("tb" + '/train/{}_{}'.format(self.exp_name,time.time()),session.graph)
		self._session = session

		#load_data(self, path)

		import pdb; pdb.set_trace()





	def forwardPass(self):
		N =4
		batch = self.sampler.random_batch()
		img_ph,depth_ph,cam_angle_ph = [np.expand_dims(batch['observations.{}'.format(key)],1) for i, key in enumerate(self.observation_keys[:3])]
		# st()
		depth_ph = np.expand_dims(depth_ph,-1)
		self.model(img_ph,cam_angle_ph,depth_ph,batch_size=self.batch_size,exp_name=self.exp_name,eager=True)
		st()




	def initGraph(self):
		N =4
		img_ph = tf.placeholder(tf.float32, [self.batch_size , 1, N, 84, 84, 3],"images")
		cam_angle_ph = tf.placeholder(tf.float32, [self.batch_size , 1, N, 2],"angles")
		depth_ph = tf.placeholder(tf.float32, [self.batch_size , 1, N, 84, 84],"zmapss")
		position_ph = tf.placeholder(tf.float32, [self.batch_size ,2],"position")
		self._observations_phs = [img_ph,depth_ph,cam_angle_ph,position_ph]
		depth_ph = tf.expand_dims(depth_ph,-1)
		st()
		self.model(img_ph,cam_angle_ph,depth_ph,batch_size=self.batch_size,exp_name=self.exp_name,position=position_ph)
		self.action_predictior =  self.model.action_predictior
		self.detector =  self.model.detector
		# st()
		# self.exp_name = self.model.exp_name





	def load_data(path):
		print("starting loading the data")

		datas = []
		

		for transition in os.listdir(path):
			with open(transition, 'rb') as f:
				data = pickle.loads(f.read())
				dates.append([data["image_observation"], data['depth_observation'], data['cam_angles_observation'],data["actions"]])

		return dates

	def process_data(datas):
		arr1 = datas[0]
		for i in range(1,len(datas)):
			arr2 = datas[i]
			res = np.vstack((arr1, arr2))




	def _get_feed_dict(self,  batch):
		"""Construct TensorFlow feed_dict from sample batch."""
		feed_dict = {}
		# st()
		feed_dict.update({
			self._observations_phs[i]: np.expand_dims(batch['observations.{}'.format(key)],1)
			for i, key in enumerate(self.observation_keys[:4])
		})

		return feed_dict


	def _read_py_function(filename):
			with open(path + '/' + str(filename,encoding ="utf-8" ), 'rb') as f:
				data = pickle.loads(f.read())	
			return data["image_observation"], data['depth_observation'], data['cam_angles_observation'],data["actions"]
		


	def train_epoch(self, epoch,  batches=200):
		training = True
		losses = []
		log_probs = []
		kles = []
		# tempData = {}
		filenames = os.listdir(path)
		filenames = tf.constant(filenames)
		#["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
		#labels = [0, 37, 29, 1, ...]

		dataset = tf.data.Dataset.from_tensor_slices(filenames)
		dataset = dataset.map(lambda filename: tuple(tf.py_func(self._read_py_function, [filename],[tf.uint8,tf.float32,tf.float32,tf.float32])))
		batch_size = dataset//filenames.shape[0]

		batched_dataset = dataset.batch(batch_size)
		iterator = batched_dataset.make_one_shot_iterator()
		next_element = iterator.get_next()


		for batch_idx in range(batches):

			#observation = self.sampler.random_batch()

			# st()
			elem = self._session.run(next_element)

			fd = self._get_feed_dict(elem)



			if batch_idx % self.export_interval == 0 and not self.detector:
				_,summ, loss,pred_view,query_view = self._session.run([self.model.opt,self.model.summ,self.model.loss_,self.model.vis["pred_views"][0],self.model.vis["query_views"][0]],
																feed_dict=fd)
				# st()
				utils.img.imsave01("vis_new/{}_{}_pred.png".format(epoch,batch_idx), pred_view)
				utils.img.imsave01("vis_new/{}_{}_gt.png".format(epoch,batch_idx), query_view)	
			else:
				_, loss,summ = self._session.run([self.model.opt,self.model.loss_,self.model.summ],feed_dict=fd)
			# st()
			step = epoch * batches + batch_idx
			self.train_writer.add_summary(summ,step)
			# utils.img.imsave01("pred_view_{}.png".format(batch_idx), pred_view)
			# utils.img.imsave01("gt_view_{}.png".format(batch_idx), query_view)


			# losses.append(loss)
			# log_probs.append(log_prob)
			# kles.append(kle)

			if batch_idx % self.log_interval == 0:
				print('Train Epoch: {} {}/{}  \tLoss: {:.6f}'.format(
									  epoch,batch_idx,batches,
									  loss ))

 
if __name__ == "__main__":


	# def _read_py_function(filename):
	# 	with open(path + '/' + str(filename,encoding ="utf-8" ), 'rb') as f:
	# 		data = pickle.loads(f.read())	
	# 	return data["image_observation"], data['depth_observation'], data['cam_angles_observation'],data["actions"]
		


	# path = "/projects/katefgroup/yunchu/expert_mug3"

	# filenames = os.listdir(path)
	# filenames = tf.constant(filenames)
	# #["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
	# #labels = [0, 37, 29, 1, ...]

	# dataset = tf.data.Dataset.from_tensor_slices(filenames)
	# dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function, [filename],[tf.uint8,tf.float32,tf.float32,tf.float32])))




	# batched_dataset = dataset.batch(4)

	# iterator = batched_dataset.make_one_shot_iterator()
	# # next_element = iterator.get_next()

	# # print(sess.run(next_element)) 
	# #iterator = dataset.make_initializable_iterator()
	# next_element = iterator.get_next()
	# print(batched_dataset)

	# with tf.Session() as sess:
	# 	while True:

	# 		try:
	# 			elem = sess.run(next_element)s
	# 			print('Success')
	# 		except tf.errors.OutOfRangeError:
	# 			print('End of dataset.')
	# 			break



	mapTrain = MappingTrainer()
	mapTrain.eager_train()