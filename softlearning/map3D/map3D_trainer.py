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
			session):
		# session = tf.keras.backend.get_session()
		# st()
		self._n_initial_exploration_steps = variant['algorithm_params']['kwargs']["n_initial_exploration_steps"]

		self._n_initial_exploration_steps 

		self.sampler = sampler
		
		self.batch_size =batch_size
		self.observation_keys = observation_keys
		self._initial_exploration_hook(training_environment, initial_exploration_policy, pool)        
		self.pool = pool
		self.log_interval = 10
		self.export_interval = 100
		self.model = map3D
		self.initGraph()
		# st()

		self._session = session







	def initGraph(self):
		N =4
		img_ph = tf.placeholder(tf.float32, [self.batch_size , 1, N, 84, 84, 4],"images")
		cam_angle_ph = tf.placeholder(tf.float32, [self.batch_size , 1, N, 2],"angles")
		depth_ph = tf.placeholder(tf.float32, [self.batch_size , 1, N, 84, 84],"zmapss")
		self._observations_phs = [img_ph,depth_ph,cam_angle_ph]
		depth_ph = tf.expand_dims(depth_ph,-1)
		self.model(img_ph,cam_angle_ph,depth_ph,batch_size=self.batch_size,exp_name="rl_new_reach")


	def _initial_exploration_hook(self, env, initial_exploration_policy, pool):
		print("starting initial exploration")
		if self._n_initial_exploration_steps < 1: return

		if not initial_exploration_policy:
			raise ValueError(
				"Initial exploration policy must be provided when"
				" n_initial_exploration_steps > 0.")

		self.sampler.initialize(env, initial_exploration_policy, pool)
		t =time.time()
		while pool.size < self._n_initial_exploration_steps:
			self.sampler.sample()
			print(time.time()-t, pool.size)

		print("finished initial exploration")

	def _get_feed_dict(self,  batch):
		"""Construct TensorFlow feed_dict from sample batch."""
		feed_dict = {}

		feed_dict.update({
			self._observations_phs[i]: np.expand_dims(batch['observations.{}'.format(key)],1)
			for i, key in enumerate(self.observation_keys[:3])
		})

		return feed_dict

	def train_epoch(self, epoch,  batches=500):
		training = True
		losses = []
		log_probs = []
		kles = []
		# tempData = {}
		for batch_idx in range(batches):
			observation = self.sampler.random_batch()
			fd = self._get_feed_dict(observation)

			if batch_idx % self.export_interval == 0:
				_, loss,pred_view,query_view = self._session.run([self.model.opt,self.model.loss_,self.model.vis["pred_views"][0],self.model.vis["query_views"][0]],
																feed_dict=fd)
				# st()
				utils.img.imsave01("vis/pred_view_epoch_{}_{}.png".format(epoch,batch_idx), pred_view)
				utils.img.imsave01("vis/gt_view_epoch_{}_{}.png".format(epoch,batch_idx), query_view)			
			else:
				_, loss = self._session.run([self.model.opt,self.model.loss_],feed_dict=fd)
			# st()

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
	mapTrain = MappingTrainer()
	mapTrain.train()
