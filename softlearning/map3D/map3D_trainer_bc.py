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

import time



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


class SampleBuffer(object):
	def __init__(self, path):
		self.storage = []
		self.path = path

	def __len__(self):
		return

	# Expects tuples of (image_observation, depth_observation, cam_angles_observation, state_desired_goal, actions)
	def add(self, data):
		self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		img_obs, depth_obs, cam_angles, sdg, actions = [], [], [], [], []

		for i in ind: 
			s, s2, a, r, d = self.storage[i]
			img_obs.append(np.array(s, copy=False))
			depth_obs.append(np.array(s2, copy=False))
			cam_angles.append(np.array(a, copy=False))
			sdg.append(np.array(r, copy=False))
			actions.append(np.array(d, copy=False))

		return [np.array(img_obs), 
			np.array(depth_obs), 
			np.array(cam_angles), 
			np.array(sdg), 
			np.array(actions)]

	def save(self, filename):
		np.save("./buffers/"+filename+".npy", self.storage)

	def load(self, filename):
		#with open(self.path + '/' + str(filename,encoding ="utf-8" ), 'rb') as f:
		with open(self.path + '/' + filename, 'rb') as f:
			data = pickle.loads(f.read())
		#self.storage. = ( data["image_observation"], data['depth_observation'], data['cam_angles_observation'],data['state_desired_goal'],data["actions"]) #as in the orig implementatino

		self.storage.append(( data["image_observation"], data['depth_observation'], data['cam_angles_observation'],data['state_desired_goal'],data["actions"]))
		





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
			expert_name,
			session):
		# session = tf.keras.backend.get_session()
		# st()
		self._n_initial_exploration_steps = variant['algorithm_params']['kwargs']["n_initial_exploration_steps"]

		self.eager_enabled = eager_enabled

		self._n_initial_exploration_steps 

		self.sampler = sampler

		#self.path = "/projects/katefgroup/yunchu/expert_mug3"
		self.expert_name = expert_name
		#st()
		self.path = os.path.join("/projects/katefgroup/yunchu/",self.expert_name)
		filenames = os.listdir(self.path)
		filenames = tf.constant(filenames)
		#["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
		#labels = [0, 37, 29, 1, ...]

		dataset = tf.data.Dataset.from_tensor_slices(filenames)

		self.batch_size = 15 #changed from 4
		#self.batches = 200 

		self.batch = (filenames.get_shape().as_list()[0])// self.batch_size
		
		#self.batch_size =batch_size
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
		#self.train_writer = tf.summary.FileWriter("tb" + '/train/{}_self.exp_name_{}'.format(self.exp_name,time.time()),session.graph)
		self._session = session

		#load_data(self, path)

		#import pdb; pdb.set_trace()



			# expert_data = {'image_observation': np.array(replay_pool.fields["observations.image_observation"][counter]),
			# 	'depth_observation': np.array(replay_pool.fields["observations.depth_observation"][counter]),
			# 	'cam_angles_observation':np.array(replay_pool.fields["observations.cam_angles_observation"][counter]),
			# 	'actions':expert_actions[counter],
			# 	'rewards':np.array(replay_pool.fields["rewards"][counter]),
			# 	'observation_with_orientation':np.array(replay_pool.fields["observations.observation_with_orientation"][counter]),
			# 	'state_desired_goal':np.array(replay_pool.fields["observations.state_desired_goal"][counter]),
			# 	'terminals':np.array(replay_pool.fields["terminals"][counter]),
			# 	'desired_goal': np.array(replay_pool.fields["observations.desired_goal"][counter]),
			# 	'achieved_goal': np.array(replay_pool.fields["observations.achieved_goal"][counter]),
			# 	'state_observation': np.array(replay_pool.fields["observations.state_observation"][counter]),
			# 	'state_achieved_goal': np.array(replay_pool.fields["observations.state_achieved_goal"][counter]),
			# 	'proprio_observation': np.array(replay_pool.fields["observations.proprio_observation"][counter]),
			# 	'proprio_desired_goal': np.array(replay_pool.fields["observations.proprio_desired_goal"][counter]),
			# 	'proprio_achieved_goal':np.array(replay_pool.fields["observations.proprio_achieved_goal"][counter])}

	def forwardPass(self):
		N =4
		batch = self.sampler.random_batch()
		img_ph,depth_ph,cam_angle_ph = [np.expand_dims(batch['observations.{}'.format(key)],1) for i, key in enumerate(self.observation_keys[:3])]
		# st()
		depth_ph = np.expand_dims(depth_ph,-1)
		self.model(img_ph,cam_angle_ph,depth_ph,batch_size=self.batch_size,exp_name=self.exp_name,eager=True)
		#st()




	def initGraph(self):
		N =4
		img_ph = tf.placeholder(tf.float32, [self.batch_size , 1, N, 84, 84, 3],"images")
		cam_angle_ph = tf.placeholder(tf.float32, [self.batch_size , 1, N, 2],"angles")
		depth_ph = tf.placeholder(tf.float32, [self.batch_size , 1, N, 84, 84],"zmapss")
		goal_ph =  tf.placeholder(tf.float32, [self.batch_size, 1,5], "goal_centroid")
		action_ph = tf.placeholder(tf.float32, [self.batch_size ,1,2],"position")
		self._observations_phs = [img_ph,depth_ph,cam_angle_ph,goal_ph,action_ph]
		depth_ph = tf.expand_dims(depth_ph,-1)
		
		self.model(img_ph,cam_angle_ph,depth_ph, goal_ph,batch_size=self.batch_size,exp_name=self.exp_name,position=action_ph)
		#st()
		self.action_predictor =  self.model.action_predictor
		# st()
		# self.exp_name = self.model.exp_name





	# def load_data(path):
	# 	print("starting loading the data")

	# 	datas = []
		

	# 	for transition in os.listdir(path):
	# 		with open(transition, 'rb') as f:
	# 			data = pickle.loads(f.read())
	# 			dates.append([data["image_observation"], data['depth_observation'], data['cam_angles_observation'],data["actions"]])

	# 	return dates


	def _get_feed_dict(self,  batch):
		"""Construct TensorFlow feed_dict from sample batch."""
		feed_dict = {}
		#st()
		feed_dict.update({
			self._observations_phs[i]: np.expand_dims(batch[i],1)
			for i in range(5)
		})

		return feed_dict


	def _read_py_function(self, filename):
		#st()
		with open(self.path + '/' + str(filename,encoding ="utf-8" ), 'rb') as f:
			data = pickle.loads(f.read())
			
		return data["image_observation"], data['depth_observation'], data['cam_angles_observation'],data['state_desired_goal'],data["actions"]
		


	def train_epoch(self, expert,epoch,iteration):
		training = True
		losses = []
		log_probs = []
		kles = []
		# tempData = {}
		filenames = os.listdir(self.path)
		sampleBuffer = SampleBuffer(self.path)
		for transition_name in filenames:
			sampleBuffer.load(transition_name)
		print("finished loading")
		#["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
		#labels = [0, 37, 29, 1, ...]
		#self.train_writer = tf.summary.FileWriter("tb" + '/train/{}_'+expert+'_{}_{}'.format(self.exp_name,epoch,time.time()),self._session.graph)
		self.train_writer = tf.summary.FileWriter("tb" + '/train/'+expert+"_"+str(2)+"/dagger_iteration_"+str(iteration),self._session.graph)

		#checkpoint_path = "/projects/katefgroup/yunchu/store/" +  mesh + "_dagger"+"/model_"+ str(iteration-1)
		# create saver to save model variables
		if iteration != 0:
			st()
			checkpoint_path = "/projects/katefgroup/yunchu/store/" +  expert + "_dagger"
			saver = tf.train.import_meta_graph(checkpoint_path+ "/model_"+ str(iteration-1)+"-"+str(iteration-1)+".meta")
			print("i am reloading", tf.train.latest_checkpoint(checkpoint_path))
			saver.restore(self._session,tf.train.latest_checkpoint(checkpoint_path))
		else:
			saver = tf.train.Saver()

		#st()
		#self.batches = (filenames.get_shape().as_list()[0])// self.batch_size
		self.batches = len(filenames) // self.batch_size
		self.batches = 2 #to speed it up
		for training_step in range(epoch):

			starting_time = time.time()
			for batch_idx in range(self.batches):

				#elem = self._session.run(sampleBuffer.sample(self.batch_size))

				fd = self._get_feed_dict(sampleBuffer.sample(self.batch_size))

				#??no use for image?

				
				# if batch_idx % self.export_interval == 0 and not self.action_predictor:
				# 	_,summ, loss,pred_view,query_view = self._session.run([self.model.opt,self.model.summ,self.model.loss_,self.model.vis["pred_views"][0],self.model.vis["query_views"][0]],
				# 													feed_dict=fd)
				# 	# st()
				# 	#utils.img.imsave01("vis_new/{}_{}_pred.png".format(epoch,batch_idx), pred_view)
				# 	#utils.img.imsave01("vis_new/{}_{}_gt.png".format(epoch,batch_idx), query_view)	
				# else:
				
				# _, loss,summ = self._session.run([self.model.opt,self.model.loss_,self.model.summ],feed_dict=fd)
				_, loss = self._session.run([self.model.opt,self.model.loss_],feed_dict=fd)
				# st()


				# step = epoch * self.batches + batch_idx
				# self.train_writer.add_summary(summ,step)


				# utils.img.imsave01("pred_view_{}.png".format(batch_idx), pred_view)
				# utils.img.imsave01("gt_view_{}.png".format(batch_idx), query_view)


				# losses.append(loss)
				# log_probs.append(log_prob)
				# kles.append(kle)

				if batch_idx % self.log_interval == 0:
					print('Train Epoch: {} {}/{}  \tLoss: {:.6f} \tEpochs runs since: {}'.format( training_step,batch_idx,self.batches,loss, time.time()-starting_time ))


		store_path = "/projects/katefgroup/yunchu/store/" +  expert + "_dagger"+ "/model_"+ str(iteration)  #TODO store the last, change maybe to store the best 
		#saver.save(sess, "store/model.ckpt")
		print(store_path)
		saver.save(self._session, store_path, global_step = iteration)

 
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