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
import argparse
from tensorboardX import SummaryWriter
from test_bc import rollout_and_gather_data
import utils_map as utils

st = ipdb.set_trace



def _read_py_function(filename):
	with open(path + '/' + str(filename,encoding ="utf-8" ), 'rb') as f:
		data = pickle.loads(f.read())
	#st()
	return np.concatenate((data["observation_with_orientation"],data['state_desired_goal']), 0), data["actions"]

def _read_py_function_dg(filename):
	with open(path_dagger + '/' + str(filename,encoding ="utf-8" ), 'rb') as f:
		data = pickle.loads(f.read())
	#st()
	return np.concatenate((data["observation_with_orientation"],data['state_desired_goal']), 0), data["actions"]


def build_model():

	concatendated_state_ph = tf.placeholder(tf.float32, [None, 16])
	actions_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2])
	out = tf.placeholder(dtype=tf.float32, shape=[None, 2])

	out = tf.layers.dense(concatendated_state_ph, 128, activation = tf.nn.relu)
	out = tf.layers.dense(out, 64, activation = tf.nn.relu)
	out = tf.layers.dense(out, 32, activation = tf.nn.relu)
	out = tf.layers.dense(out, 2)
	return concatendated_state_ph, actions_ph, out
def main_dagger():

	gpu_options = tf.GPUOptions(allow_growth=True)
	session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	tf.keras.backend.set_session(session)
	sess = tf.keras.backend.get_session()


	#sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
	parser = argparse.ArgumentParser()
	# parser.add_argument('path', type=str)
	# parser.add_argument('envname', type=str)
	# parser.add_argument("--max_timesteps", type=int)
	parser.add_argument("--batch_size", type=int, default = 64)
	#parser.add_argument('--num_rollouts', type=int, default=200,
	help=('Number of expert roll outs')
	args = parser.parse_args()

	print('loading dagger data')
	#filename = args.path
	concatendated_state_ph, actions_ph, predicted_action_ph = build_model()

	# create loss
	mse = tf.reduce_mean(0.5 * tf.square(actions_ph - predicted_action_ph))

	#set learning rate
	global_step = tf.Variable(0, trainable=False)
	decay_steps = 500
	lr = tf.train.exponential_decay(0.003,
                                    global_step,
                                    decay_steps,
                                    0.96,
                                    staircase=True)

	# create optimizer
	opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(mse, global_step=global_step)

	# initialize variables
	sess.run(tf.global_variables_initializer())
	# create saver to save model variables
	saver = tf.train.Saver()
	#st()
	batch_size = args.batch_size
	#path = "/projects/katefgroup/yunchu/expert_mug2"

	filenames_a = os.listdir(path)
	filenames_a = tf.constant(filenames_a)
	dataset = tf.data.Dataset.from_tensor_slices(filenames_a)



	dataset_a = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function, [filename],[tf.float32,tf.float32])))
	



	#path = "/projects/katefgroup/yunchu/expert_mug2"
	
	filenames = os.listdir(path_dagger)
	filenames = tf.constant(filenames)
	dataset = tf.data.Dataset.from_tensor_slices(filenames)



	dataset_b = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function_dg, [filename],[tf.float32,tf.float32])))

	dataset=dataset_a.concatenate(dataset_b)



	batches = ((filenames_a.get_shape().as_list()[0])+(filenames.get_shape().as_list()[0]))// batch_size


	for training_step in range(201):

		dataset = dataset.shuffle(buffer_size=100)
		#batches = (filenames.get_shape().as_list()[0])// batch_size

		batched_dataset = dataset.batch(batch_size)
		#iterator = batched_dataset.make_one_shot_iterator()
		iterator = batched_dataset.make_initializable_iterator()
		next_element = iterator.get_next()
		sess.run(iterator.initializer)


		# run training
		

		for batch_idx in range(batches):

		#observation = self.sampler.random_batch()

		# st()
			elem = sess.run(next_element)
			observations = elem[0]
			actions = elem[1]
			#fd = _get_feed_dict(elem)
			_,output_pred_run, mse_run = sess.run([opt,predicted_action_ph, mse], feed_dict={concatendated_state_ph: observations, actions_ph: actions})


		#writer.add_scalars('scalar/train',{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)
		writer.add_scalars(summary_dir,{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)
		if training_step % 10 == 0:
			print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
			print((output_pred_run - actions).mean())
			print((output_pred_run - actions).sum())
			# print the mse every so often
		if training_step % 50 == 0:
			store_path = "store/" + object_name + "_dagger" + "/model.ckpt"
			#saver.save(sess, "store/model.ckpt")
			saver.save(sess, store_path)



def main():
	#sess = tf.Session()

	gpu_options = tf.GPUOptions(allow_growth=True)
	session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	tf.keras.backend.set_session(session)
	sess = tf.keras.backend.get_session()


	#sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
	parser = argparse.ArgumentParser()
	# parser.add_argument('path', type=str)
	# parser.add_argument('envname', type=str)
	# parser.add_argument("--max_timesteps", type=int)
	parser.add_argument("--batch_size", type=int, default = 64)
	#parser.add_argument('--num_rollouts', type=int, default=200,
	help=('Number of expert roll outs')
	args = parser.parse_args()

	print('loading expert data')
	#filename = args.path
	concatendated_state_ph, actions_ph, predicted_action_ph = build_model()

	# create loss
	mse = tf.reduce_mean(0.5 * tf.square(actions_ph - predicted_action_ph))

	#set learning rate
	global_step = tf.Variable(0, trainable=False)
	decay_steps = 500
	lr = tf.train.exponential_decay(0.003,
                                    global_step,
                                    decay_steps,
                                    0.96,
                                    staircase=True)

	# create optimizer
	opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(mse, global_step=global_step)

	# initialize variables
	sess.run(tf.global_variables_initializer())
	# create saver to save model variables
	saver = tf.train.Saver()


	#path = "/projects/katefgroup/yunchu/expert_mug2"
	batch_size = args.batch_size
	filenames = os.listdir(path)
	filenames = tf.constant(filenames)
	dataset = tf.data.Dataset.from_tensor_slices(filenames)



	dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function, [filename],[tf.float32,tf.float32])))
	batches = (filenames.get_shape().as_list()[0])// batch_size




	for training_step in range(201):

		dataset = dataset.shuffle(buffer_size=100)
		#batches = (filenames.get_shape().as_list()[0])// batch_size

		batched_dataset = dataset.batch(batch_size)
		#iterator = batched_dataset.make_one_shot_iterator()
		iterator = batched_dataset.make_initializable_iterator()
		next_element = iterator.get_next()
		sess.run(iterator.initializer)


		# run training
		

		for batch_idx in range(batches):

		#observation = self.sampler.random_batch()

		# st()
			elem = sess.run(next_element)
			observations = elem[0]
			actions = elem[1]
			#fd = _get_feed_dict(elem)
			_,output_pred_run, mse_run = sess.run([opt,predicted_action_ph, mse], feed_dict={concatendated_state_ph: observations, actions_ph: actions})


		#writer.add_scalars('scalar/train',{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)
		writer.add_scalars(summary_dir,{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)
		if training_step % 10 == 0:
			print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
			print((output_pred_run - actions).mean())
			print((output_pred_run - actions).sum())
			# print the mse every so often
		if training_step % 50 == 0:
			store_path = "store/" + object_name + "/model.ckpt"
			#saver.save(sess, "store/model.ckpt")
			saver.save(sess, store_path)


def dagger(number_iterations, mesh):
	for iteration in range(number_iterations):
		#sample trajectories and store the experts actions
		max_rollouts = 50
		rollout_and_gather_data(max_rollouts, mesh)
		#combine old experience and the expertactions on the sample trajectories to dataset D

		#train classifier on D


if __name__ == '__main__':
	base_path = "/projects/katefgroup/yunchu/"
	#expert_list = ["expert_mug1"]
	#object_list = ["bowl2","car2","hat1","mug2","boat","can1","car3","hat2","mug3","bowl1 ","car1","car4","mug1"]
	object_list = ["bowl2","car2","hat1","mug2","boat","can1","car3"]
	for object_name in object_list:
		print("expert ", object_name)
		path = base_path+"expert_"+object_name
		path_dagger = base_path+"dagger_"+object_name
		writer = SummaryWriter()
		summary_dir = "scalar/"+object_name
		writer = SummaryWriter(log_dir=summary_dir)
		#main()
		main_dagger()




