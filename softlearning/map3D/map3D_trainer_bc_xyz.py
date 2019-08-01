from os import path as osp
import numpy as np
import tensorflow as tf
import math
import time
from numbers import Number
from collections import OrderedDict
#from nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain
import ipdb
import time
import os 
import pickle
import argparse
from tensorboardX import SummaryWriter
from test_bc import rollout_and_gather_data
import utils_map as utils
import itertools    
st = ipdb.set_trace



# 	batch = 64

# 	concatendated_state_ph = tf.placeholder(tf.float32, [None, 16])
# 	actions_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2])
# 	out = tf.placeholder(dtype=tf.float32, shape=[None, 2])

# 	lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=16, forget_bias=1.0, state_is_tuple=True)
# 	init_state = lstm_cell.zero_state(batch,dtype=tf.float32) 
# #dagger batch = 1 other 128
# 	out = tf.reshape(concatendated_state_ph,[-1,1,16])

# 	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, out, dtype=tf.float32, time_major=False)

# 	outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

	
# 	out = tf.layers.dense(outputs[-1], 36, activation = tf.nn.relu)
# 	out = tf.layers.dense(out, 24, activation = tf.nn.relu)
# 	out = tf.layers.dense(out, 30, activation = tf.nn.relu)
# 	out = tf.layers.dense(out, 20, activation = tf.nn.relu)
# 	out = tf.layers.dense(out, 10, activation = tf.nn.relu)
# 	out = tf.layers.dense(out, 5, activation = tf.nn.relu)
# 	out = tf.layers.dense(out, 2)
# concatendated_state_ph = tf.placeholder(tf.float32, [None, 16])
# actions_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2])
# out = tf.placeholder(dtype=tf.float32, shape=[None, 2])

# out = tf.layers.dense(concatendated_state_ph, 48, activation = tf.nn.relu)
# out = tf.layers.dense(out, 36, activation = tf.nn.relu)
# out = tf.layers.dense(out, 24, activation = tf.nn.relu)
# out = tf.layers.dense(out, 30, activation = tf.nn.relu)
# out = tf.layers.dense(out, 20, activation = tf.nn.relu)
# out = tf.layers.dense(out, 10, activation = tf.nn.relu)
# out = tf.layers.dense(out, 5, activation = tf.nn.relu)
# out = tf.layers.dense(out, 2)


def _read_py_function(filename):
	with open(path + '/' + str(filename,encoding ="utf-8" ), 'rb') as f:
		data = pickle.loads(f.read())
	#st()
	return np.concatenate((data["observation_with_orientation"],data['state_desired_goal'], data['state_achieved_goal'],\
		data['state_observation'],data['state_desired_goal'],data['state_achieved_goal'],data['proprio_observation'],\
		data['proprio_desired_goal'],data['proprio_achieved_goal']), 0), data["actions"]




def _read_py_function_dg(filename):
	with open(path_dagger + '/' + str(filename,encoding ="utf-8" ), 'rb') as f:
		data = pickle.loads(f.read())

	return np.hstack((data["observation_with_orientation"],data['state_desired_goal'], data['state_achieved_goal'],\
		data['state_observation'],data['state_desired_goal'],data['state_achieved_goal'],data['proprio_observation'],\
		data['proprio_desired_goal'],data['proprio_achieved_goal'])), data["actions"]


def build_model():

	concatendated_state_ph = tf.placeholder(tf.float32, [None, 48])
	actions_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2])
	out = tf.placeholder(dtype=tf.float32, shape=[None, 2])

	out = tf.layers.dense(concatendated_state_ph, 48, activation = tf.nn.relu)
	out = tf.layers.dense(out, 36, activation = tf.nn.relu)
	out = tf.layers.dense(out, 24, activation = tf.nn.relu)
	out = tf.layers.dense(out, 30, activation = tf.nn.relu)
	out = tf.layers.dense(out, 20, activation = tf.nn.relu)
	out = tf.layers.dense(out, 10, activation = tf.nn.relu)
	out = tf.layers.dense(out, 5, activation = tf.nn.relu)
	out = tf.layers.dense(out, 2)
	return concatendated_state_ph, actions_ph, out
def main_dagger(iteration, mesh):
	tf.reset_default_graph()
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
	#st()

	print('loading dagger data')
	#filename = args.path
	concatendated_state_ph, actions_ph, predicted_action_ph = build_model()

	# create loss
	mse = tf.reduce_mean(0.5 * tf.square(actions_ph - predicted_action_ph))

	#set learning rate
	global_step = tf.Variable(0, trainable=False)
	decay_steps = 20000
	lr = tf.train.exponential_decay(0.003,
									global_step,
									decay_steps,
									0.4,
									staircase=True)

	# create optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	#opt  = optimizer.minimize(mse, global_step=global_step)
	#gradients = optimizer.compute_gradients(loss=mse)

	saver = tf.train.Saver()
	#checkpoint_path = "/projects/katefgroup/yunchu/store/" +  mesh + "_dagger"+"/model_"+ str(iteration-1)
	# create saver to save model variables
	if iteration != 0:
		#st()
		saver.restore(sess,"/projects/katefgroup/yunchu/store/" +  mesh + "_dagger"+"/model_"+ str(iteration-1)+"-"+str(iteration-1))
													 


	gradients = tf.gradients(mse, tf.trainable_variables())
	gradients = list(zip(gradients, tf.trainable_variables()))
	# Op to update all variables according to their gradient
	apply_grads = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

	l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
	#setup tensorboard
	




	tf.summary.scalar("mse", mse)

	for var in tf.trainable_variables():
		tf.summary.histogram(var.name, var)


		#check the gradients
	for gradient, variable in gradients:
	   tf.summary.histogram("gradient_norm/" + variable.name, l2_norm(gradient))
	   tf.summary.histogram("gradient/" + variable.name, gradient)
	   tf.summary.histogram("variable_norm/" + variable.name, l2_norm(variable))
	   tf.summary.histogram("variable/" + variable.name, variable)
	#   tf.summary.histogram("variables/" + variable.name, l2_norm(variable))

	#summary_writer.add_scalars(summary_dir,{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)

	merged_summary_op = tf.summary.merge_all()
	#train_op = optimizer.apply_gradients(gradients)
	 
	# with tf.Session() as sess:
	#   summaries_op = tf.summary.merge_all()
	#   #summary_writer = tf.summary.FileWriter("scalar/"+mesh, sess.graph)
	#   for step in itertools.count():
	#     _, summary = sess.run([train_op, summaries_op], feed_dict={concatendated_state_ph: observations, actions_ph: actions})
	#     summary_writer.add_summary(summary, step)






		

	#st()
	batch_size = args.batch_size
	sess.run(tf.global_variables_initializer())
	#path = "/projects/katefgroup/yunchu/expert_mug2"
	# initialize variables
	
	summary_dir = "scalar/"+mesh+"/dagger_iteration_"+str(iteration)
	summary_writer = tf.summary.FileWriter(summary_dir)

	# filenames_a = os.listdir(path)
	# filenames_a = tf.constant(filenames_a)
	# dataset = tf.data.Dataset.from_tensor_slices(filenames_a)



	# dataset_a = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function, [filename],[tf.float32,tf.float32])))
	



	#path = "/projects/katefgroup/yunchu/expert_mug2"
	
	filenames = os.listdir(path_dagger)
	filenames = tf.constant(filenames)
	dataset = tf.data.Dataset.from_tensor_slices(filenames)



	dataset_b = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function_dg, [filename],[tf.float32,tf.float32])))

	dataset=dataset_b

	DATASET_SIZE = (filenames.get_shape().as_list()[0])

	# test_dataset = dataset.take(5000) 
	# train_dataset = dataset.skip(5000)
	#train_dataset = dataset
	train_size = int(0.8 * DATASET_SIZE)
	test_size  = DATASET_SIZE - train_size

	train_dataset = dataset.take(train_size)
	test_dataset = dataset.skip(train_size)



	batches = train_size// batch_size
	# print("Total Data",((filenames_a.get_shape().as_list()[0])+(filenames.get_shape().as_list()[0])))


	for training_step in range(101): #201

		dataset = dataset.shuffle(buffer_size=100)
		#batches = (filenames.get_shape().as_list()[0])// batch_size

		batched_dataset = dataset.batch(batch_size)
		#iterator = batched_dataset.make_one_shot_iterator()
		iterator = batched_dataset.make_initializable_iterator()
		next_element = iterator.get_next()
		sess.run(iterator.initializer)


		# run training
		mse_total = 0.0

		for batch_idx in range(batches):

		#observation = self.sampler.random_batch()

		# st()
			elem = sess.run(next_element)
			observations = elem[0]
			actions = elem[1]
			#fd = _get_feed_dict(elem)
			#st()
			_,output_pred_run, mse_run, merged_summary_op_run = sess.run([apply_grads,predicted_action_ph, mse, merged_summary_op], feed_dict={concatendated_state_ph: observations, actions_ph: actions})
			summary_writer.add_summary(merged_summary_op_run, training_step*batches+1)
			mse_total += mse_run

		
		#writer.add_scalars('scalar/train',{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)
		#st()
		#summary_writer.add_scalars(summary_dir,{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)
		
		if training_step % 10 == 0:
			print('{0:04d} mse: {1:.6f}'.format(training_step, mse_total))
			print((output_pred_run - actions).mean())
			print((output_pred_run - actions).sum())

			test_dataset = test_dataset.batch(test_size)
			#iterator = batched_dataset.make_one_shot_iterator()
			iterator1 = test_dataset.make_initializable_iterator()
			next_element = iterator1.get_next()
			sess.run(iterator1.initializer)

			elem = sess.run(next_element)
			observations = np.reshape(elem[0],[-1,48])
			actions = np.reshape(elem[1],[-1,2])
			#fd = _get_feed_dict(elem)
			output_pred_run, mse_run = sess.run([predicted_action_ph, mse], feed_dict={concatendated_state_ph: observations, actions_ph: actions})



			print('{0:04d} Eva_mse: {1:.6f}'.format(training_step, mse_run))
			print((output_pred_run - actions).mean())
			print((output_pred_run - actions).sum())



			# print the mse every so often
		#if training_step % 50 == 0:
			#store_path = "store/" + object_name + "_dagger" + "/model.ckpt"
	#st()
	store_path = "/projects/katefgroup/yunchu/store/" +  object_name + "_dagger"+ "/model_"+ str(iteration)  #TODO store the last, change maybe to store the best 
	#saver.save(sess, "store/model.ckpt")
	print(store_path)
	saver.save(sess, store_path, global_step = iteration)
#tf.reset_default_graph()
	#rollout the current agent

def main_dagger_without(iteration, mesh):
	tf.reset_default_graph()
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
	#st()

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
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	#opt  = optimizer.minimize(mse, global_step=global_step)
	#gradients = optimizer.compute_gradients(loss=mse)

	saver = tf.train.Saver()
	#checkpoint_path = "/projects/katefgroup/yunchu/store/" +  mesh + "_dagger"+"/model_"+ str(iteration-1)
	# create saver to save model variables
	if iteration != 0:
		#st()
		saver.restore(sess,"/projects/katefgroup/yunchu/store/" +  mesh + "_dagger"+"/model_"+ str(iteration-1)+"-"+str(iteration-1))
													 


	gradients = tf.gradients(mse, tf.trainable_variables())
	gradients = list(zip(gradients, tf.trainable_variables()))
	# Op to update all variables according to their gradient
	apply_grads = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

	l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
	#setup tensorboard
	




	tf.summary.scalar("mse", mse)

	for var in tf.trainable_variables():
		tf.summary.histogram(var.name, var)


		#check the gradients
	for gradient, variable in gradients:
	   tf.summary.histogram("gradient_norm/" + variable.name, l2_norm(gradient))
	   tf.summary.histogram("gradient/" + variable.name, gradient)
	   tf.summary.histogram("variable_norm/" + variable.name, l2_norm(variable))
	   tf.summary.histogram("variable/" + variable.name, variable)
	#   tf.summary.histogram("variables/" + variable.name, l2_norm(variable))

	#summary_writer.add_scalars(summary_dir,{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)

	merged_summary_op = tf.summary.merge_all()
	#train_op = optimizer.apply_gradients(gradients)
	 
	# with tf.Session() as sess:
	#   summaries_op = tf.summary.merge_all()
	#   #summary_writer = tf.summary.FileWriter("scalar/"+mesh, sess.graph)
	#   for step in itertools.count():
	#     _, summary = sess.run([train_op, summaries_op], feed_dict={concatendated_state_ph: observations, actions_ph: actions})
	#     summary_writer.add_summary(summary, step)






		

	#st()
	batch_size = args.batch_size
	sess.run(tf.global_variables_initializer())
	#path = "/projects/katefgroup/yunchu/expert_mug2"
	# initialize variables
	
	summary_dir = "scalar/"+mesh+"/dagger_iteration_"+str(iteration)
	summary_writer = tf.summary.FileWriter(summary_dir)

	# filenames_a = os.listdir(path)
	# filenames_a = tf.constant(filenames_a)
	# dataset = tf.data.Dataset.from_tensor_slices(filenames_a)



	#dataset_a = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function, [filename],[tf.float32,tf.float32])))
	



	#path = "/projects/katefgroup/yunchu/expert_mug2"
	st()
	filenames = os.listdir(path_dagger)
	filenames = tf.constant(filenames)
	dataset = tf.data.Dataset.from_tensor_slices(filenames)



	dataset_b = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function_dg, [filename],[tf.float32,tf.float32])))

	#dataset=dataset_a.concatenate(dataset_b)



	#batches = ((filenames_a.get_shape().as_list()[0])+(filenames.get_shape().as_list()[0]))// batch_size
	batches = ((filenames.get_shape().as_list()[0]))// batch_size


	for training_step in range(10): #201

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
			st()
			_,output_pred_run, mse_run, merged_summary_op_run = sess.run([apply_grads,predicted_action_ph, mse, merged_summary_op], feed_dict={concatendated_state_ph: observations, actions_ph: actions})
			summary_writer.add_summary(merged_summary_op_run, training_step*batches+1)

		
		#writer.add_scalars('scalar/train',{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)
		#st()
		#summary_writer.add_scalars(summary_dir,{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)
		
		if training_step % 10 == 0:
			print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
			print((output_pred_run - actions).mean())
			print((output_pred_run - actions).sum())
			# print the mse every so often
		#if training_step % 50 == 0:
			#store_path = "store/" + object_name + "_dagger" + "/model.ckpt"
	#st()
	store_path = "/projects/katefgroup/yunchu/store/" +  object_name + "_dagger"+ "/model_"+ str(iteration)  #TODO store the last, change maybe to store the best 
	#saver.save(sess, "store/model.ckpt")
	saver.save(sess, store_path, global_step = iteration)
#tf.reset_default_graph()
	#rollout the current agent



def test():
	tf.reset_default_graph()
	gpu_options = tf.GPUOptions(allow_growth=True)
	session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	tf.keras.backend.set_session(session)
	sess = tf.keras.backend.get_session()

	print('loading dagger data')

	concatendated_state_ph, actions_ph, predicted_action_ph = build_model()

	# create loss
	mse = tf.reduce_mean(0.5 * tf.square(actions_ph - predicted_action_ph))

	#set learning rate
	global_step = tf.Variable(0, trainable=False)
	decay_steps = 20000
	lr = tf.train.exponential_decay(0.004,
	global_step,
	decay_steps,
	0.5,
	staircase=True)

	# create optimizer
	opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(mse, global_step=global_step)
	# initialize variables
	#sess.run(tf.global_variables_initializer())
	# create saver to save model variables
	checkpoint_path = "/projects/katefgroup/yunchu/store/" +  "mug2" + "_dagger"+ "/model_0-0"
	saver = tf.train.import_meta_graph(checkpoint_path+".meta")

	saver.restore(sess,tf.train.latest_checkpoint("/projects/katefgroup/yunchu/store/" +  "mug2" + "_dagger"))

	sess.run(tf.global_variables_initializer())
	batch_size = 64
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
	DATASET_SIZE = (filenames_a.get_shape().as_list()[0])+(filenames.get_shape().as_list()[0])

	# test_dataset = dataset.take(5000) 
	# train_dataset = dataset.skip(5000)
	#train_dataset = dataset
	train_size = int(0.8 * DATASET_SIZE)
	test_size  = DATASET_SIZE - train_size

	train_dataset = dataset.take(train_size)
	test_dataset = dataset.skip(train_size)

	test_dataset = test_dataset.batch(test_size)
	#iterator = batched_dataset.make_one_shot_iterator()
	iterator1 = test_dataset.make_initializable_iterator()
	next_element = iterator1.get_next()
	sess.run(iterator1.initializer)

	elem = sess.run(next_element)
	observations = np.reshape(elem[0],[-1,16])
	actions = np.reshape(elem[1],[-1,2])
	#fd = _get_feed_dict(elem)
	output_pred_run, mse_run = sess.run([predicted_action_ph, mse], feed_dict={concatendated_state_ph: observations, actions_ph: actions})



	print(mse_run)
	print((output_pred_run - actions).mean())
	print((output_pred_run - actions).sum())


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
	decay_steps = 2000
	lr = tf.train.exponential_decay(0.003,
									global_step,
									decay_steps,
									0.5,
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




	for training_step in range(101):

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
		summary_writer.add_scalars(summary_dir,{'mse_run':mse_run, 'avg_error': (output_pred_run - actions).mean()}, training_step)
		if training_step % 10 == 0:
			print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
			print((output_pred_run - actions).mean())
			print((output_pred_run - actions).sum())
			# print the mse every so often
		if training_step % 50 == 0:
			store_path= "store/" + object_name + "/model.ckpt"
			#saver.save(sess, "store/model.ckpt")
			saver.save(sess, store_path)


def dagger(number_iterations, mesh):
	for iteration in range(number_iterations):
		#combine old experience and the expertactions on the sample trajectories to dataset D
		# and train bc agent on D
		#main_dagger(iteration, mesh)
		main_dagger(iteration, mesh)
		#test()
		#sample trajectories and store the experts actions
		max_rollouts = 25 #300 #how many starting conditions to sample and to roll out
		succes_rate = rollout_and_gather_data(max_rollouts, mesh, iteration)
		#main_dagger_without(iteration, mesh)


		print("done with iteration ", iteration," on object", mesh, "with succes rate", succes_rate)



if __name__ == '__main__':
	base_path = "/projects/katefgroup/yunchu/"
	#expert_list = ["expert_mug1"]
	#object_list = ["bowl2","car2","hat1","mug2","boat","can1","car3","hat2","mug3","bowl1 ","car1","car4","mug1"]
	object_list = ["mug1"]#, "car2","hat1","boat","can1","car3", "bowl2"]
	for object_name in object_list:
		print("expert ", object_name)
		path = base_path+"expert_"+object_name
		path_dagger = base_path+"dagger_"+object_name
		#summary_writer = SummaryWriter()
		#main()
		#main_dagger()
		number_iterations = 10 #number of dagger iterations
		dagger(number_iterations, object_name)




