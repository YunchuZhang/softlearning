import importlib
import sys
import ipdb
st = ipdb.set_trace
import tensorflow as tf
import sys
from examples.map3D.main_map3d import ExperimentRunner
from examples.instrument import change_env_to_use_correct_mesh
from softlearning.map3D.test_bc import rollout_and_gather_data
import time
import os
import pickle
import numpy as np
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
		
def test():
	tf.reset_default_graph()
	gpu_options = tf.GPUOptions(allow_growth=True)
	session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	tf.keras.backend.set_session(session)
	sess = tf.keras.backend.get_session()

	print('loading ')

	checkpoint_path = "/projects/katefgroup/yunchu/store/" +  "hat1" + "_dagger"
	saver = tf.train.import_meta_graph(checkpoint_path+ "/model_0"+"-0"+".meta")
	print("i am reloading", tf.train.latest_checkpoint(checkpoint_path))
	saver.restore(sess,tf.train.latest_checkpoint(checkpoint_path))
	
	losses = []
	log_probs = []
	kles = []
	# tempData = {}
	path = "/projects/katefgroup/yunchu/dagger_minimalhat1"
	filenames = os.listdir(path)
	sampleBuffer = SampleBuffer(path)
	for transition_name in filenames:
		sampleBuffer.load(transition_name)
	print("finished loading")
	
	images,zmapss,angles,goal_centroid,position= sampleBuffer.sample(15)
	graph = tf.get_default_graph()
	prediction = graph.get_tensor_by_name('Variables/main/action_predictor/final_result/BiasAdd:0')
	action = sess.run([prediction], feed_dict={'images:0': np.reshape(images,(15,1,4,84,84,3)), \
					'zmapss:0': np.reshape(zmapss,(15,1,4,84,84)),'angles:0':np.reshape(angles,(15,1,4,2)),\
					'goal_centroid:0':np.reshape(goal_centroid,(15,1,5)),\
					'position:0':np.reshape(position,(15,1,2))})

	print('action_predictor',action)





example_module_name = "examples.map3D"

#list_of_experts = ["expert_boat","expert_bowl1","expert_can1","expert_car1","expert_car2","expert_mug3"]
list_of_experts = ["hat1"]

example_argv = ('--universe=gym', '--checkpoint-frequency=0', '--domain=SawyerPushAndReachEnvEasy', '--task=v0', '--trial-gpus=1', '--exp-name=test', '--replay_pool=SimpleReplayPoolTemp', '--algorithm=SAC', '--mesh=expert_mug3')
for expert in list_of_experts:
	#expert_name = "--mesh=dagger_" + expert
	expert_name = "--mesh=dagger_minimal" + expert

	example_argv = list(example_argv)
	example_argv[8] = expert_name
	example_argv = tuple(example_argv)


	example_module = importlib.import_module(example_module_name)

	example_args = example_module.get_parser().parse_args(example_argv)
	# print("example_args",example_args)
	variant_spec = example_module.get_variant_spec(example_args)
	#st()
	eager = False
	#action_predictor = True
	er = ExperimentRunner()

	if eager:
		tf.enable_eager_execution()

	er._setup("rl_new_reach",variant_spec,example_args.mesh ,eager)

	# er._setup("rl_new_reach_detect",variant_spec,eager)

	er._build()
	#st()
	#test()
	#er._train()
	number_iterations = 5 #number of dagger iterations

	for iteration in range(number_iterations):
		
		#combine old experience and the expertactions on the sample trajectories to dataset D
		# and train bc agent on D
		#main_dagger(iteration, mesh)
		#main_dagger(iteration, mesh) #expert,epoch,iteration)
		#test()
		training_epochs = 3
		er.algorithm.train_epoch(expert,training_epochs,iteration)
		#sample trajectories and store the experts actions
		#tf.reset_default_graph()
		max_rollouts = 30 #300 #how many starting conditions to sample and to roll out
		succes_rate = rollout_and_gather_data(max_rollouts, expert, iteration)
		#main_dagger_without(iteration, mesh)
	

		print("done with iteration ", iteration," on object", expert, "with succes rate", succes_rate)




	print("done with expert", expert)

