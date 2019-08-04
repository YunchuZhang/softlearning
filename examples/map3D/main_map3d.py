import os
import copy
import glob
import pickle
import sys
import tensorflow as tf
from ray import tune
from softlearning.samplers.simple_sampler import SimpleSampler
from softlearning.environments.utils import get_environment_from_params,get_environment_from_params_custom
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.map3D.fig import Config
from softlearning.map3D import utils_map as utils
import os.path as path
from softlearning.misc.utils import set_seed, initialize_tf_variables
from softlearning.map3D.map3D_trainer_bc import MappingTrainer as bulledtPushTrainer #import from the file for bc cloning
from examples.instrument import run_example_local
import ipdb 
import numpy as np
st = ipdb.set_trace

class ExperimentRunner():
	def _setup(self,exp_name, variant, expert_name,eager=False):
		# st()
		set_seed(np.random.randint(0, 10000))
		self.eager = eager

		self._variant = variant

		# self.detector = detector

		gpu_options = tf.GPUOptions(allow_growth=True)
		session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		tf.keras.backend.set_session(session)
		self._session = tf.keras.backend.get_session()
		self.exp_name = exp_name
		self.train_generator = None
		self._built = False
		#import pdb; pdb.set_trace()
		self.expert_name = expert_name

	def _stop(self):
		tf.reset_default_graph()
		tf.keras.backend.clear_session()

	def map3d_setup(self,sess,map3D=None):
		# coord = tf.train.Coordinator()
		# threads = tf.train.start_queue_runners(coord=coord, sess=sess)

		#must come after the queue runners
		# if const.DEBUG_NAN:
		#     self.sess = debug.LocalCLIDebugWrapperSession(self.sess)
		#     self.sess.add_tensor_filter("has_inf_or_nan", debug.has_inf_or_nan)

		step =self.map_load(sess,self.algorithm.model.load_name,map3D=map3D)

		# if not const.eager:
		#     tf.get_default_graph().finalize()
		# print('finished graph initialization in %f seconds' % (time() - T1))
		return step

	def initsaver(self):
		self.savers = {}
		parts = self.algorithm.model.weights

		for partname in parts:
			partweights = parts[partname][1]
			if partweights:
				self.savers[partname] = tf.train.Saver(partweights)


	def save(self, sess, name, step):
		# st()
		config = Config(name, step)
		parts = self.algorithm.model.weights
		if len(name.split("/")) > 1:
			savepath = path.join(self.algorithm.model.ckpt_dir, name.split("/")[0])
			utils.utils.ensure(savepath)

		#savepath = path.join(self.algorithm.model.ckpt_base,self.algorithm.model.ckpt_dir, name)
		#import pdb; pdb.set_trace()
		savepath = path.join(self.algorithm.model.ckpt_base,"weights_"+self.expert_name)
		#expert_name
		utils.utils.ensure(savepath)
		for partname in parts:
			partpath = path.join(savepath, partname)
			utils.utils.ensure(partpath)
			partscope, weights = parts[partname]

			if not weights: #nothing to do
				continue

			partsavepath = path.join(partpath, 'X')

			saver = self.savers[partname]
			saver.save(sess, partsavepath, global_step=step)

			config.add(partname, partscope, partpath)
		config.save()


	def map_load(self,sess, name,map3D=None):
		if not path.exists(path.join(self.algorithm.model.ckpt_base,self.algorithm.model.ckpt_cfg_dir, name)):
			return 0
		config = Config(name)
		config.load()
		# st()
		parts = map3D.weights
		for partname in config.dct:
			partscope, partpath = config.dct[partname]
			if partname not in parts:
				raise Exception("cannot load, part %s not in model" % partpath)
			partpath =  path.join(self.algorithm.model.ckpt_base,partpath)
			ckpt = tf.train.get_checkpoint_state(partpath)
			if not ckpt:
				raise Exception("checkpoint not found? (1)")
			elif not ckpt.model_checkpoint_path:
				raise Exception("checkpoint not found? (2)")
			loadpath = ckpt.model_checkpoint_path
			st()
			print(loadpath)
			#loadpath = '/projects/katefgroup/mprabhud/rl/ckpt/rl_new_detector/1/main_weights/X-220'

			scope, weights = parts[partname]

			if not weights: #nothing to do
				continue


			weights = [v for v in weights if v.op.name.split('/')[2]!='action_predictor']

			
			weights = {utils.utils.exchange_scope(weight.op.name, scope, partscope): weight for weight in weights}

			saver = tf.train.Saver(weights)
			saver.restore(sess, loadpath)
			print(f"restore model from {loadpath}")
		return config.step

	def _train(self):
		for i in range(self.step,self.step+1000):
			self.algorithm.train_epoch(i)
			# st()
			if i % 10 ==0:
				self.save(self._session,self.algorithm.model.load_name,i)
				# num = 0
				# print("Further Sampling")
				# while num < 1000:
				# 	self.sampler.sample()
				# 	num = num+1
					# print(num,1e3)


	def _build(self):
		variant = copy.deepcopy(self._variant)

		environment_params = variant['environment_params']
		training_environment = self.training_environment = (
			get_environment_from_params_custom(environment_params['training']))


		batch_size = variant['sampler_params']['kwargs']['batch_size']
		observation_keys = environment_params['training']["kwargs"]["observation_keys"]
		bulledtPush = variant["map3D"]

		replay_pool = self.replay_pool = (
			get_replay_pool_from_variant(variant, training_environment))

		#sampler = self.sampler = get_sampler_from_variant(variant)
		sampler = SimpleSampler(batch_size=40, max_path_length=50, min_pool_size=0,mesh = self.expert_name, iteration=0)
		initial_exploration_policy = self.initial_exploration_policy = (
			get_policy('UniformPolicy', training_environment))

		#st()
		self.algorithm = bulledtPushTrainer		(
			variant=variant,
			map3D =bulledtPush,
			training_environment=training_environment,
			initial_exploration_policy=initial_exploration_policy,
			pool=replay_pool,
			batch_size = batch_size,
			observation_keys = observation_keys,
			sampler=sampler,
			eager_enabled = self.eager,
			# detector = self.detector,
			exp_name=self.exp_name,
			session=self._session, 
			expert_name = self.expert_name)
		

		# st()

		initialize_tf_variables(self._session, only_uninitialized=True)
		st()
		# if not self.algorithm.detector:
		self.step = self.map3d_setup(self._session,map3D=bulledtPush)
		# st()
		#self.initsaver()

		self._built = True

if __name__ == "__main__":
	er = ExperimentRunner()

