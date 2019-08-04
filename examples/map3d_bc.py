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
example_module_name = "examples.map3D"

#list_of_experts = ["expert_boat","expert_bowl1","expert_can1","expert_car1","expert_car2","expert_mug3"]
list_of_experts = ["hat1"]

example_argv = ('--universe=gym', '--checkpoint-frequency=0', '--domain=SawyerPushAndReachEnvEasy', '--task=v0', '--trial-gpus=1', '--exp-name=test', '--replay_pool=SimpleReplayPoolTemp', '--algorithm=SAC', '--mesh=expert_mug3')
for expert in list_of_experts:
	expert_name = "--mesh=dagger_" + expert
	example_argv = list(example_argv)
	example_argv[8] = expert_name
	example_argv = tuple(example_argv)


	example_module = importlib.import_module(example_module_name)

	example_args = example_module.get_parser().parse_args(example_argv)
	# print("example_args",example_args)
	variant_spec = example_module.get_variant_spec(example_args)
	#st()
	eager = False
	action_predictor = True
	er = ExperimentRunner()

	if eager:
		tf.enable_eager_execution()

	er._setup("rl_new_reach",variant_spec,example_args.mesh ,eager)

	# er._setup("rl_new_reach_detect",variant_spec,eager)

	er._build()
	#st()
	#er._train()
	number_iterations = 5 #number of dagger iterations

	for iteration in range(number_iterations):
		start_time = time.time()
		#combine old experience and the expertactions on the sample trajectories to dataset D
		# and train bc agent on D
		#main_dagger(iteration, mesh)
		#main_dagger(iteration, mesh) #expert,epoch,iteration)
		#test()
		training_epochs = 21
		er.algorithm.train_epoch(expert,training_epochs,iteration)
		#sample trajectories and store the experts actions
		max_rollouts = 2 #300 #how many starting conditions to sample and to roll out
		succes_rate = rollout_and_gather_data(max_rollouts, expert, iteration)
		#main_dagger_without(iteration, mesh)
		time.stop

		print("done with iteration ", iteration," on object", mesh, "with succes rate", succes_rate, "and it took me ", start_time - time.time())




	print("done with expert", expert)

