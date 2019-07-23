import importlib
import sys
import ipdb
st = ipdb.set_trace
import tensorflow as tf
import sys
from examples.map3D.main_map3d import ExperimentRunner


example_module_name = "examples.map3D"

#list_of_experts = ["expert_boat","expert_bowl1","expert_can1","expert_car1","expert_car2","expert_mug3"]
list_of_experts = ["expert_can1"]

example_argv = ('--universe=gym', '--checkpoint-frequency=0', '--domain=SawyerPushAndReachEnvEasy', '--task=v0', '--trial-gpus=1', '--exp-name=test', '--replay_pool=SimpleReplayPoolTemp', '--algorithm=SAC', '--expert_name=expert_mug3')
for expert in list_of_experts:
	expert_name = "--expert_name=" + expert
	example_argv = list(example_argv)
	example_argv[8] = expert_name
	example_argv = tuple(example_argv)
	#st()

	example_module = importlib.import_module(example_module_name)

	example_args = example_module.get_parser().parse_args(example_argv)
	# print("example_args",example_args)
	variant_spec = example_module.get_variant_spec(example_args)
	#st()
	eager = False
	# detector = True
	er = ExperimentRunner()

	if eager:
		tf.enable_eager_execution()

	er._setup("rl_new_reach_action_predictor",variant_spec,example_args.expert_name ,eager)

	# er._setup("rl_new_reach_detect",variant_spec,eager)

	er._build()

	er._train()

	print("done with expert", expert)






# print(variant_spec)
