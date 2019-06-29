import importlib
import sys
import ipdb
st = ipdb.set_trace
import tensorflow as tf
import sys
from examples.map3D.main_map3d import ExperimentRunner

example_module_name = "examples.map3D"
example_argv = ('--universe=gym', '--checkpoint-frequency=0', '--domain=SawyerReachXYEnv', '--task=v1', '--trial-gpus=1', '--exp-name=test', '--replay_pool=SimpleReplayPoolTemp', '--algorithm=SAC')



example_module = importlib.import_module(example_module_name)

example_args = example_module.get_parser().parse_args(example_argv)
# print("example_args",example_args)
variant_spec = example_module.get_variant_spec(example_args)
# st()
er = ExperimentRunner()

er._setup(variant_spec)

er._build()

er._train()




# print(variant_spec)
