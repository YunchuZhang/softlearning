import numpy as np
from softlearning.environments.gym.flex.flex_wrappers import FetchReach

env = FetchReach(1)
env.reset()
for i in range(100):
	print(env.step(np.random.randn(3)))