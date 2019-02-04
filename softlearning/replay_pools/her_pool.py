import numpy as np

from .simple_replay_pool import SimpleReplayPool

class HerPool(SimpleReplayPool):

    def __init__(self, *args, **kwargs):
        super(HerPool, self).__init__(*args, **kwargs)

    def batch_by_indices(self, indices, field_name_filter=None, observation_keys=None):
        pass
