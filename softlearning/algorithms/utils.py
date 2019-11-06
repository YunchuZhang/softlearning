from copy import deepcopy
import os
import ipdb
st = ipdb.set_trace

def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC

    algorithm = SAC(variant, *args, **kwargs)

    return algorithm


def create_SQL_algorithm(variant, *args, **kwargs):
    from .sql import SQL

    algorithm = SQL(*args, **kwargs)

    return algorithm

def create_RemoteSAC_algorithm(variant, *args, **kwargs):
    from .remote_sac import RemoteSAC

    algorithm = RemoteSAC(variant, *args, remote=True, **kwargs)

    return algorithm


def create_SAC_VAE_algorithm(variant, *args, **kwargs):
    from .sac_vae import SAC_VAE

    algorithm = SAC_VAE(*args, **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'SQL': create_SQL_algorithm,
    'RemoteSAC': create_RemoteSAC_algorithm,
    'SAC_VAE': create_SAC_VAE_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    # st()
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm

def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    # st()
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm

def map3D_save(sess, checkpt, map3D_saver, gstep=1):
    for k, saver in map3D_saver.items():
        path = os.path.join(checkpt, k)
        os.makedirs(path, exist_ok=True)
        saver.save(sess, os.path.join(path, k), write_meta_graph=False, global_step=gstep)

def map3D_load(sess, checkpt, map3D_saver):
    for k, saver in map3D_saver.items():
        path = os.path.join(checkpt, k)
        saver.restore(sess, os.path.join(path, k + "-0"))
