import constants as const
import tensorflow as tf
# from fig import Config
# import utils_map as utils
# from nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain
const.set_experiment("rl_temp")
# import ipdb
from scipy.misc import imsave
import time
import multiprocessing as mp
import numpy as np
import pathos.pools as pp

# st = ipdb.set_trace
N=4
# input_shapes = [(1,4,84,84,4),(1,4,84,84,1),(1,4,2)]
# inputs = [tf.keras.layers.Input(shape=input_shape) for input_shape in input_shapes]
def initialize_graph():
    global d3_2
    global img_ph,cam_angle_ph,depth_ph,sess
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph = graph)
    output_name = "map3D"
    model_filepath =  output_name + ".pb"
    t= time.time()
    with tf.gfile.GFile(model_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # print('Check out the input placeholders:')
    # nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
    # for node in nodes:
    #     print(node)
    # print('Check out the input placeholders:')
    # nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
    # for node in nodes:
    #     print(node)

    # # Define input tensor
    # input = tf.placeholder(np.float32, shape = [None, 32, 32, 3], name='input')
    # dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')
    img_ph = tf.placeholder(tf.int32, [const.BS, 1, N, 84, 84, 4],"images")
    cam_angle_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 2],"angles")
    depth_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 84, 84, 1],"zmapss")

    tf.import_graph_def(graph_def, {'images': img_ph, "zmapss":depth_ph,"angles":cam_angle_ph})
    print(time.time()-t,"loading time")
    print('Model loading complete!')
    d3_2 = graph.get_tensor_by_name('import/Variables/main/3dstuff:0')
    return sess,d3_2,img_ph,cam_angle_ph,depth_ph


def setup(sess):
    # print('finished graph creation in %f seconds' % (time() - const.T0))

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    #must come after the queue runners
    # if const.DEBUG_NAN:
    #     self.sess = debug.LocalCLIDebugWrapperSession(self.sess)
    #     self.sess.add_tensor_filter("has_inf_or_nan", debug.has_inf_or_nan)

    step =load(sess,const.load_name)

    if not const.eager:
        tf.get_default_graph().finalize()
    # print('finished graph initialization in %f seconds' % (time() - T1))

server = False




def load(sess, name):
    if server:
        const.set_experiment("rl_new")
    else:
        const.set_experiment("rl_temp")

    # st()
    config = Config(name)
    config.load()
    parts = bulletpush.weights
    for partname in config.dct:
        partscope, partpath = config.dct[partname]
        if partname not in parts:
            raise Exception("cannot load, part %s not in model" % partpath)
        if server:
            ckpt = tf.train.get_checkpoint_state("/home/mprabhud/rl/softlearning/softlearning/map3D/"+partpath)
        else:
            ckpt = tf.train.get_checkpoint_state(partpath)

        if not ckpt:
            raise Exception("checkpoint not found? (1)")
        elif not ckpt.model_checkpoint_path:
            raise Exception("checkpoint not found? (2)")

        loadpath = ckpt.model_checkpoint_path

        scope, weights = parts[partname]

        if not weights: #nothing to do
            continue
        
        weights = {utils.utils.exchange_scope(weight.op.name, scope, partscope): weight
                   for weight in weights}

        saver = tf.train.Saver(weights)
        saver.restore(sess, loadpath)
        print(f"restore model from {loadpath}")
    return config.step

# initialize_graph()
def save_graph():
    sess = tf.Session(config=tf.ConfigProto(device_count={ "CPU": 1 }))
    bulletpush = BulletPush3DTensor4_cotrain()

    img_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 84, 84, 4],"images")
    cam_angle_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 2],"angles")
    depth_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 84, 84, 1],"zmapss")
    m3dt = bulletpush(img_ph,cam_angle_ph,depth_ph)
    setup(sess)
    if save_graph:

        output_node = [bulletpush.memory_3D]
        output_node_names =[node.op.name for node in output_node]
        # st()
        output_names = ["memory_3D"]
        output_dict = dict()
        for node_id, node in enumerate(output_node):
            output_dict[output_names[node_id]] = node.name
        t = time.time()
        gd = sess.graph.as_graph_def()    
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, gd, output_node_names)
        output_name = "map3D_newtry"
        with tf.gfile.GFile(output_name + ".pb", 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("saving ",time.time()-t)

# val= []

def frozen_graph(a_f,sess,d3_2,img_ph,cam_angle_ph,depth_ph):
    # st()
    # sess,d3_2,img_ph,cam_angle_ph,depth_ph=initialize_graph()
    # global img_ph,cam_angle_ph,depth_ph,sess, 
    # a_f = get_dict()
    # print("start")
    val1 =sess.run(d3_2,{img_ph:a_f[0],cam_angle_ph:a_f[1],depth_ph:a_f[2]})
    return val1
    # print("done")


def get_dict():
    global img_ph,cam_angle_ph,depth_ph
    # a_f = {img_ph:np.random.randn(*[const.BS, 1, N, 84, 84, 4]), cam_angle_ph:np.random.randn(*[const.BS, 1, N, 2]), depth_ph:np.random.randn(*[const.BS, 1, N, 84, 84, 1])}
    a_f = [np.random.randn(*[const.BS, 1, N, 84, 84, 4]), np.random.randn(*[const.BS, 1, N, 2]),np.random.randn(*[const.BS, 1, N, 84, 84, 1])]

    return a_f
# st()
# d3 = graph.get_tensor_by_name('import/Variables/main/split_4:0')


if __name__ == "__main__":
    # cp = mp.cpu_count()
    # print(cp)
    num =10 
    
    val = [get_dict() for _ in range(num)]
    init_s_time = time.time()
    sess,d3_2,img_ph,cam_angle_ph,depth_ph = initialize_graph()

    print("initialization time ",time.time()-init_s_time)

    for i in range(num):
        print(i)
        frozen_graph(val[i],sess,d3_2,img_ph,cam_angle_ph,depth_ph)

