import constants as const
import tensorflow as tf
from fig import Config
import utils_map as utils
from nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain
const.set_experiment("rl_temp")
import ipdb
from scipy.misc import imsave

st = ipdb.set_trace
# input_shapes = [(1,4,84,84,4),(1,4,84,84,1),(1,4,2)]
# inputs = [tf.keras.layers.Input(shape=input_shape) for input_shape in input_shapes]


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



        # self.images = tf.placeholder(tf.float32, [const.BS, T, const.NUM_VIEWS, W, H, 4],"images")
        # self.angles = tf.placeholder(tf.float32, [const.BS, T, const.NUM_VIEWS, 2],"angles")
        # self.zmaps = tf.placeholder(tf.float32, [const.BS, T, const.NUM_VIEWS, W, H, 1],"zmapss")




# input_shapes = [(const.BS,1,4,84,84,4),(const.BS,1,4,84,84,1),(const.BS,1,4,2)]

# inputsZ = [tf.ones(input_shape) for input_shape in input_shapes]
# inputsO = [tf.zeros(input_shape) for input_shape in input_shapes]
import numpy as np
bulletpush = BulletPush3DTensor4_cotrain()

import pickle
req=False
if req:
    val = pickle.load(open("softlearning/map3D/req.p","rb"))
else:
    val = pickle.load(open("softlearning/map3D/og.p","rb"))

# st()
# st()
# img =  np.concatenate([np.expand_dims(val["observations.image_observation"],0),np.ones([const.BS,1,4,84,84,1])],-1).reshape([const.BS,1,4,84,84,4])
# # n
# depth = val["observations.depth_observation"].reshape([const.BS,1,4,84,84,1])
# cam_angle = val["observations.cam_angles_observation"].reshape([const.BS,1,4,2])
if req:
    img =  np.concatenate([np.expand_dims(val["observations.image_observation"],0),np.ones([1,1,4,84,84,1])],-1).reshape([1,1,4,84,84,4])
    cam_angle = np.expand_dims(val["observations.cam_angles_observation"],0)
    depth = np.expand_dims(np.expand_dims(val["observations.depth_observation"],0),-1)
    N=4
else:
    img = np.expand_dims(val["observations.image_observation"],0)/255.0
    cam_angle = np.expand_dims(val["observations.cam_angles_observation"],0)
    depth = np.expand_dims(val["observations.depth_observation"],0)
    N=54
import time
# st()
eager = False
save_graph = False
load_graph = False
if const.eager and eager:
    tf.enable_eager_execution()
    tf.random.set_random_seed(1)

else:
    if not load_graph:
        img_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 84, 84, 4],"images")
        cam_angle_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 2],"angles")
        depth_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 84, 84, 1],"zmapss")


# st()
if not load_graph:
    if eager:
        m3dt = bulletpush(img,cam_angle,depth)


    else:
    # st()
        sess = tf.keras.backend.get_session()

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
            output_name = "map3D"
            with tf.gfile.GFile(output_name + ".pb", 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("saving ",time.time()-t)
            # tf.train.write_graph(output_graph_def, "", output_name + ".pb", as_text=False)
            # with open(output_name + ".pb.pickle", "wb") as f:
            #     pickle.dump(output_dict, f)

        # m3dt2 = bulletpush(img_ph2,cam_angle_ph2,depth_ph2,reuse=True)
        # bulletpush.set_batchSize(1)
        # m3dt3 = bulletpush(img_ph3,cam_angle_ph3,depth_ph3,reuse=True)
else:
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
    img_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 84, 84, 4],"images")
    cam_angle_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 2],"angles")
    depth_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 84, 84, 1],"zmapss")

    tf.import_graph_def(graph_def, {'images': img_ph, "zmapss":depth_ph,"angles":cam_angle_ph})
    print(time.time()-t,"loading time")
    print('Model loading complete!')


# st()
# val = tf.keras.layers.Conv2D(3, 1, 1, 'SAME',name="new")(tf.ones([1,64,64,3]))
# setup(sess)

a_f = {img_ph:np.repeat(img,4,0), cam_angle_ph:np.repeat(cam_angle,4,0), depth_ph:np.repeat(depth,4,0)}

def get_dict():
    a_f = {img_ph:np.random.randn(*img_ph.shape.as_list()), cam_angle_ph:np.random.randn(*cam_angle_ph.shape.as_list()), depth_ph:np.random.randn(*depth_ph.shape.as_list())}
    return a_f
# st()
# d3 = graph.get_tensor_by_name('import/Variables/main/split_4:0')

debug_time = False

if debug_time:
    if load_graph:
        d3_2 = graph.get_tensor_by_name('import/Variables/main/3dstuff:0')
    else:
        d3_2 = bulletpush.memory_3D
    import time
    # st()
    for i in range(10):
        a_f = get_dict()
        t = time.time()
        val1 =sess.run(d3_2,a_f)
        print(time.time() -t)
else:
    a,b = sess.run([bulletpush.predicted_view,bulletpush.inputs.state.vp_frame],feed_dict=a_f)
    st()
    imsave("out.png",a[0])
    imsave("new.png",b[0])

# st()
# b_f = {img_ph2:np.ones_like(np.repeat(img,4,0)),cam_angle_ph2:np.repeat(cam_angle,4,0),depth_ph2:np.repeat(depth,4,0)}
# c_f = {img_ph3:img,cam_angle_ph3:cam_angle,depth_ph3:depth}

# sess.run(tf.global_variables_initializer())
# sess.run()
# a = sess.run(m3dt,feed_dict=a_f)
# b = sess.run(m3dt2,feed_dict=b_f)
# c = sess.run(m3dt3,feed_dict=c_f)
# a_f.update(b_f)
# c = sess.run(m3dt2+m3dt,feed_dict=a_f)

# print(np.sum(a+b),np.sum(c))

# imsave("out.png",a[0])
# st()

# st()


# load(sess,const.load_name)
# # st()
# print(len(tf.trainable_variables()))

# n3dt = bulletpush(inputsO[0],inputsO[2],inputsO[1],reuse=False)

# print(len(tf.trainable_variables()),len(bulletpush.weights))
