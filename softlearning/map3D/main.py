import constants as const
import tensorflow as tf
from fig import Config
import utils_map as utils
from nets.BulletPush3DTensor import BulletPush3DTensor4_cotrain
const.set_experiment("rl_new")
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






def load(sess, name):
    config = Config(name)
    config.load()
    parts = bulletpush.weights
    for partname in config.dct:
        partscope, partpath = config.dct[partname]
        if partname not in parts:
            raise Exception("cannot load, part %s not in model" % partpath)

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
req=True
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
    img = np.expand_dims(val["observations.image_observation"],0)
    cam_angle = np.expand_dims(val["observations.cam_angles_observation"],0)
    depth = np.expand_dims(val["observations.depth_observation"],0)
    N=54

# st()
eager = False

if const.eager and eager:
    tf.enable_eager_execution()
    tf.random.set_random_seed(1)

else:
    img_ph = tf.placeholder(tf.int32, [const.BS, 1, N, 84, 84, 4],"images")
    cam_angle_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 2],"angles")
    depth_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 84, 84, 1],"zmapss")
    img_ph2 = tf.placeholder(tf.int32, [const.BS, 1, N, 84, 84, 4],"images2")
    cam_angle_ph2 = tf.placeholder(tf.float32, [const.BS, 1, N, 2],"angles2")
    depth_ph2 = tf.placeholder(tf.float32, [const.BS, 1, N, 84, 84, 1],"zmapss2")
    img_ph3 = tf.placeholder(tf.int32, [1, 1, N, 84, 84, 4],"images3")
    cam_angle_ph3 = tf.placeholder(tf.float32, [1, 1, N, 2],"angles3")
    depth_ph3 = tf.placeholder(tf.float32, [1, 1, N, 84, 84, 1],"zmapss3")

if eager:
    m3dt = bulletpush(img,cam_angle,depth)


else:
# st()
    sess = tf.keras.backend.get_session()

    m3dt = bulletpush(img_ph,cam_angle_ph,depth_ph)
    m3dt2 = bulletpush(img_ph2,cam_angle_ph2,depth_ph2,reuse=True)
    bulletpush.set_batchSize(1)
    m3dt3 = bulletpush(img_ph3,cam_angle_ph3,depth_ph3,reuse=True)

# st()
val = tf.keras.layers.Conv2D(3, 1, 1, 'SAME',name="new")(tf.ones([1,64,64,3]))
# setup(sess)
# st()
a_f = {img_ph:np.repeat(img,4,0),cam_angle_ph:np.repeat(cam_angle,4,0),depth_ph:np.repeat(depth,4,0)}
b_f = {img_ph2:np.ones_like(np.repeat(img,4,0)),cam_angle_ph2:np.repeat(cam_angle,4,0),depth_ph2:np.repeat(depth,4,0)}
c_f = {img_ph3:img,cam_angle_ph3:cam_angle,depth_ph3:depth}

sess.run(tf.global_variables_initializer())
a = sess.run(m3dt,feed_dict=a_f)
b = sess.run(m3dt2,feed_dict=b_f)
c = sess.run(m3dt3,feed_dict=c_f)
a_f.update(b_f)
c = sess.run(m3dt2+m3dt,feed_dict=a_f)

print(np.sum(a+b),np.sum(c))

# imsave("out.png",a[0])
# st()

# st()


# load(sess,const.load_name)
# # st()
# print(len(tf.trainable_variables()))

# n3dt = bulletpush(inputsO[0],inputsO[2],inputsO[1],reuse=False)

# print(len(tf.trainable_variables()),len(bulletpush.weights))
