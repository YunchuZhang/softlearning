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
# val = pickle.load(open("softlearning/map3D/og.p","rb"))
# img = np.expand_dims(val["observations.image_observation"],0)
# cam_angle = np.expand_dims(val["observations.depth_observation"],0)
# depth = np.expand_dims(val["observations.cam_angles_observation"],0)



# st()
# img =  np.concatenate([np.expand_dims(val["observations.image_observation"],0),np.ones([const.BS,1,4,84,84,1])],-1).reshape([const.BS,1,4,84,84,4])
# # n
# depth = val["observations.depth_observation"].reshape([const.BS,1,4,84,84,1])
# cam_angle = val["observations.cam_angles_observation"].reshape([const.BS,1,4,2])

# st()

N=54
img_ph = tf.placeholder(tf.int32, [const.BS, 1, N, 84, 84, 4],"images")
cam_angle_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 2],"angles")
depth_ph = tf.placeholder(tf.float32, [const.BS, 1, N, 84, 84, 1],"zmapss")


sess = tf.Session()

m3dt = bulletpush(img_ph,cam_angle_ph,depth_ph)
train_var = tf.trainable_variables()
val = ["2Dencoder","2Ddecoder","3DED","depthchannel_net","gqn3d_2Ddecoder"]
check = []
for i in train_var:
    if ((val[0] not in i.name) and  (val[2] not in i.name) and (val[3] not in i.name) and (val[4] not in i.name)):
        check.append(i.name)
# val = [i for i in train_var if ((val[0] not in i.name))]
# print(val)
# st()
# st()
setup(sess)
# sess.run(tf.global_variables_initializer())
a = sess.run(bulletpush.predicted_view,feed_dict={img_ph:img,cam_angle_ph:cam_angle,depth_ph:depth})
imsave("out.png",a[0])
st()

# st()


# load(sess,const.load_name)
# # st()
# print(len(tf.trainable_variables()))

# n3dt = bulletpush(inputsO[0],inputsO[2],inputsO[1],reuse=False)

# print(len(tf.trainable_variables()),len(bulletpush.weights))