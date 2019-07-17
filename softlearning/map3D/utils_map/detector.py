import tensorflow.contrib.slim as slim
import tensorflow as tf
 # import ipdb
 # st = ipdb.set_trace

def detector(inputs):
	bn = True
	with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
						activation_fn=tf.nn.relu,
						normalizer_fn=slim.batch_norm if bn else None,
						):
		d0 = 16
		dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
		ksizes = [4, 4, 4, 4, 8]
		strides = [2, 2, 2, 2, 1]
		paddings = ['SAME'] * 4 + ['VALID']

		# if const.S == 64:
		# 	ksizes[-1] = 4
		# elif const.S == 32:
		ksizes[-1] = 2

		net = inputs

		skipcons = [net]
		for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
			net = slim.conv3d(net, dim, ksize, stride=1, padding=padding)
			net = tf.nn.pool(net, [3,3,3], 'MAX', 'SAME', strides = [2,2,2])
		net = tf.layers.flatten(net)
		net = tf.layers.dense(net,128,activation=tf.nn.relu)
		net = tf.layers.dense(net,64,activation=tf.nn.relu)
		net = tf.layers.dense(net,32,activation=tf.nn.relu)
		net = tf.layers.dense(net,3)

	return net

if __name__ == "__main__":
	val = tf.zeros([1,32,32,32,8])
	res = detector(val)
	print(res.shape)