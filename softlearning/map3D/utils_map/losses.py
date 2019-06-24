import tensorflow as tf
import constants as const



def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1

def l1loss_stackedmask(x, y, mask):
    f = lambda q: tf.reshape(q, [const.BS, const.H, const.W, const.V, -1])
    return l1loss(f(x), f(y), f(mask))


def l1loss(x, y, mask=None, verify_shape=True, stopgrad = True):
    if verify_shape:
        assert len(x.get_shape()) == len(y.get_shape())
    if mask is None:
        return tf.reduce_mean(tf.abs(x - y))
    else:
        mask = tf.stop_gradient(mask)
        # where the mask is 0, the loss is 0
        mask = tf.stop_gradient(mask)
        return tf.reduce_mean(tf.abs(mask * (x - y))) / (tf.reduce_mean(mask) + const.eps)


def quatloss(x, y, mask=None):
    """
    mask does not include the last dimension
    """
    loss = 1- tf.square(tf.reduce_sum(tf.multiply(x, y), -1))
    if mask == None:
        mean_loss = tf.reduce_mean(loss)
    else:
        mean_loss = tf.reduce_sum(loss * mask)/tf.reduce_sum(mask)
    return mean_loss

def quat_angle_loss(x, y, mask=None):
    """
    mask does not include the last dimension
    """
    x = tf.math.l2_normalize(x, -1)
    y = tf.math.l2_normalize(y, -1)
    loss = 2 * tf.square(tf.reduce_sum(tf.multiply(x, y), -1)) - 1
    loss = tf.acos(loss)

    if mask == None:
        mean_loss = tf.reduce_mean(loss)
    else:
        mean_loss = tf.reduce_sum(loss * mask)/tf.reduce_sum(mask)
    return mean_loss


def l2loss(x, y, mask=None, verify_shape = True, strict = False):
    if verify_shape:
        assert len(x.get_shape()) == len(y.get_shape())
    if strict:
        xshp = x.shape.as_list()
        yshp = y.shape.as_list()
        assert (None not in xshp) and (None not in yshp) and xshp == yshp
    if mask is None:
        return tf.reduce_mean(tf.square(x - y))
    else:
        # where the mask is 0, the loss is 0
        return tf.reduce_mean(tf.square(mask * (x - y))) / (tf.reduce_mean(mask) + const.eps)

def eudloss(x, y, mask=None, debug=False):

    assert(len(x.shape) == len(mask.shape) + 1, f"x shape: {len(x.shape)}, mask shape: {len(mask.shape)}")
    loss = tf.sqrt(tf.reduce_sum(tf.square(x - y), -1)) * mask

    if mask is None:
        mean_loss = tf.reduce_mean(loss)
    else:
        mean_loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    if debug:
        return mean_loss, tf.sqrt(tf.reduce_sum(tf.square(x - y), -1)), mask
    else:
        return mean_loss

def binary_uncertainty(x, p=0.1):
    return x * (1 - p) + p / 2.0


def binary_ce_loss(x, y, mask=None, positive_weight = 1.0, negative_weight = 1.0):
    inner = (positive_weight * y * tf.log(x + const.eps) +
             negative_weight * (1 - y) * tf.log(1 - x + const.eps))
    
    if mask is None:
        return -tf.reduce_mean(inner)
    else:
        return -tf.reduce_mean(mask * inner) / (tf.reduce_mean(mask) + const.eps)


def two_way_ce_loss(x, y, mask=None, p=0.1):
    x = binary_uncertainty(x, p)
    y = binary_uncertainty(y, p)
    return binary_ce_loss(x, y, mask) + binary_ce_loss(y, x, mask)


def smoothLoss(x):
    bs, h, w, c = x.get_shape()
    kernel = tf.transpose(tf.constant([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]],
                                       [[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]],
                                      dtype=tf.float32), perm=[3, 2, 1, 0],
                          name="kernel")
    diffs = [tf.nn.conv2d(tf.expand_dims(x_, axis=3), kernel, [1, 1, 1, 1],
                          padding="SAME", name="diff") for x_ in tf.unstack(x, axis=-1)]
    diff = tf.concat(axis=3, values=diffs)
    mask = tf.ones([bs, h - 1, w - 1, 1], name="mask")
    mask = tf.concat(axis=1, values=[mask, tf.zeros([bs, 1, w - 1, 1])])
    mask = tf.concat(axis=2, values=[mask, tf.zeros([bs, h, 1, 1])])
    loss = tf.reduce_mean(tf.abs(diff * mask), name="loss")
    return loss
