import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_op_util


def draw_dot(input_images, dot, sigma):
    """ 
    input image: batch_size x h x w x 1
    dot: batch_size x 2

    """
    bs, sh, sw, _ = input_images.get_shape()
    
    grid_z = tf.tile(tf.expand_dims((tf.dtypes.cast(tf.range(sh), tf.float32) + 0.5\
             - 0.5 * sh.value)/(0.5 * sh.value), 1), [1, sw])
    grid_x = tf.tile(tf.expand_dims((tf.dtypes.cast(tf.range(sw), tf.float32) + 0.5\
             - 0.5 * sw.value)/(0.5 * sw.value), 0), [sh, 1])
    grid_zx = tf.stack([grid_z, grid_x], 2)
        
    hand_grid = tf.expand_dims(grid_zx, 0) - tf.expand_dims(tf.expand_dims(dot, 1), 2)
    gauss_map = (1.0/np.sqrt(2.0 * np.pi * sigma**2)) * tf.exp((-0.5) *\
                tf.reduce_sum(tf.square(hand_grid), 3) * (1.0/(sigma**2)))
    
    gauss_map = gauss_map/tf.reduce_max(gauss_map, axis=[1,2], keepdims=True)
    gauss_map = tf.expand_dims(gauss_map, 3)
    #   / tf.max(gauss_map)

    return gauss_map


def create_diff_images(image1, image2):
   pos_image = tf.clip_by_value(tf.reduce_mean(image1 - image2, -1), 0, 1000)
   negative_image = tf.clip_by_value(tf.reduce_mean(image2 - image1, -1), 0, 1000)
   zero_pad = tf.zeros_like(pos_image)
   final_image = tf.stack([pos_image, negative_image, zero_pad], -1)
   return final_image


def encode_gif(images, fps):
  """Encodes numpy images into gif string.
  Args:
    images: A 5-D `uint8` `np.array` (or a list of 4-D images) of shape
      `[batch_size, time, height, width, channels]` where `channels` is 1 or 3.
    fps: frames per second of the animation
  Returns:
    The encoded gif string.
  Raises:
    IOError: If the ffmpeg command returns an error.
  """
  from subprocess import Popen, PIPE
  h, w, c = images[0].shape
  cmd = [
      'ffmpeg', '-y',
      '-f', 'rawvideo',
      '-vcodec', 'rawvideo',
      '-r', '%.02f' % fps,
      '-s', '%dx%d' % (w, h),
      '-pix_fmt', {1: 'gray', 3: 'rgb24'}[c],
      '-i', '-',
      '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
      '-r', '%.02f' % fps,
      '-f', 'gif',
      '-']
  proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in images:
    proc.stdin.write(image.tostring())
  out, err = proc.communicate()
  if proc.returncode:
    err = '\n'.join([' '.join(cmd), err.decode('utf8')])
    raise IOError(err)
  del proc
  return out


def py_gif_summary(tag, images, max_outputs, fps):
  """Outputs a `Summary` protocol buffer with gif animations.
  Args:
    tag: Name of the summary.
    images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height, width,
      channels]` where `channels` is 1 or 3.
    max_outputs: Max number of batch elements to generate gifs for.
    fps: frames per second of the animation
  Returns:
    The serialized `Summary` protocol buffer.
  Raises:
    ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
  """
  is_bytes = isinstance(tag, bytes)
  if is_bytes:
    tag = tag.decode("utf-8")
  images = np.asarray(images)
  if images.dtype != np.uint8:
    raise ValueError("Tensor must have dtype uint8 for gif summary.")
  if images.ndim != 5:
    raise ValueError("Tensor must be 5-D for gif summary.")
  batch_size, _, height, width, channels = images.shape
  if channels not in (1, 3):
    raise ValueError("Tensors must have 1 or 3 channels for gif summary.")

  summ = tf.Summary()
  num_outputs = min(batch_size, max_outputs)
  for i in range(num_outputs):
    image_summ = tf.Summary.Image()
    image_summ.height = height
    image_summ.width = width
    image_summ.colorspace = channels  # 1: grayscale, 3: RGB
    try:
      image_summ.encoded_image_string = encode_gif(images[i], fps)
    except (IOError, OSError) as e:
      tf.logging.warning(
          "Unable to encode images to a gif string because either ffmpeg is "
          "not installed or ffmpeg returned an error: %s. Falling back to an "
          "image summary of the first frame in the sequence.", e)
      try:
        from PIL import Image  # pylint: disable=g-import-not-at-top
        import io  # pylint: disable=g-import-not-at-top
        with io.BytesIO() as output:
          Image.fromarray(images[i][0]).save(output, "PNG")
          image_summ.encoded_image_string = output.getvalue()
      except:
        tf.logging.warning(
            "Gif summaries requires ffmpeg or PIL to be installed: %s", e)
        image_summ.encoded_image_string = "".encode('utf-8') if is_bytes else ""
    if num_outputs == 1:
      summ_tag = "{}/gif".format(tag)
    else:
      summ_tag = "{}/gif/{}".format(tag, i)
    summ.value.add(tag=summ_tag, image=image_summ)
  summ_str = summ.SerializeToString()
  return summ_str


def gif_summary(name, tensor, max_outputs, fps, collections=None, family=None):
  """Outputs a `Summary` protocol buffer with gif animations.
  Args:
    name: Name of the summary.
    tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width,
      channels]` where `channels` is 1 or 3.
    max_outputs: Max number of batch elements to generate gifs for.
    fps: frames per second of the animation
    collections: Optional list of tf.GraphKeys.  The collections to add the
      summary to.  Defaults to [tf.GraphKeys.SUMMARIES]
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.
  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  tensor = tf.convert_to_tensor(tensor)
  if summary_op_util.skip_summary():
    return tf.constant("")
  with summary_op_util.summary_scope(
      name, family, values=[tensor]) as (tag, scope):
    val = tf.py_func(
        py_gif_summary,
        [tag, tensor, max_outputs, fps],
        tf.string,
        #stateful=False,
        name=scope)
    summary_op_util.collect(val, collections, [tf.GraphKeys.SUMMARIES])
  return val

