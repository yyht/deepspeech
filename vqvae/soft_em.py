import tensorflow as tf
import numpy as np

"""
https://github.com/jaywalnut310/waveglow-vqvae
https://github.com/HaebinShin/dec-tensorflow/blob/master/dec/model.py
"""

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def init_discrete_bottleneck(bottleneck_size, bottleneck_dims, dtype="float32"):
  """Get lookup table for discrete bottleneck."""
  code_book = tf.get_variable(
      name="code_book",
      shape=[bottleneck_size, bottleneck_dims],
      dtype=tf.float32)
  return code_book

def vqvae(x, 
          bottleneck_size, 
          bottleneck_dims,
          code_book, 
          beta=0.25,
          gamma=0.1,
          is_training=False):
  """Combining EM and VAE"""

  # Caculate square distance
  def _square_distance(x, code_book):
    x = tf.cast(x, tf.float32)
    code_book = tf.cast(code_book, tf.float32)
    x_sg = tf.stop_gradient(x)
    x_norm_sq = tf.reduce_sum(tf.square(x_sg), axis=-1, keepdims=True) # [b, 1]
    code_book_norm_sq = tf.reduce_sum(tf.square(code_book), axis=-1, keepdims=True) # [V, 1]
    scalar_prod = tf.matmul(x_sg, code_book, transpose_b=True) # [b, V]
    dist_sq = x_norm_sq + tf.transpose(code_book_norm_sq) - 2 * scalar_prod # [b, V]

    return tf.cast(dist_sq, x.dtype.base_dtype)

  dist_sq = _square_distance(x, code_book)
  tf.logging.info("*** dist_sq ***")
  tf.logging.info(dist_sq)

  q = tf.stop_gradient(tf.nn.softmax(-.5 * dist_sq))

  tf.logging.info("*** q-prob ***")
  tf.logging.info(q)

  discrete = tf.one_hot(tf.argmax(-dist_sq, axis=-1), depth=bottleneck_size, dtype=code_book.dtype.base_dtype)
  
  tf.logging.info("*** discrete ***")
  tf.logging.info(discrete)

  dense = tf.matmul(discrete, code_book)

  tf.logging.info("*** dense ***")
  tf.logging.info(dense)

  if is_training:
    dense = dense + x - tf.stop_gradient(x)

    tf.logging.info("*** apply straight through dense ***")
    tf.logging.info(dense)

  def _get_losses(x, dense, dist_sq, q):
    x = tf.cast(x, tf.float32)
    dense = tf.cast(dense, tf.float32)
    dist_sq = tf.cast(dist_sq, tf.float32)
    q = tf.cast(q, tf.float32)
    disc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - tf.stop_gradient(dense)), axis=-1))
    # # M-step
    em_loss = -tf.reduce_mean(tf.reduce_sum(-.5 * dist_sq * q, axis=-1))
    return disc_loss, em_loss
  disc_loss, em_loss = _get_losses(x, dense, dist_sq, q)

  loss_dict = {
    "disc_loss": beta * disc_loss,
    "em_loss": gamma * em_loss,
  }
  return discrete, dense, loss_dict

def discrete_bottleneck(
          x,
          bottleneck_size,
          bottleneck_dims,
          beta=0.25,
          gamma=0.1,
          is_training=False):

  code_book = init_discrete_bottleneck(bottleneck_size, 
                          bottleneck_dims, 
                          dtype="float32")

  tf.logging.info("*** code_book ***")
  tf.logging.info(code_book)

  x_shape = shape_list(x)
  if x_shape[-1] != bottleneck_dims:
    tf.logging.info("*** linear transformation to bottleneck_dims ***")
    x = tf.layers.dense(x, bottleneck_dims)
    x_shape = shape_list(x)

  x = layer_norm(x)

  tf.logging.info("*** code_book ***")
  tf.logging.info(code_book)

  # from [B, T, V] to [B*T, V]
  x = tf.reshape(x, [-1, bottleneck_dims])

  tf.logging.info("*** x resshape ***")
  tf.logging.info(x)

  [discrete, 
  dense, 
  loss_dict] = vqvae(x, 
        bottleneck_size, 
        bottleneck_dims,
        code_book, 
        beta=beta,
        gamma=gamma,
        is_training=is_training)

  discrete = tf.reshape(discrete, x_shape[:-1] + [bottleneck_size])
  dense = tf.reshape(dense, x_shape[:-1] + [bottleneck_dims])

  return code_book, discrete, dense, loss_dict