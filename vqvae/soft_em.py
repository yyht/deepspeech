import tensorflow as tf
import numpy as np

"""
https://github.com/jaywalnut310/waveglow-vqvae
https://github.com/HaebinShin/dec-tensorflow/blob/master/dec/model.py
"""

def init_discrete_bottleneck(bottleneck_bits, bottleneck_dims_per_bit, dtype="float32"):
  """Get lookup table for discrete bottleneck."""
  bottleneck_size = 2 ** bottleneck_bits
  discrete_channels = bottleneck_bits * bottleneck_dims_per_bit
  code_book = tf.get_variable(
      name="code_book",
      shape=[bottleneck_size, discrete_channels],
      dtype=tf.float32)
  return code_book

def vqvae(x, 
          bottleneck_bits, 
          code_book, 
          beta=0.25,
          gamma=0.1,
          is_training=False):
  """Combining EM and VAE"""
  bottleneck_size = 2**config.bottleneck_bits

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

  q = tf.stop_gradient(tf.nn.softmax(-.5 * dist_sq))

  discrete = tf.one_hot(tf.argmax(-dist_sq, axis=-1), depth=bottleneck_size, dtype=code_book.dtype.base_dtype)
  dense = tf.matmul(discrete, code_book)
  if is_training:
    dense = dense + x - tf.stop_gradient(x)

  def _get_losses(x, dense, dist_sq, q):
    x = tf.cast(x, tf.float32)
    dense = tf.cast(dense, tf.float32)
    dist_sq = tf.cast(dist_sq, tf.float32)
    q = tf.cast(q, tf.float32)
    disc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - tf.stop_gradient(dense)), -1))
    em_loss = -tf.reduce_mean(tf.reduce_sum(-.5 * dist_sq * q, -1)) # M-step
    return disc_loss, em_loss
  disc_loss, em_loss = _get_losses(x, dense, dist_sq, q)

  losses = {
    "disc_loss": beta * disc_loss,
    "em_loss": gamma * em_loss,
  }
  return discrete, dense, losses