import numpy as np
import tensorflow as tf

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def get_shape_invariants(tensor):
  shapes = shape_list(tensor)
  return tf.TensorShape([i if isinstance(i, int) else None for i in shapes])

def batch_time_mask(input_tensor, 
                num_masks=1, 
                mask_factor=100, 
                p_upperbound=1.0):
  """
  input_tensor: [batch_size, T, num_feature_bins*V]
  --->
   (T, num_feature_bins, batch_size)
  """
  spectrogram = tf.tranpose(input_tensor, [1,2,0])
  T, F, V = shape_list(spectrogram, out_type=tf.int32)
  for _ in range(num_masks):
    t = tf.random.uniform([], minval=0, maxval=mask_factor, dtype=tf.int32)
    t = tf.minimum(t, tf.cast(tf.cast(T, dtype=tf.float32) * p_upperbound, dtype=tf.int32))
    t0 = tf.random.uniform([], minval=0, maxval=(T - t), dtype=tf.int32)
    mask = tf.concat([
      tf.ones([t0, F, V], dtype=spectrogram.dtype),
      tf.zeros([t, F, V], dtype=spectrogram.dtype),
      tf.ones([T - t0 - t, F, V], dtype=spectrogram.dtype)
    ], axis=0)
    spectrogram = spectrogram * mask
  spectrogram = tf.tranpose(spectrogram, [2,0,1])
  return spectrogram

def batch_freq_mask(input_tensor, 
                    num_masks=1, 
                    mask_factor=27):
  """
  input_tensor: [batch_size, T, num_feature_bins]
  --->
   (T, num_feature_bins, batch_size)
  """
  spectrogram = tf.tranpose(input_tensor, [1,2,0])
  T, F, V = shape_list(spectrogram, out_type=tf.int32)
  for _ in range(num_masks):
    f = tf.random.uniform([], minval=0, maxval=mask_factor, dtype=tf.int32)
    f = tf.minimum(f, F)
    f0 = tf.random.uniform([], minval=0, maxval=(F - f), dtype=tf.int32)
    mask = tf.concat([
      tf.ones([T, f0, V], dtype=spectrogram.dtype),
      tf.zeros([T, f, V], dtype=spectrogram.dtype),
      tf.ones([T, F - f0 - f, V], dtype=spectrogram.dtype)
    ], axis=1)
    spectrogram = spectrogram * mask
  spectrogram = tf.tranpose(spectrogram, [2,0,1])
  return spectrogram

class TFTimeMasking:
  def __init__(self, num_masks = 1, mask_factor = 100, p_upperbound = 1.0):
    self.num_masks = num_masks
    self.mask_factor = mask_factor
    self.p_upperbound = p_upperbound

  def augment(self, spectrogram):
    """
    Masking the time channel (shape[0])
    Args:
      spectrogram: shape (T, num_feature_bins, V)
    Returns:
      frequency masked spectrogram
    """
    T, F, V = shape_list(spectrogram, out_type=tf.int32)
    for _ in range(self.num_masks):
      t = tf.random.uniform([], minval=0, maxval=self.mask_factor, dtype=tf.int32)
      t = tf.minimum(t, tf.cast(tf.cast(T, dtype=tf.float32) * self.p_upperbound, dtype=tf.int32))
      t0 = tf.random.uniform([], minval=0, maxval=(T - t), dtype=tf.int32)
      mask = tf.concat([
        tf.ones([t0, F, V], dtype=spectrogram.dtype),
        tf.zeros([t, F, V], dtype=spectrogram.dtype),
        tf.ones([T - t0 - t, F, V], dtype=spectrogram.dtype)
      ], axis=0)
      spectrogram = spectrogram * mask
    return spectrogram

class TFFreqMasking:
  def __init__(self, num_masks=1, mask_factor=27):
    self.num_masks = num_masks
    self.mask_factor = mask_factor

  def augment(self, spectrogram):
    """
    Masking the frequency channels (shape[1])
    Args:
      spectrogram: shape (T, num_feature_bins, V)
    Returns:
      frequency masked spectrogram
    """
    T, F, V = shape_list(spectrogram, out_type=tf.int32)
    for _ in range(self.num_masks):
      f = tf.random.uniform([], minval=0, maxval=self.mask_factor, dtype=tf.int32)
      f = tf.minimum(f, F)
      f0 = tf.random.uniform([], minval=0, maxval=(F - f), dtype=tf.int32)
      mask = tf.concat([
        tf.ones([T, f0, V], dtype=spectrogram.dtype),
        tf.zeros([T, f, V], dtype=spectrogram.dtype),
        tf.ones([T, F - f0 - f, V], dtype=spectrogram.dtype)
      ], axis=1)
      spectrogram = spectrogram * mask
    return spectrogram