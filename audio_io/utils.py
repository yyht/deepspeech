

import tensorflow as tf

def count_non_blank(tensor, blank, axis=None):
  return tf.reduce_sum(tf.where(tf.not_equal(tensor, blank), x=tf.ones_like(tensor), y=tf.zeros_like(tensor)), axis=axis)

def float_feature(list_of_floats):
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def int64_feature(list_of_ints):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def get_reduced_length(length, reduction_factor):
  tf.logging.info("** length **")
  tf.logging.info(length)

  tf.logging.info("** reduction_factor **")
  tf.logging.info(reduction_factor)

  length = tf.cast(length, dtype=tf.float32)
  reduced_length = tf.math.ceil(length/reduction_factor)
  reduced_length = tf.cast(reduced_length, dtype=tf.int32)
  return reduced_length
  # return tf.cast(tf.math.ceil(tf.divide(length, tf.cast(reduction_factor, dtype=length.dtype))), dtype=tf.int32)
	