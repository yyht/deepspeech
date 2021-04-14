from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

"""
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/ngram.py
"""

def Ngram(inputs, input_dim, minval, maxval):

  """
  The layer takes as input an integer Tensor of shape [..., length], each
  element of which is a token index in [0, input_dim). It returns a real-valued
  Tensor of shape [..., num_ngrams], counting the number of times each n-gram
  appears in a batch element. The total number of n-grams is
  ```none
  num_ngrams = \\sum_{minval <= n < maxval} input_dim^n.
  """

  batch_shape = tf.shape(inputs)[:-1]
  length = tf.shape(inputs)[-1]
  ngram_range_counts = []
  for n in range(minval, maxval):
    # Reshape inputs from [..., length] to [..., 1, length // n, n], dropping
    # remainder elements. Each n-vector is an ngram.
    reshaped_inputs = tf.reshape(
        inputs[..., :(n * (length // n))],
        tf.concat([batch_shape, [1], (length // n)[tf.newaxis], [n]], 0))
    # Count the number of times each ngram appears in the input. We do so by
    # checking whether each n-vector in the input is equal to each n-vector
    # in a Tensor of all possible ngrams. The comparison is batched between
    # the input Tensor of shape [..., 1, length // n, n] and the ngrams Tensor
    # of shape [..., input_dim**n, 1, n].
    ngrams = tf.reshape(
        list(np.ndindex((input_dim,) * n)),
        [1] * (len(inputs.shape)-1) + [input_dim**n, 1, n])
    is_ngram = tf.equal(
        tf.reduce_sum(tf.cast(tf.equal(reshaped_inputs, ngrams), tf.int32),
                      axis=-1),
        n)
    ngram_counts = tf.reduce_sum(tf.cast(is_ngram, tf.float32), axis=-1)
    ngram_range_counts.append(ngram_counts)
  return tf.concat(ngram_range_counts, axis=-1)

