
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf

from model import dropout_utils
stable_dropout = dropout_utils.ReuseDropout()

def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.
  Args:
    activation_string: String name of the activation function.
  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.
  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)

def dropout(input_tensor, dropout_prob, dropout_name=None):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """

  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor
  if dropout_name:
    output = stable_dropout.dropout(input_tensor, dropout_prob, dropout_name)
  else:
    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output

  # if dropout_prob is None or dropout_prob == 0.0:
  #   return input_tensor

  # output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  # return output


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None, dropout_name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob, dropout_name=dropout_name)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.
  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].
  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def _generate_relative_positions_matrix(length, max_relative_position,
                                        num_buckets=32,
                                        cache=False,
                                        bidirectional=True):
  """Generates matrix of relative positions between inputs."""
  if not cache:
    range_vec = tf.range(length)

    q_idxs = tf.expand_dims(range_vec, 1)
    v_idxs = tf.expand_dims(range_vec, 0)

    distance_mat = v_idxs - q_idxs
    # range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
    # distance_mat = range_mat - tf.transpose(range_mat)
  else:
    distance_mat = tf.expand_dims(tf.range(-length+1, 1, 1), 0)
  distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                          max_relative_position)
  # Shift values to be >= 0. Each integer still uniquely identifies a relative
  # position difference.
  final_mat = distance_mat_clipped + max_relative_position
  return final_mat

def _generate_relative_positions_matrix_t5(length, max_relative_position,
                                        num_buckets=32,
                                        cache=False,
                                        bidirectional=True):

  if not cache:
    range_vec = tf.range(length)

    q_idxs = tf.expand_dims(range_vec, 1)
    v_idxs = tf.expand_dims(range_vec, 0)

    distance_mat = v_idxs - q_idxs
    # range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
    # distance_mat = range_mat - tf.transpose(range_mat)
  else:
    distance_mat = tf.expand_dims(tf.range(-length+1, 1, 1), 0)

  num_buckets = num_buckets
  max_distance = max_relative_position
  ret = 0
  n = -distance_mat
  if bidirectional:
    num_buckets //= 2
    ret += tf.cast(tf.less(n, 0), 'int32') * num_buckets
    n = tf.abs(n)
  else:
    n = tf.maximum(n, 0)
  # now n is in the range [0, inf)
  max_exact = num_buckets // 2
  is_small = tf.less(n, max_exact)
  val_if_large = max_exact + tf.cast(
      tf.log(tf.cast(n, dtype=tf.float32) / max_exact) /
      tf.log(max_distance / max_exact) * (num_buckets - max_exact),
      'int32',
  )
  val_if_large = tf.minimum(val_if_large, num_buckets - 1)
  tf_switch = (tf.cast(is_small, dtype=tf.int32)) * n + (1-tf.cast(is_small, dtype=tf.int32)) * val_if_large
  ret += tf_switch #tf.switch(is_small, n, val_if_large)
  # ret += tf.where(is_small, n, val_if_large)
  return ret


def _generate_relative_positions_embeddings(length, depth,
                            max_relative_position, name,
                            num_buckets=32,
                            initializer_range=0.02,
                            cache=False,
                            bidirectional=True,
                            relative_position_type='relative_normal',
                            relative_position_embedding_type='sinusoidal'):
  """
  Generates tensor of size [1 if cache else length, length, depth].
  example:
      # `relation_keys` = [F|T, F|T, H]
         relations_keys = _generate_relative_positions_embeddings(
      to_seq_length, size_per_head, max_relative_position, "relative_positions_keys",
      cache=False)
    relations_keys = tf.saturate_cast(relations_keys, compute_type)
  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`
    length = to_seq_length
    depth = size_per_head
    max_relative_position
    name = "relative_positions_keys"
  """
 # '''
  #with tf.variable_scope(name):
  if relative_position_type == 'relative_normal':
    relative_positions_matrix = _generate_relative_positions_matrix(
        length, max_relative_position, cache=cache,
        bidirectional=bidirectional)
    vocab_size = max_relative_position * 2 + 1
  elif relative_position_type == 'relative_t5':
    relative_positions_matrix = _generate_relative_positions_matrix_t5(
        length, max_relative_position, 
        num_buckets=num_buckets,
        cache=cache,
        bidirectional=bidirectional)
    vocab_size = num_buckets
    # Generates embedding for each relative position of dimension depth.
  embeddings_table = np.zeros([vocab_size, depth]).astype(np.float32)

  if relative_position_embedding_type == 'sinusoidal':
    for pos in range(vocab_size):
      for i in range(depth // 2):
        embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
        embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))
  
    relative_position_table = tf.get_variable(name="relative_position_bias", 
                      shape=[vocab_size, depth], 
                      initializer=tf.constant_initializer(embeddings_table, dtype=tf.float32),
                      trainable=False)
  elif relative_position_embedding_type == 'sinusoidal_trainable':
    for pos in range(vocab_size):
      for i in range(depth // 2):
        embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
        embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))
  
    relative_position_table = tf.get_variable(name="relative_position_bias", 
                      shape=[vocab_size, depth], 
                      initializer=tf.constant_initializer(embeddings_table, dtype=tf.float32),
                      trainable=True)
  elif relative_position_embedding_type == 'trainable':
    relative_position_table = tf.get_variable(name="relative_position_bias", 
                      shape=[vocab_size, depth], 
                      initializer=create_initializer(initializer_range),
                      trainable=True)

  relative_position_embeddings = tf.gather(relative_position_table, relative_positions_matrix)
  return relative_position_embeddings, relative_position_table

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    use_relative_position=False,
                    relative_position_embeddings=None,
                    relative_position_type='relative_normal',
                    dropout_name=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.
  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.
  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].
  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.
  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.
  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).
  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  tf.logging.info("*** from_shape ***")
  tf.logging.info(from_shape)

  tf.logging.info("*** to_shape ***")
  tf.logging.info(to_shape)

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  tf.logging.info("*** from_tensor_2d ***")
  tf.logging.info(from_tensor_2d)

  tf.logging.info("*** to_tensor_2d ***")
  tf.logging.info(to_tensor_2d)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  tf.logging.info("*** query_layer ***")
  tf.logging.info(query_layer)

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  tf.logging.info("*** key_layer ***")
  tf.logging.info(key_layer)

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  tf.logging.info("*** value_layer ***")
  tf.logging.info(value_layer)

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  tf.logging.info("*** query_layer ***")
  tf.logging.info(query_layer)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  tf.logging.info("*** key_layer ***")
  tf.logging.info(key_layer)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  
  tf.logging.info("*** attention_scores ***")
  tf.logging.info(attention_scores)

  if use_relative_position:
    # print(from_seq_length, "==from_seq_length==")
    # print(to_seq_length, "==to_seq_length==")
    # assert from_seq_length == to_seq_length
    if relative_position_type == 'relative_normal':
      # max_relative_position = 64
      # `relation_keys` = [F|T, F|T, H]
      relations_keys = tf.identity(relative_position_embeddings)
      # query_layer_t is [F, B, N, H]
      query_layer_t = tf.transpose(query_layer, [2, 0, 1, 3])
      # query_layer_r is [F, B * N, H]
      query_layer_r = tf.reshape(query_layer_t, [from_seq_length, batch_size * num_attention_heads, size_per_head])
      # key_position_scores is [F, B * N, F|T]
      key_position_scores = tf.matmul(query_layer_r, relations_keys, transpose_b=True)
      # key_position_scores_r is [F, B , N, F|T]
      key_position_scores_r = tf.reshape(key_position_scores, [from_seq_length, batch_size, num_attention_heads, from_seq_length])
      # key_position_scores_r_t is [B, N, F, F|T]
      key_position_scores_r_t = tf.transpose(key_position_scores_r, [1, 2, 0, 3])
      attention_scores = attention_scores + key_position_scores_r_t
      tf.logging.info("**** apply nazhe-relative position bias on attention_scores ****")
    elif relative_position_type == 'relative_t5':
      # relative_position_embeddings: [F, T, N]--> [N, F, T]
      relative_position_embeddings = tf.transpose(relative_position_embeddings, [2,0,1])
      # relative_position_embeddings: [N, F, T] ---> [1, N, F, T]
      tf.logging.info("***** apply t5-relative position bias on attention_scores ***")
      attention_scores += tf.expand_dims(relative_position_embeddings, axis=0)
    
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  tf.logging.info("*** attention_scores ***")
  tf.logging.info(attention_scores)

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

    tf.logging.info("*** adder ***")
    tf.logging.info(adder)

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  tf.logging.info("*** attention_probs ***")
  tf.logging.info(attention_probs)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob, dropout_name=dropout_name)

  tf.logging.info("*** attention_probs ***")
  tf.logging.info(attention_probs)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  tf.logging.info("*** value_layer ***")
  tf.logging.info(value_layer)

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  tf.logging.info("*** value_layer ***")
  tf.logging.info(value_layer)

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  tf.logging.info("*** context_layer ***")
  tf.logging.info(context_layer)

  if use_relative_position:
    if relative_position_type == 'relative_normal':
      # `relation_values` = [F|T, F|T, H]
      relations_values = tf.identity(relative_position_embeddings)
      # attention_probs_t is [F, B, N, T]
      attention_probs_t = tf.transpose(attention_probs, [2, 0, 1, 3])
      # attention_probs_r is [F, B * N, T]
      attention_probs_r = tf.reshape(attention_probs_t, [from_seq_length, batch_size * num_attention_heads, to_seq_length])
      # key_position_scores is [F, B * N, H]
      value_position_scores = tf.matmul(attention_probs_r, relations_values, transpose_b=False)
      # value_position_scores_r is [F, B , N, H]
      value_position_scores_r = tf.reshape(value_position_scores, [from_seq_length, batch_size, num_attention_heads, size_per_head])
      # value_position_scores_r_t is [B, N, F, H]
      value_position_scores_r_t = tf.transpose(value_position_scores_r, [1, 2, 0, 3])
      # attention_scores = attention_scores + value_position_scores_r_t
      context_layer = context_layer + value_position_scores_r_t
      tf.logging.info("*** apply nazhe-relative position bias on attention_scores ****")

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  tf.logging.info("*** context_layer ***")
  tf.logging.info(context_layer)

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer, attention_scores


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      use_relative_position=False,
                      dropout_name=None,
                      relative_position_embeddings=None):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".
  This is almost an exact implementation of the original Transformer encoder.
  See the original paper:
  https://arxiv.org/abs/1706.03762
  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.
  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.
  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))
  tf.logging.info('use_relative_position: %s' % use_relative_position)

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  attn_maps = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):

          if dropout_name:
            attention_dropout_name = tf.get_variable_scope().name
          else:
            attention_dropout_name = None

          attention_head, probs = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length,
              use_relative_position=use_relative_position,
              dropout_name=attention_dropout_name,
              relative_position_embeddings=relative_position_embeddings)

          attention_heads.append(attention_head)
          attn_maps.append(probs)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          
          if dropout_name:
            output_dropout_name = tf.get_variable_scope().name
          else:
            output_dropout_name = None

          attention_output = dropout(attention_output, hidden_dropout_prob, dropout_name=output_dropout_name)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))

        if dropout_name:
          ffn_dropout_name = tf.get_variable_scope().name
        else:
          ffn_dropout_name = None

        layer_output = dropout(layer_output, hidden_dropout_prob, dropout_name=ffn_dropout_name)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  attn_maps = tf.stack(attn_maps, 0)
  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs, attn_maps
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output, attn_maps

def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
