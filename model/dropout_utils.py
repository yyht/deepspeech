
import collections
import copy
import json
import math
import re
import six
import tensorflow as tf
import numpy as np

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
  if isinstance(tensor, np.ndarray) or isinstance(tensor, list):
    shape = np.array(tensor).shape
    if isinstance(expected_rank, six.integer_types):
      assert len(shape) == expected_rank
    elif expected_rank is not None:
      assert len(shape) in expected_rank
    return shape

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


class DropoutContext(object):
  def __init__(self):
    self.dropout = 0
    self.mask = None
    self.scale = 1
    self.reuse_mask = True
    self.noise_shape = None
    self.seed = None

class XDropout(object):
  def get_mask(self, input_tensor, local_context):

    if not isinstance(local_context, DropoutContext):
      dropout = local_context
      mask = None
      noise_shape = None
      seed = None
      tf.logging.info("==not reuse dropout mask==")
    else:
      dropout = local_context.dropout
      dropout *= local_context.scale
      mask = local_context.mask if local_context.reuse_mask else None
      noise_shape = local_context.noise_shape
      seed = local_context.seed
      tf.logging.info("==reuse dropout mask==")

    if dropout > 0 and mask is None:
      if not noise_shape:
        noise_shape = get_shape_list(input_tensor)
      random_tensor = tf.random_uniform(
          noise_shape, seed=seed, 
          dtype=input_tensor.dtype)
      mask = tf.cast(random_tensor > dropout, dtype=tf.float32)
      tf.logging.info("==generate new mask==")

    if isinstance(local_context, DropoutContext):
      if local_context.mask is None:
        local_context.mask = mask
        tf.logging.info("==push mask==")
      if local_context.noise_shape is None:
        local_context.noise_shape = noise_shape
        tf.logging.info("==push noise shape==")

    return mask, dropout

  def dropout(self, input_tensor, local_context):
    mask, dropout = self.get_mask(input_tensor, local_context)
    scale = 1.0 / (1.0-dropout)
    if dropout > 0:
      output = input_tensor * scale * mask
    else:
      output = input_tensor
    return output

class ReuseDropout(object):
  def __init__(self):
    self.context_stack = {}

  def get_context(self, dropout_prob,
            context_name=None,
            noise_shape=None,
            seed=None):
    if context_name:
      if context_name not in self.context_stack:
        self.context_stack[context_name] = DropoutContext()
        tf.logging.info("==add new dropout context: %s==" % (context_name))
      ctx = self.context_stack[context_name]
      ctx.dropout = dropout_prob
      ctx.noise_shape = noise_shape
      ctx.seed = seed
      return ctx
    else:
      return dropout_prob

  def dropout(self, input_tensor, dropout_prob, 
            context_name=None,
            noise_shape=None,
            seed=None):
    if dropout_prob > 0:
      dropout_fn = XDropout()
      output = dropout_fn.dropout(input_tensor, 
          self.get_context(dropout_prob,
                  context_name,
                  noise_shape,
                  seed))
      return output
    else:
      return input_tensor
            

