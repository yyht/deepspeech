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


def get_labels_of_similarity(query_input_ids, anchor_query_ids):
  idxs_1 = tf.expand_dims(query_input_ids, axis=1) # batch 1 seq
  idxs_2 = tf.expand_dims(anchor_query_ids, axis=0) # 1 batch seq
  # batch x batch x seq
  labels = tf.cast(tf.not_equal(idxs_1, idxs_2), tf.float32) # not equal:1, equal:0
  equal_num = tf.reduce_sum(labels, axis=-1) # [batch, batch]
  not_equal_label = tf.cast(tf.not_equal(equal_num, 0), tf.float32)
  # equal_label = tf.cast(tf.equal(equal_num, 0), tf.float32)
  not_equal_label_shape = get_shape_list(not_equal_label, expected_rank=[2,3])
  not_equal_label *= tf.cast(1 - tf.eye(not_equal_label_shape[0]), tf.float32) 
  equal_label = 1 - not_equal_label - tf.cast(tf.equal(equal_num, 0), tf.float32)
  return equal_label, not_equal_label

def get_sparse_labels_of_similarity(query_input_ids, anchor_query_ids):
  idxs_1 = tf.expand_dims(query_input_ids, axis=1) # batch 1 seq
  idxs_2 = tf.expand_dims(anchor_query_ids, axis=0) # 1 batch seq
  not_equal_label = tf.cast(tf.not_equal(idxs_1, idxs_2), tf.float32) # not equal:1, equal:0
  equal_label = tf.cast(tf.equal(idxs_1, idxs_2), tf.float32) # not equal:1, equal:0
  equal_label_shape = get_shape_list(equal_label, expected_rank=[2,3])
  equal_label_with_self = tf.cast(tf.eye(equal_label_shape[0]), dtype=tf.float32) * equal_label
  equal_label_without_self = equal_label - equal_label_with_self
  return equal_label_without_self, not_equal_label

def circle_loss(pair_wise_cosine_matrix, 
                pred_true_mask, 
                pred_neg_mask,
                margin=0.25,
                gamma=64):
  """
  https://github.com/zhen8838/Circle-Loss/blob/master/circle_loss.py
  """
  O_p = 1 + margin
  O_n = -margin

  Delta_p = 1 - margin
  Delta_n = margin

  ap = tf.nn.relu(-tf.stop_gradient(pair_wise_cosine_matrix) + 1 + margin)
  an = tf.nn.relu(tf.stop_gradient(pair_wise_cosine_matrix) + margin)

  logit_p = -ap * (pair_wise_cosine_matrix - Delta_p) * gamma * pred_true_mask
  logit_n = an * (pair_wise_cosine_matrix - Delta_n) * gamma * pred_neg_mask

  logit_p = logit_p -  (1 - pred_true_mask) * 1e12
  logit_n = logit_n - (1 - pred_neg_mask) * 1e12

  joint_neg_loss = tf.reduce_logsumexp(logit_n, axis=-1)
  joint_pos_loss = tf.reduce_logsumexp(logit_p, axis=-1)
  per_example_loss = tf.nn.softplus(joint_neg_loss+joint_pos_loss)
  return per_example_loss

def matching_embedding_hinge_loss_v2(emb1, emb2, labels, margin=0.3):
  dis = tf.reduce_sum(emb1*emb2, axis=-1)
  labels = tf.cast(labels, dtype=tf.float32)

  all_ones = tf.ones_like(labels)
  
  per_example_negative_loss = (1 - labels) * tf.nn.relu(dis)
  per_example_positive_loss = labels * (1.0 - dis)

  per_example_loss = per_example_negative_loss + per_example_positive_loss
  return per_example_loss

def matching_embedding_hinge_loss(emb1, emb2, labels, margin=0.3):
  dis = tf.reduce_sum(emb1*emb2, axis=-1)
  labels = tf.cast(labels, dtype=tf.float32)
  
  per_example_loss = tf.abs(labels - dis)
  return per_example_loss

def matching_embedding_mse_loss(emb1, emb2, labels, margin=0.3):
  dis = tf.reduce_sum(emb1*emb2, axis=-1)
  labels = tf.cast(labels, dtype=tf.float32)
  
  per_example_loss = tf.pow(labels - dis, 2.0)
  return per_example_loss

def matching_embedding_contrastive_loss(emb1, emb2, labels, margin=0.3):
  dis = tf.reduce_sum(emb1*emb2, axis=-1)
  dis = 1 - dis
  labels = tf.cast(labels, dtype=tf.float32)

  per_example_loss = 0.5 * (labels * tf.pow(dis, 2) + (1 - labels) * tf.pow(tf.nn.relu(margin - dis), 2))

  return per_example_loss


def matching_embedding_hinge_loss_v1(emb1, emb2, labels, margin=0.3):
  dis = tf.reduce_sum(emb1*emb2, axis=-1)
  labels = tf.cast(labels, dtype=tf.float32)

  all_ones = tf.ones_like(labels)

  labels = 2 * labels - all_ones
  per_example_loss = tf.nn.relu(all_ones - labels*dis)

  return per_example_loss

def circle_loss_pairwise(y_pred, labels, margin=0.25,
                              gamma=64):

  """
  https://github.com/zhen8838/Circle-Loss/blob/master/circle_loss.py
  """

  O_p = 1 + margin
  O_n = -margin

  Delta_p = 1 - margin
  Delta_n = margin

  alpha_p = tf.nn.relu(O_p - tf.stop_gradient(y_pred))
  alpha_n = tf.nn.relu(tf.stop_gradient(y_pred) - O_n)

  labels = tf.cast(labels, tf.float32)
  y_pred_true = labels * (alpha_p * (y_pred - Delta_p))
  y_pred_fake = (1 - labels) * (alpha_n * (y_pred - Delta_n))

  y_pred_output = (y_pred_true + y_pred_fake) * gamma
  per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_pred_output)

  return per_example_loss