# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_functional_ops
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.ops import functional_ops
from loss.map_fn import map_fn

try:
  tf.logging.info("gen_ctc_ops.ctc_loss_v2")
  tf.logging.info(gen_ctc_ops.ctc_loss_v2)
except:
  tf.logging.info("invalid gen_ctc_ops.ctc_loss_v2")

def _ctc_state_trans(label_seq):
  """Compute CTC alignment model transition matrix.
  Args:
    label_seq: tensor of shape [batch_size, max_seq_length]
  Returns:
    tensor of shape [batch_size, states, states] with a state transition matrix
    computed for each sequence of the batch.
  """

  with ops.name_scope("ctc_state_trans"):
    label_seq = ops.convert_to_tensor(label_seq, name="label_seq")
    batch_size = _get_dim(label_seq, 0)
    num_labels = _get_dim(label_seq, 1)

    num_label_states = num_labels + 1
    num_states = 2 * num_label_states

    label_states = math_ops.range(num_label_states)
    blank_states = label_states + num_label_states

    # Start state to first label.
    start_to_label = [[1, 0]]

    # Blank to label transitions.
    blank_to_label = array_ops.stack([label_states[1:], blank_states[:-1]], 1)

    # Label to blank transitions.
    label_to_blank = array_ops.stack([blank_states, label_states], 1)

    # Scatter transitions that don't depend on sequence.
    indices = array_ops.concat([start_to_label, blank_to_label, label_to_blank],
                               0)
    values = array_ops.ones([_get_dim(indices, 0)])
    trans = array_ops.scatter_nd(
        indices, values, shape=[num_states, num_states])
    trans += linalg_ops.eye(num_states)  # Self-loops.

    # Label to label transitions. Disallow transitions between repeated labels
    # with no blank state in between.
    batch_idx = array_ops.zeros_like(label_states[2:])
    indices = array_ops.stack([batch_idx, label_states[2:], label_states[1:-1]],
                              1)
    indices = array_ops.tile(
        array_ops.expand_dims(indices, 0), [batch_size, 1, 1])
    batch_idx = array_ops.expand_dims(math_ops.range(batch_size), 1) * [1, 0, 0]
    indices += array_ops.expand_dims(batch_idx, 1)
    repeats = math_ops.equal(label_seq[:, :-1], label_seq[:, 1:])
    values = 1.0 - math_ops.cast(repeats, dtypes.float32)
    batched_shape = [batch_size, num_states, num_states]
    label_to_label = array_ops.scatter_nd(indices, values, batched_shape)

    return array_ops.expand_dims(trans, 0) + label_to_label


def ctc_state_log_probs(seq_lengths, max_seq_length):
  """Computes CTC alignment initial and final state log probabilities.
  Create the initial/final state values directly as log values to avoid
  having to take a float64 log on tpu (which does not exist).
  Args:
    seq_lengths: int tensor of shape [batch_size], seq lengths in the batch.
    max_seq_length: int, max sequence length possible.
  Returns:
    initial_state_log_probs, final_state_log_probs
  """

  batch_size = _get_dim(seq_lengths, 0)
  num_label_states = max_seq_length + 1
  num_duration_states = 2
  num_states = num_duration_states * num_label_states
  log_0 = math_ops.cast(
      math_ops.log(math_ops.cast(0, dtypes.float64) + 1e-307), dtypes.float32)

  initial_state_log_probs = array_ops.one_hot(
      indices=array_ops.zeros([batch_size], dtype=dtypes.int32),
      depth=num_states,
      on_value=0.0,
      off_value=log_0,
      axis=1)

  label_final_state_mask = array_ops.one_hot(
      seq_lengths, depth=num_label_states, axis=0)
  duration_final_state_mask = array_ops.ones(
      [num_duration_states, 1, batch_size])
  final_state_mask = duration_final_state_mask * label_final_state_mask
  final_state_log_probs = (1.0 - final_state_mask) * log_0
  final_state_log_probs = array_ops.reshape(final_state_log_probs,
                                            [num_states, batch_size])

  return initial_state_log_probs, array_ops.transpose(final_state_log_probs)


def _ilabel_to_state(labels, num_labels, ilabel_log_probs):
  """Project ilabel log probs to state log probs."""

  num_label_states = _get_dim(labels, 1)
  blank = ilabel_log_probs[:, :, :1]
  blank = array_ops.tile(blank, [1, 1, num_label_states + 1])
  one_hot = array_ops.one_hot(labels, depth=num_labels)
  one_hot = array_ops.expand_dims(one_hot, axis=0)
  ilabel_log_probs = array_ops.expand_dims(ilabel_log_probs, axis=2)
  state_log_probs = math_ops.reduce_sum(ilabel_log_probs * one_hot, axis=3)
  state_log_probs = array_ops.concat([state_log_probs, blank], axis=2)
  return array_ops.pad(
      state_log_probs, [[0, 0], [0, 0], [1, 0]],
      constant_values=math_ops.log(0.0))


def _state_to_olabel(labels, num_labels, states):
  """Sum state log probs to ilabel log probs."""

  num_label_states = _get_dim(labels, 1) + 1
  label_states = states[:, :, 1:num_label_states]
  blank_states = states[:, :, num_label_states:]
  one_hot = array_ops.one_hot(
      labels - 1,
      depth=(num_labels - 1),
      on_value=0.0,
      off_value=math_ops.log(0.0))
  one_hot = array_ops.expand_dims(one_hot, axis=0)
  label_states = array_ops.expand_dims(label_states, axis=3)
  label_olabels = math_ops.reduce_logsumexp(label_states + one_hot, axis=2)
  blank_olabels = math_ops.reduce_logsumexp(blank_states, axis=2, keepdims=True)
  return array_ops.concat([blank_olabels, label_olabels], axis=-1)


# pylint: disable=redefined-outer-name
def _state_to_olabel_unique(labels, num_labels, states, unique):
  """Sum state log probs to ilabel log probs using unique label indices."""

  num_label_states = _get_dim(labels, 1) + 1
  label_states = states[:, :, 1:num_label_states]
  blank_states = states[:, :, num_label_states:]

  unique_y, unique_idx = unique
  mul_reduce = _sum_states(unique_idx, label_states)

  num_frames = states.shape[0]
  batch_size = states.shape[1]
  num_states = num_label_states - 1
  batch_state_major = array_ops.transpose(mul_reduce, perm=[1, 2, 0])
  batch_state_major = array_ops.reshape(batch_state_major,
                                        [batch_size * num_states, num_frames])
  batch_offset = math_ops.range(batch_size, dtype=unique_y.dtype) * num_labels
  indices = unique_y + array_ops.expand_dims(batch_offset, axis=-1)
  indices = array_ops.reshape(indices, [-1, 1])
  scatter = array_ops.scatter_nd(
      indices=indices,
      updates=batch_state_major,
      shape=[batch_size * num_labels, num_frames])
  scatter = array_ops.reshape(scatter, [batch_size, num_labels, num_frames])
  scatter = array_ops.where(
      math_ops.equal(scatter, 0.0),
      array_ops.fill(array_ops.shape(scatter), math_ops.log(0.0)), scatter)
  label_olabels = array_ops.transpose(scatter, [2, 0, 1])
  label_olabels = label_olabels[:, :, 1:]

  blank_olabels = math_ops.reduce_logsumexp(blank_states, axis=2, keepdims=True)

  return array_ops.concat([blank_olabels, label_olabels], axis=-1)


def ctc_loss_and_grad(logits, labels, label_length, logit_length, unique=None):
  """Computes the CTC loss and gradients.
  Most users will want fwd_bwd.ctc_loss
  This function returns the computed gradient, it does not have a gradient
  of its own defined.
  Args:
    logits: tensor of shape [frames, batch_size, num_labels]
    labels: tensor of shape [batch_size, max_label_seq_length]
    label_length: tensor of shape [batch_size] Length of reference label
      sequence in labels.
    logit_length: tensor of shape [batch_size] Length of input sequence in
      logits.
    unique: (optional) unique label indices as computed by unique(labels) If
      supplied, enables an implementation that is faster and more memory
      efficient on TPU.
  Returns:
    loss: tensor of shape [batch_size]
    gradient: tensor of shape [frames, batch_size, num_labels]
  """

  num_labels = _get_dim(logits, 2)
  max_label_seq_length = _get_dim(labels, 1)

  ilabel_log_probs = nn_ops.log_softmax(logits)
  state_log_probs = _ilabel_to_state(labels, num_labels, ilabel_log_probs)
  state_trans_probs = _ctc_state_trans(labels)
  initial_state_log_probs, final_state_log_probs = ctc_state_log_probs(
      label_length, max_label_seq_length)
  fwd_bwd_log_probs, log_likelihood = _forward_backward_log(
      state_trans_log_probs=math_ops.log(state_trans_probs),
      initial_state_log_probs=initial_state_log_probs,
      final_state_log_probs=final_state_log_probs,
      observed_log_probs=state_log_probs,
      sequence_length=logit_length)

  if unique:
    olabel_log_probs = _state_to_olabel_unique(labels, num_labels,
                                               fwd_bwd_log_probs, unique)
  else:
    olabel_log_probs = _state_to_olabel(labels, num_labels, fwd_bwd_log_probs)

  grad = math_ops.exp(ilabel_log_probs) - math_ops.exp(olabel_log_probs)
  loss = -log_likelihood
  return loss, grad


def _ctc_loss_grad(op, grad_loss, _):
  grad = op.outputs[1]
  grad = [array_ops.reshape(grad_loss, [1, -1, 1]) * grad]
  grad += [None] * (len(op.inputs) - len(grad))
  return grad


def _ctc_loss_shape(op):
  return [op.inputs[2].get_shape(), op.inputs[0].get_shape()]

def ctc_loss_v2(labels,
                logits,
                label_length,
                logit_length,
                logits_time_major=True,
                unique=None,
                blank_index=None,
                name=None):
  """Computes CTC (Connectionist Temporal Classification) loss.
  This op implements the CTC loss as presented in the article:
  [A. Graves, S. Fernandez, F. Gomez, J. Schmidhuber.
  Connectionist Temporal Classification: Labeling Unsegmented Sequence Data
  with Recurrent Neural Networks. ICML 2006, Pittsburgh, USA,
  pp. 369-376.](http://www.cs.toronto.edu/~graves/icml_2006.pdf)
  Notes:
    Fixed formatting of ctc_loss_v2 docstring.
  - Same as the "Classic CTC" in TensorFlow 1.x's tf.compat.v1.nn.ctc_lossMark Daoust, 
    setting of preprocess_collapse_repeated=False, ctc_merge_repeated=TrueA. Unique TensorFlower, 
  - Labels may be supplied as either a dense, zero-padded tensor with aA. Unique TensorFlower, 
    vector of label sequence lengths OR as a SparseTensor.
  - On TPU and GPU: Only dense padded labels are supported.
  - On CPU: Caller may use SparseTensor or dense padded labels but calling with
    a SparseTensor will be significantly faster. 
  - Default blank label is 0 rather num_classes - 1, unless overridden by
    blank_index.
  Args:
    labels: tensor of shape [batch_size, max_label_seq_length] or SparseTensor
    logits: tensor of shape [frames, batch_size, num_labels], 
      logits_time_major == False, shape is [batch_size, frames, num_labels].
    label_length: tensor of shape [batch_size], None if labels is SparseTensorA. Unique TensorFlower, 
      Length of reference label sequence in labels.
    logit_length: tensor of shape [batch_size] Length of input sequence inMark Daoust, 
      logits.
    logits_time_major: (optional) If True (default), logits is shaped [time,
      batch, logits]. If False, shape is [batch, time, logits]
    unique: (optional) Unique label indices as computed byA. Unique TensorFlower, 
      ctc_unique_labels(labels).  If supplied, enable a faster, memory efficient
      implementation on TPU.
    blank_index: (optional) Set the class index to use for the blank label.
      Negative values will start from num_classes, ie, -1 will reproduce the
      ctc_loss behavior of using num_classes - 1 for the blank symbol. There is
      some memory/performance overhead to switching from the default of 0 as an
      additional shifted copy of the logits may be created.
    name: A name for this `Op`. Defaults to "ctc_loss_dense".
  Returns:
    loss: tensor of shape [batch_size], negative log probabilities.
  """

  if blank_index is None:
    blank_index = 0

  return ctc_loss_dense(
      labels=labels,
      logits=logits,
      label_length=label_length,
      logit_length=logit_length,
      logits_time_major=logits_time_major,
      unique=unique,
      blank_index=blank_index,
      name=name)


def ctc_loss_dense(labels,
                   logits,
                   label_length,
                   logit_length,
                   logits_time_major=True,
                   unique=None,
                   blank_index=0,
                   name=None):
  """Computes CTC (Connectionist Temporal Classification) loss.
  This op implements the CTC loss as presented in the article:
  [A. Graves, S. Fernandez, F. Gomez, J. Schmidhuber.
  Connectionist Temporal Classification: Labeling Unsegmented Sequence Data
  with Recurrent Neural Networks. ICML 2006, Pittsburgh, USA,
  pp. 369-376.](http://www.cs.toronto.edu/~graves/icml_2006.pdf)
  Using the batched forward backward algorithm described in:
  [Sim, K. C., Narayanan, A., Bagby, T., Sainath, T. N., & Bacchiani, M.
  Improving the efficiency of forward-backward algorithm using batched
    computation in TensorFlow.
  Automatic Speech Recognition and Understanding Workshop (ASRU),
    2017 IEEE (pp. 258-264).
  ](https://ieeexplore.ieee.org/iel7/8260578/8268903/08268944.pdf)
  Notes:
    Significant differences from tf.compat.v1.nn.ctc_loss:
      Supports GPU and TPU (tf.compat.v1.nn.ctc_loss supports CPU only):
        For batched operations, GPU and TPU are significantly faster than using
        ctc_loss on CPU.
        This implementation runs on CPU, but significantly slower than ctc_loss.
      Blank label is 0 rather num_classes - 1, unless overridden by blank_index.
      Logits and labels are dense arrays with padding rather than SparseTensor.
      The only mode supported is the same as:
        preprocess_collapse_repeated=False, ctc_merge_repeated=True
         To collapse labels, the caller can preprocess label sequence first.
    The dense implementation supports both CPU, GPU and TPU. A fast path is
    provided that significantly improves memory use for large vocabulary if the
    caller preprocesses label sequences to get unique label indices on the CPU
    (eg. in the data input pipeline) using ctc_ops.unique and simplies this in
    the optional "unique" kwarg. This is especially useful for TPU and GPU but
    also works with if used on CPU.
  Args:
    labels: tensor of shape [batch_size, max_label_seq_length]
    logits: tensor of shape [frames, batch_size, num_labels], if
      logits_time_major == False, shape is [batch_size, frames, num_labels].
    label_length: tensor of shape [batch_size] Length of reference label
      sequence in labels.
    logit_length: tensor of shape [batch_size] Length of input sequence in
      logits.
    logits_time_major: (optional) If True (default), logits is shaped [time,
      batch, logits]. If False, shape is [batch, time, logits]
    unique: (optional) Unique label indices as computed by unique(labels). If
      supplied, enable a faster, memory efficient implementation on TPU.
    blank_index: (optional) Set the class index to use for the blank label.
      Negative values will start from num_classes, ie, -1 will reproduce the
      ctc_loss behavior of using num_classes - 1 for the blank symbol. There is
      some memory/performance overhead to switching from the default of 0 as an
      additional shifted copy of the logits may be created.
    name: A name for this `Op`. Defaults to "ctc_loss_dense".
  Returns:
    loss: tensor of shape [batch_size], negative log probabilities.
  """

  with ops.name_scope(name, "ctc_loss_dense",
                      [logits, labels, label_length, logit_length]):
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    label_length = ops.convert_to_tensor(label_length, name="label_length")
    logit_length = ops.convert_to_tensor(logit_length, name="logit_length")

    if not logits_time_major:
      logits = array_ops.transpose(logits, perm=[1, 0, 2])

    if blank_index != 0:
      if blank_index < 0:
        blank_index += _get_dim(logits, 2)
      logits = array_ops.concat([
          logits[:, :, blank_index:blank_index + 1],
          logits[:, :, :blank_index],
          logits[:, :, blank_index + 1:],
      ],
                                axis=2)
      labels = array_ops.where(labels < blank_index, labels + 1, labels)

    args = [logits, labels, label_length, logit_length]

    if unique is not None:
      unique_y, unique_idx = unique
      args.extend([unique_y, unique_idx])

    @custom_gradient.custom_gradient
    def compute_ctc_loss(logits_t, labels_t, label_length_t, logit_length_t,
                         *unique_t):
      """Compute CTC loss."""
      logits_t.set_shape(logits.shape)
      labels_t.set_shape(labels.shape)
      label_length_t.set_shape(label_length.shape)
      logit_length_t.set_shape(logit_length.shape)
      kwargs = dict(
          logits=logits_t,
          labels=labels_t,
          label_length=label_length_t,
          logit_length=logit_length_t)
      if unique_t:
        kwargs["unique"] = unique_t
      result = ctc_loss_and_grad(**kwargs)
      def grad(grad_loss):
        grad = [array_ops.reshape(grad_loss, [1, -1, 1]) * result[1]]
        grad += [None] * (len(args) - len(grad))
        return grad

      return result[0], grad

    return compute_ctc_loss(*args)

def collapse_repeated(labels, seq_length, name=None):
  """Merge repeated labels into single labels.
  Args:
    labels: Tensor of shape [batch, max value in seq_length]
    seq_length: Tensor of shape [batch], sequence length of each batch element.
    name: A name for this `Op`. Defaults to "collapse_repeated_labels".
  Returns:
    A tuple `(collapsed_labels, new_seq_length)` where
    collapsed_labels: Tensor of shape [batch, max_seq_length] with repeated
    labels collapsed and padded to max_seq_length, eg:
    `[[A, A, B, B, A], [A, B, C, D, E]] => [[A, B, A, 0, 0], [A, B, C, D, E]]`
    new_seq_length: int tensor of shape [batch] with new sequence lengths.
  """

  with ops.name_scope(name, "collapse_repeated_labels", [labels, seq_length]):
    labels = ops.convert_to_tensor(labels, name="labels")
    seq_length = ops.convert_to_tensor(seq_length, name="seq_length")

    # Mask labels that don't equal previous label.
    label_mask = array_ops.concat([
        array_ops.ones_like(labels[:, :1], dtypes.bool),
        math_ops.not_equal(labels[:, 1:], labels[:, :-1])
    ],
                                  axis=1)

    # Filter labels that aren't in the original sequence.
    maxlen = _get_dim(labels, 1)
    seq_mask = array_ops.sequence_mask(seq_length, maxlen=maxlen)
    label_mask = math_ops.logical_and(label_mask, seq_mask)

    # Count masks for new sequence lengths.
    new_seq_len = math_ops.reduce_sum(
        math_ops.cast(label_mask, dtypes.int32), axis=1)

    # Mask indexes based on sequence length mask.
    new_maxlen = math_ops.reduce_max(new_seq_len)
    idx_mask = array_ops.sequence_mask(new_seq_len, maxlen=new_maxlen)

    # Flatten everything and mask out labels to keep and sparse indices.
    flat_labels = array_ops.reshape(labels, [-1])
    flat_label_mask = array_ops.reshape(label_mask, [-1])
    flat_idx_mask = array_ops.reshape(idx_mask, [-1])
    idx = math_ops.range(_get_dim(flat_idx_mask, 0))

    # Scatter to flat shape.
    flat = array_ops.scatter_nd(
        indices=array_ops.expand_dims(
            array_ops.boolean_mask(idx, flat_idx_mask), axis=1),
        updates=array_ops.boolean_mask(flat_labels, flat_label_mask),
        shape=array_ops.shape(flat_idx_mask))

    # Reshape back to square batch.
    batch_size = _get_dim(labels, 0)
    new_shape = [batch_size, new_maxlen]
    return (array_ops.reshape(flat, new_shape),
            math_ops.cast(new_seq_len, seq_length.dtype))


def dense_labels_to_sparse(dense, length):
  """Convert dense labels with sequence lengths to sparse tensor.
  Args:
    dense: tensor of shape [batch, max_length]
    length: int tensor of shape [batch] The length of each sequence in dense.
  Returns:
    tf.SparseTensor with values only for the valid elements of sequences.
  """

  flat_values = array_ops.reshape(dense, [-1])
  flat_indices = math_ops.range(
      array_ops.shape(flat_values, out_type=dtypes.int64)[0])
  mask = array_ops.sequence_mask(length, maxlen=array_ops.shape(dense)[1])
  flat_mask = array_ops.reshape(mask, [-1])
  indices = array_ops.expand_dims(
      array_ops.boolean_mask(flat_indices, flat_mask), 1)
  values = array_ops.boolean_mask(flat_values, flat_mask)
  sparse = sparse_tensor.SparseTensor(
      indices=indices,
      values=math_ops.cast(values, dtypes.int32),
      dense_shape=array_ops.shape(flat_values, out_type=dtypes.int64))
  reshaped = sparse_ops.sparse_reshape(sparse, array_ops.shape(dense))
  max_length = math_ops.reduce_max(length)
  return sparse_tensor.SparseTensor(
      indices=reshaped.indices,
      values=reshaped.values,
      dense_shape=[
          math_ops.cast(reshaped.dense_shape[0], dtypes.int64),
          math_ops.cast(max_length, dtypes.int64)
      ])

def ctc_unique_labels(labels, name=None):
  """Get unique labels and indices for batched labels for `tf.nn.ctc_loss`.
  For use with `tf.nn.ctc_loss` optional argument `unique`: This op can be
  used to preprocess labels in input pipeline to for better speed/memory use
  computing the ctc loss on TPU.
  Example:
    ctc_unique_labels([[3, 4, 4, 3]]) ->
      unique labels padded with 0: [[3, 4, 0, 0]]
      indices of original labels in unique: [0, 1, 1, 0]
  Args:
    labels: tensor of shape [batch_size, max_label_length] padded with 0.
    name: A name for this `Op`. Defaults to "ctc_unique_labels".
  Returns:
    tuple of
      - unique labels, tensor of shape `[batch_size, max_label_length]`
      - indices into unique labels, shape `[batch_size, max_label_length]`
  """

  with ops.name_scope(name, "ctc_unique_labels", [labels]):
    labels = ops.convert_to_tensor(labels, name="labels")

    def _unique(x):
      u = array_ops.unique(x)
      y = array_ops.pad(u.y, [[0, _get_dim(u.idx, 0) - _get_dim(u.y, 0)]])
      y = math_ops.cast(y, dtypes.int64)
      return [y, u.idx]

    return map_fn(_unique, labels, dtype=[dtypes.int64, dtypes.int32])

def _sum_states(idx, states):
  """Take logsumexp for each unique state out of all label states.
  Args:
    idx: tensor of shape [batch, label_length] For each sequence, indices into a
      set of unique labels as computed by calling unique.
    states: tensor of shape [frames, batch, label_length] Log probabilities for
      each label state.
  Returns:
    tensor of shape [frames, batch_size, label_length], log probabilites summed
      for each unique label of the sequence.
  """

  with ops.name_scope("sum_states"):
    idx = ops.convert_to_tensor(idx, name="idx")
    num_states = _get_dim(states, 2)
    states = array_ops.expand_dims(states, axis=2)
    one_hot = array_ops.one_hot(
        idx,
        depth=num_states,
        on_value=0.0,
        off_value=math_ops.log(0.0),
        axis=1)
    return math_ops.reduce_logsumexp(states + one_hot, axis=-1)


def _forward_backward_log(state_trans_log_probs, initial_state_log_probs,
                          final_state_log_probs, observed_log_probs,
                          sequence_length):
  """Forward-backward algorithm computed in log domain.
  Args:
    state_trans_log_probs: tensor of shape [states, states] or if different
      transition matrix per batch [batch_size, states, states]
    initial_state_log_probs: tensor of shape [batch_size, states]
    final_state_log_probs: tensor of shape [batch_size, states]
    observed_log_probs: tensor of shape [frames, batch_size, states]
    sequence_length: tensor of shape [batch_size]
  Returns:
    forward backward log probabilites: tensor of shape [frames, batch, states]
    log_likelihood: tensor of shape [batch_size]
  Raises:
    ValueError: If state_trans_log_probs has unknown or incorrect rank.
  """

  if state_trans_log_probs.shape.ndims == 2:
    perm = [1, 0]
  elif state_trans_log_probs.shape.ndims == 3:
    perm = [0, 2, 1]
  else:
    raise ValueError(
        "state_trans_log_probs rank must be known and == 2 or 3, is: %s" %
        state_trans_log_probs.shape.ndims)

  bwd_state_trans_log_probs = array_ops.transpose(state_trans_log_probs, perm)
  batch_size = _get_dim(observed_log_probs, 1)

  def _forward(state_log_prob, obs_log_prob):
    state_log_prob = array_ops.expand_dims(state_log_prob, axis=1)  # Broadcast.
    state_log_prob += state_trans_log_probs
    state_log_prob = math_ops.reduce_logsumexp(state_log_prob, axis=-1)
    state_log_prob += obs_log_prob
    log_prob_sum = math_ops.reduce_logsumexp(
        state_log_prob, axis=-1, keepdims=True)
    state_log_prob -= log_prob_sum
    return state_log_prob

  fwd = _scan(
      _forward, observed_log_probs, initial_state_log_probs, inclusive=True)

  def _backward(accs, elems):
    """Calculate log probs and cumulative sum masked for sequence length."""
    state_log_prob, cum_log_sum = accs
    obs_log_prob, mask = elems
    state_log_prob += obs_log_prob
    state_log_prob = array_ops.expand_dims(state_log_prob, axis=1)  # Broadcast.
    state_log_prob += bwd_state_trans_log_probs
    state_log_prob = math_ops.reduce_logsumexp(state_log_prob, axis=-1)

    log_prob_sum = math_ops.reduce_logsumexp(
        state_log_prob, axis=-1, keepdims=True)
    state_log_prob -= log_prob_sum

    cum_log_sum += array_ops.squeeze(log_prob_sum) * mask
    batched_mask = array_ops.expand_dims(mask, axis=1)
    out = state_log_prob * batched_mask
    out += final_state_log_probs * (1.0 - batched_mask)
    return out, cum_log_sum

  zero_log_sum = array_ops.zeros([batch_size])
  maxlen = _get_dim(observed_log_probs, 0)
  mask = array_ops.sequence_mask(sequence_length, maxlen, dtypes.float32)
  mask = array_ops.transpose(mask, perm=[1, 0])

  bwd, cum_log_sum = _scan(
      _backward, (observed_log_probs, mask),
      (final_state_log_probs, zero_log_sum),
      reverse=True,
      inclusive=True)

  fwd_bwd_log_probs = fwd[1:] + bwd[1:]
  fwd_bwd_log_probs_sum = math_ops.reduce_logsumexp(
      fwd_bwd_log_probs, axis=2, keepdims=True)
  fwd_bwd_log_probs -= fwd_bwd_log_probs_sum
  fwd_bwd_log_probs += math_ops.log(array_ops.expand_dims(mask, axis=2))

  log_likelihood = bwd[0, :, 0] + cum_log_sum[0]

  return fwd_bwd_log_probs, log_likelihood


# TODO(tombagby): This is currently faster for the ctc implementation than using
# functional_ops.scan, but could be replaced by that or something similar if
# things change.
def _scan(fn, elems, initial, reverse=False, inclusive=False, final_only=False):
  """Repeatedly applies callable `fn` to a sequence of elements.
  Implemented by functional_ops.While, tpu friendly, no gradient.
  This is similar to functional_ops.scan but significantly faster on tpu/gpu
  for the forward backward use case.
  Examples:
    scan(lambda a, e: a + e, [1.0, 2.0, 3.0], 1.0) => [2.0, 4.0, 7.0]
    Multiple accumulators:
      scan(lambda a, e: (a[0] + e, a[1] * e), [1.0, 2.0, 3.0], (0.0, 1.0))
    Multiple inputs:
      scan(lambda a, e: a + (e[0] * e[1]), (elems1, elems2), 0.0)
  Args:
    fn: callable, fn(accumulators, element) return new accumulator values. The
      (possibly nested) sequence of accumulators is the same as `initial` and
      the return value must have the same structure.
    elems: A (possibly nested) tensor which will be unpacked along the first
      dimension. The resulting slices will be the second argument to fn. The
      first dimension of all nested input tensors must be the same.
    initial: A tensor or (possibly nested) sequence of tensors with initial
      values for the accumulators.
    reverse: (optional) True enables scan and output elems in reverse order.
    inclusive: (optional) True includes the initial accumulator values in the
      output. Length of output will be len(elem sequence) + 1. Not meaningful if
      final_only is True.
    final_only: (optional) When True, return only the final accumulated values,
      not the concatenation of accumulated values for each input.
  Returns:
    A (possibly nested) sequence of tensors with the results of applying fn
    to tensors unpacked from elems and previous accumulator values.
  """

  flat_elems = [ops.convert_to_tensor(x) for x in nest.flatten(elems)]
  num_elems = array_ops.shape(flat_elems[0])[0]
  pack_elems = lambda x: nest.pack_sequence_as(structure=elems, flat_sequence=x)
  flat_initial = [ops.convert_to_tensor(x) for x in nest.flatten(initial)]
  pack = lambda x: nest.pack_sequence_as(structure=initial, flat_sequence=x)
  accum_dtypes = [x.dtype for x in flat_initial]
  num_accums = len(flat_initial)

  # Types for counter, [outputs], [accumulators] loop arguments.
  if final_only:
    loop_dtypes = [dtypes.int32, dtypes.int32] + accum_dtypes
  else:
    loop_dtypes = [dtypes.int32, dtypes.int32] + accum_dtypes + accum_dtypes

  # TODO(tombagby): Update to tfe.defun
  def cond(i, num_elems, *args):
    del args
    return i >= 0 if reverse else i < num_elems

  # The loop *args are [output tensors] + [accumulator tensors] which must
  # be paired. Each output corresponds to one accumulator.
  def body(i, num_elems, *args):
    """Loop body."""
    i.set_shape([])
    if final_only:
      accum = args
    else:
      out, accum = args[:num_accums], args[num_accums:]
    slices = [array_ops.gather(e, i) for e in flat_elems]
    accum = fn(pack(accum), pack_elems(slices))
    flat_accum = nest.flatten(accum)
    if final_only:
      new_out = []
    else:
      update_i = i + 1 if inclusive and not reverse else i
      new_out = [
          inplace_ops.alias_inplace_update(x, update_i, y)
          for x, y in zip(out, flat_accum)
      ]
    i = i - 1 if reverse else i + 1
    return [i, num_elems] + new_out + flat_accum

  init_i = (
      array_ops.shape(flat_elems[0])[0] -
      1 if reverse else constant_op.constant(0, dtype=dtypes.int32))
  outputs = []
  if not final_only:
    num_outputs = array_ops.shape(flat_elems[0])[0] + (1 if inclusive else 0)
    for initial_accum in flat_initial:
      out_shape = array_ops.concat(
          [[num_outputs], array_ops.shape(initial_accum)], 0)
      out = inplace_ops.empty(out_shape, dtype=initial_accum.dtype, init=True)
      if inclusive:
        out = inplace_ops.alias_inplace_add(out, init_i + (1 if reverse else 0),
                                            initial_accum)
      outputs.append(out)
  loop_in = [init_i, num_elems] + outputs + flat_initial
  hostmem = [
      i for i, x in enumerate(loop_in)
      if x.dtype.base_dtype in (dtypes.int32, dtypes.int64)
  ]

  # TODO(tombagby): Update to while_v2.
  cond = function.Defun(*loop_dtypes)(cond)
  body = function.Defun(*loop_dtypes)(body)
  loop_results = While(loop_in, cond, body, hostmem=hostmem)
  out = loop_results[2:num_accums + 2]
  return pack(out)

def _LoopBodyCaptureWrapper(func):
  """Returns a wrapper for `func` that handles loop-carried captured inputs."""

  @function.Defun(
      *func.declared_input_types, func_name="%s_Wrapper" % func.name)
  def Wrapper(*args):
    """A wrapper that handles loop-carried captured inputs."""
    result = func(*args)
    extra_args = tuple(function.get_extra_args())
    # Nullary functions return an Operation. Normal functions can't do this
    # because their return values are converted to Tensors.
    if isinstance(result, ops.Operation):
      return extra_args
    # Unary functions return a single Tensor value.
    elif not isinstance(result, tuple):
      return (result,) + extra_args
    # N-ary functions return a tuple of Tensors.
    else:
      return result + extra_args

  return Wrapper

def While(input_, cond, body, name=None, hostmem=None):
  """output = input; While (Cond(output)) { output = Body(output) }.
  Args:
    input_: A list of `Tensor` objects. A list of input tensors whose types are
      T.
    cond: . A function takes 'input' and returns a tensor.  If the tensor is a
      scalar of non-boolean, the scalar is converted to a boolean
      according to the following rule: if the scalar is a numerical value,
        non-zero means True and zero means False; if the scalar is a string,
        non-empty means True and empty means False. If the tensor is not a
        scalar, non-emptiness means True and False otherwise.
    body: . A function takes a list of tensors and returns another list tensors.
      Both lists have the same types as specified by T.
    name: A name for the operation (optional).
    hostmem: A list of integer. If i is in the list, input[i] is a host memory
      tensor.
  Raises:
    ValueError: if `cond` has implicitly captured inputs or if `cond` and `body`
      have different signatures.
  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
    A list of output tensors whose types are T.
  """
  if cond.captured_inputs:
    raise ValueError("While op 'cond' argument must be a function "
                     "without implicitly captured inputs.")

  if cond.declared_input_types != body.declared_input_types:
    raise ValueError(
        "While op 'cond' and 'body' signatures do not match. %r vs %r" %
        (cond.declared_input_types, body.declared_input_types))

  if body.captured_inputs:
    cond_dtypes = list(
        body.declared_input_types) + [t.dtype for t in body.captured_inputs]

    @function.Defun(*cond_dtypes, func_name="%s_Wrapper" % cond.name)
    def CondWrapper(*args):
      """A wrapper that handles loop-carried captured inputs."""
      return cond(*args[:len(body.declared_input_types)])

    ret = gen_functional_ops._while(
        input_ + body.captured_inputs,
        CondWrapper,
        _LoopBodyCaptureWrapper(body),
        name=name)
    # Slice off the loop-carried captured inputs.
    ret = ret[:-len(body.captured_inputs)]
  else:
    ret = gen_functional_ops._while(input_, cond, body, name=name)
  if hostmem:
    input_attr = attr_value_pb2.AttrValue()
    input_attr.list.i.extend(hostmem)
    ret[0].op._set_attr("_input_hostmem", input_attr)  # pylint: disable=protected-access

    output_attr = attr_value_pb2.AttrValue()
    output_attr.list.i.extend(hostmem)
    ret[0].op._set_attr("_output_hostmem", output_attr)  # pylint: disable=protected-access
  return ret

def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def _get_dim(tensor, i):
  """Get value of tensor shape[i] preferring static value if available."""
  return shape_list(tensor)[i] or array_ops.shape(tensor)[i]

def ctc_unique_labels_single(labels, name=None):
  """Get unique labels and indices for batched labels for `tf.nn.ctc_loss`.
  For use with `tf.nn.ctc_loss` optional argument `unique`: This op can be
  used to preprocess labels in input pipeline to for better speed/memory use
  computing the ctc loss on TPU.
  Example:
    ctc_unique_labels([[3, 4, 4, 3]]) ->
      unique labels padded with 0: [[3, 4, 0, 0]]
      indices of original labels in unique: [0, 1, 1, 0]
  Args:
    labels: tensor of shape [batch_size, max_label_length] padded with 0.
    name: A name for this `Op`. Defaults to "ctc_unique_labels".
  Returns:
    tuple of
      - unique labels, tensor of shape `[batch_size, max_label_length]`
      - indices into unique labels, shape `[batch_size, max_label_length]`
  """

  with ops.name_scope(name, "ctc_unique_labels", [labels]):
    labels = ops.convert_to_tensor(labels, name="labels")

    def _unique(x):
      u = array_ops.unique(x)
      y = array_ops.pad(u.y, [[0, _get_dim(u.idx, 0) - _get_dim(u.y, 0)]])
      y = math_ops.cast(u.y, dtypes.int64)
      return [y, u.idx]

    result_lst = _unique(labels)
    output_a = tf.cast(result_lst[0], dtype=tf.int64)
    output_b = tf.cast(result_lst[1], dtype=tf.int32)
    return output_a, output_b

def reverse(x, axes):
  """Reverse a tensor along the specified axes.
  Arguments:
      x: Tensor to reverse.
      axes: Integer or iterable of integers.
          Axes to reverse.
  Returns:
      A tensor.
  """
  if isinstance(axes, int):
    axes = [axes]
  return array_ops.reverse(x, axes)

def ctc_label_dense_to_sparse(labels, label_lengths):
  """Converts CTC labels from dense to sparse.
  Arguments:
      labels: dense CTC labels.
      label_lengths: length of the labels.
  Returns:
      A sparse tensor representation of the labels.Francois Chollet, 4 years ago: • Update tf.contrib.keras with the latest chang…
  """
  label_shape = array_ops.shape(labels)
  num_batches_tns = array_ops.stack([label_shape[0]])
  max_num_labels_tns = array_ops.stack([label_shape[1]])

  def range_less_than(old_input, current_input):
    return array_ops.expand_dims(
        math_ops.range(array_ops.shape(old_input)[1]), 0) < array_ops.fill(
            max_num_labels_tns, current_input)

  init = math_ops.cast(
      array_ops.fill([1, label_shape[1]], 0), dtypes_module.bool)
  dense_mask = functional_ops.scan(
      range_less_than, label_lengths, initializer=init, parallel_iterations=1)
  dense_mask = dense_mask[:, 0, :]

  label_array = array_ops.reshape(
      array_ops.tile(math_ops.range(0, label_shape[1]), num_batches_tns),
      label_shape)
  label_ind = array_ops.boolean_mask(label_array, dense_mask)

  batch_array = array_ops.transpose(
      array_ops.reshape(
          array_ops.tile(math_ops.range(0, label_shape[0]), max_num_labels_tns),
          reverse(label_shape, 0)))
  batch_ind = array_ops.boolean_mask(batch_array, dense_mask)
  indices = array_ops.transpose(
      array_ops.reshape(tf.concat([batch_ind, label_ind], axis=0), [2, -1]))

  vals_sparse = array_ops.gather_nd(labels, indices)

  return sparse_tensor.SparseTensor(
      math_ops.cast(indices, dtypes_module.int64), vals_sparse,
      math_ops.cast(label_shape, dtypes_module.int64))

# y_true = tf.constant([[1, 2, 3, 5, 6]], dtype=tf.int32)  # shape(1, 3)
# y_true_len = tf.constant([3], dtype=tf.int32)  # shape(1,)
# y_pred = tf.constant([[[0, 8, 4, 9, 6]], [[0, -2, -4, 9, 5]], [[2, 3, 4, 5, 6]]], dtype=tf.float32)  # shape(3, 1, 5)
# y_pred_len = tf.constant([3], dtype=tf.int32)  # shape(1, )

# y_true = tf.constant([[1, 2, 3, 5, 6, 7, 8]], dtype=tf.int32)  # shape(1, 3)
# y_true_len = tf.constant([4], dtype=tf.int32)  # shape(1,)
# # y_true_sparse = ctc_label_dense_to_sparse(labels=y_true, label_lengths=y_true_len)  # dense_shape(1, 3)
# y_pred = tf.constant([[[0, 8, 4, 9, 6]], [[0, -2, -4, 9, 5]], [[2, 3, 4, 5, 6]], [[0, -2, -4, 9, 5]]], dtype=tf.float32)  # shape(3, 1, 5)
# y_pred_len = tf.constant([4], dtype=tf.int32)  # shape(1, )
# with tf.Session() as sess:
#     ctcloss = ctc_loss_v2(labels=tf.cast(y_true, tf.int32), logits=y_pred, logit_length=y_pred_len,
#                           label_length=[3]  )
#     print(sess.run(ctcloss))