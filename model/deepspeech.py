
import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
import os
from audio_io import utils as audio_utils

print(os.path.realpath(os.path.dirname(__file__)), "===deepspeech-path===")

class DeepSpeechConfig(object):
  """
  https://github.com/zzw922cn/Automatic_Speech_Recognition/blob/master/speechvalley/models/deepSpeech2.py
  https://github.com/yao-matrix/deepSpeech2/blob/gpu/src/deepSpeech.py
  larger kernel is applied to feature-dims and 
  smaller kernel is applied to time-step
  """
  def __init__(self, 
                char_vocab_size,
                pinyin_vocab_size=None,
                cnn_filters=[32, 32],
                cnn_kernel_sizes=[[11, 41], [11, 21]],
                cnn_strides=[[2, 2], [1, 2]],
                rnn_layers=2,
                rnn_hidden_size=[512,512],
                fc_layers=2,
                fc_hidden_size=2,
                conv_dropout_rate=0.1,
                rnn_dropout_rate=0.1,
                fc_dropout_rate=0.1,
                is_cnn_batch_norm=True,
                is_rnn_batch_norm=True,
                is_rnn_bidirectional=True,
                is_cnn_padding=True,
                time_major=False,
                output_mode="char"):

    self.char_vocab_size = char_vocab_size
    self.pinyin_vocab_size = pinyin_vocab_size
    self.output_mode = output_mode
    if output_mode == "char":
      self.vocab_size = self.char_vocab_size
      tf.logging.info("*** output_mode ***")
      tf.logging.info(output_mode)
      tf.logging.info(self.vocab_size)
    elif output_mode == "pinyin":
      tf.logging.info("*** output_mode ***")
      self.vocab_size = self.pinyin_vocab_size
      tf.logging.info(output_mode)
      tf.logging.info(self.vocab_size)
    self.cnn_filters = cnn_filters
    self.cnn_kernel_sizes = cnn_kernel_sizes
    self.cnn_strides = cnn_strides
    self.rnn_layers = rnn_layers
    self.rnn_hidden_size = rnn_hidden_size
    self.fc_layers = fc_layers
    self.fc_hidden_size = fc_hidden_size
    self.conv_dropout_rate = conv_dropout_rate
    self.rnn_dropout_rate = rnn_dropout_rate
    self.fc_dropout_rate = fc_dropout_rate
    self.is_cnn_batch_norm = is_cnn_batch_norm
    self.is_rnn_batch_norm = is_rnn_batch_norm
    self.is_rnn_bidirectional = is_rnn_bidirectional
    self.is_cnn_padding = is_cnn_padding
    self.time_major = time_major

    self.reduction_factor = 1
    for s in self.cnn_strides: 
      self.reduction_factor *= s[0]
    tf.logging.info("*** reduction_factor ***")
    tf.logging.info(self.reduction_factor)

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = DeepSpeechConfig(char_vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
      print(key, value, '===model parameters===')
    if config.__dict__['output_mode'] == 'char':
      config.__dict__['vocab_size'] = config.__dict__['char_vocab_size']
    elif config.__dict__['output_mode'] == "pinyin":
      config.__dict__['vocab_size'] = config.__dict__['pinyin_vocab_size']
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DeepSpeech(object):
  def __init__(self, 
              config,
              sequences, 
              is_training,
              input_length=None,
              sequences_mask=None):

    config = copy.deepcopy(config)
    for key in config.__dict__:
      print(key, "==config==", config.__dict__[key])

    initializer = tf.truncated_normal_initializer(stddev=0.046875, dtype=tf.float32)
    regularizer = tf.contrib.layers.l2_regularizer(0.0046875)

    if not is_training:
      config.conv_dropout_rate = 0.0
      config.rnn_dropout_rate = 0.0
      config.fc_dropout_rate = 0.0

    sequences_shape = get_shape_list(sequences, expected_rank=[3,4])
    max_length = sequences_shape[1]
    if len(sequences_shape) == 4:
      sequences = sequences
    else:
      # sequences = [batch_size, time, NUM_INPUTS] => [batch_size, time, NUM_INPUTS, 1]
      # since feature extraction will add extra dims on input:
      # [batch_size, time, n_dims]--->[batch_size, time, n_dims, 1]
      # so, no need for dimension expandims
      sequences = tf.expand_dims(sequences, axis=-1)

    tf.logging.info("*** sequences ***")
    tf.logging.info(sequences)

    with tf.variable_scope('deepspeech', reuse=tf.AUTO_REUSE):

      # Apply convolutions.
      with tf.variable_scope('conv_module'):
        [self.conv_output, 
        self.reduction_factor] = conv2d_block(sequences,
                                filters=config.cnn_filters, 
                                kernel_sizes=config.cnn_kernel_sizes, 
                                strides=config.cnn_strides,
                                dropout_rate=config.conv_dropout_rate,
                                is_training=is_training,
                                is_batch_norm=config.is_cnn_batch_norm,
                                is_padding=config.is_cnn_padding)

        conv_output_shape = get_shape_list(self.conv_output, expected_rank=[4])
        self.conv_output = tf.reshape(self.conv_output, 
          shape=[conv_output_shape[0], -1, conv_output_shape[2] * conv_output_shape[3]])

        tf.logging.info("*** conv_output ***")
        tf.logging.info(self.conv_output)

      if sequences_mask is not None:
        tf.logging.info("*** apply sequences_mask ***")
        self.conv_output *= tf.cast(sequences_mask, dtype=tf.float32)

      if input_length is not None:
        tf.logging.info("*** apply rnn reduced_length ***")
        reduced_length = audio_utils.get_reduced_length(input_length, self.reduction_factor)
        # reduced_total_length = audio_utils.get_reduced_length(input_length, self.reduction_factor)
        # sequences_mask = tf.sequence_mask(reduced_length, )
      else:
        tf.logging.info("*** apply rnn padded_length ***")
        reduced_length = None

      with tf.variable_scope('rnn_module'):
        # [batch, seq_len//reduction_factor, dims]
        rnn_cell = tf.nn.rnn_cell.GRUCell
        self.rnn_output = rnn_block(self.conv_output, 
                            rnn_cell=rnn_cell, 
                            rnn_hidden_size=config.rnn_hidden_size, 
                            rnn_layers=config.rnn_layers,
                            is_batch_norm=config.is_rnn_batch_norm, 
                            is_bidirectional=config.is_rnn_bidirectional, 
                            is_training=is_training,
                            time_major=config.time_major,
                            sequence_length=reduced_length)

        tf.logging.info("*** rnn_output ***")
        tf.logging.info(self.rnn_output)

      with tf.variable_scope('fc_module'):
        self.fc_output = fc_block(self.rnn_output,
                  fc_layers=config.fc_layers, 
                  hidden_size=config.fc_hidden_size, 
                  dropout_rate=config.fc_dropout_rate,
                  is_training=is_training)

        tf.logging.info("*** fc_output ***")
        tf.logging.info(self.fc_output)
        
      with tf.variable_scope('cls/predictions'):
        self.logits = tf.layers.dense(self.fc_output, config.vocab_size, 
                                kernel_initializer=initializer)

        tf.logging.info("*** logits ***")
        tf.logging.info(self.logits)

  def get_conv_output(self):
    return self.conv_output

  def get_conv_reduction_factor(self):
    return self.reduction_factor

  def get_rnn_output(self):
    return self.rnn_output

  def get_fc_output(self):
    return self.fc_output

  def get_logits(self):
    return self.logits

def batch_norm(inputs, is_training, 
              batch_norm_decay=0.997, 
              batch_norm_eps=1e-5):
  return tf.layers.batch_normalization(
          inputs=inputs, 
          momentum=batch_norm_decay, 
          epsilon=batch_norm_eps,
          fused=True, 
          training=is_training)

def conv2d_bn_layer(inputs, 
                  filters, 
                  kernel_size, 
                  strides,
                  dropout_rate,
                  is_batch_norm,
                  is_training=False,
                  is_padding=True
                  ):

  if is_padding:
    paddings = [k_size//2 for k_size in kernel_size]
    inputs = tf.pad(
        inputs,
        [[0, 0], [paddings[0], paddings[0]], [paddings[1], paddings[1]], [0, 0]])
    tf.logging.info("** apply cnn padding **")
  inputs = tf.layers.conv2d(
                  inputs=inputs, 
                  filters=filters, 
                  kernel_size=kernel_size, 
                  strides=strides,
                  padding="valid", 
                  use_bias=False, 
                  activation=None,
                  kernel_initializer=tf.glorot_normal_initializer())
  if is_batch_norm:
    inputs = batch_norm(inputs, is_training)
  inputs = tf.nn.relu6(inputs)
  inputs = tf.nn.dropout(inputs, keep_prob=1-dropout_rate)
  return inputs

def conv2d_block(inputs, 
              filters=[32, 32, 96], 
              kernel_sizes=[[11, 41], [11, 21], [11, 21]], 
              strides=[[2, 2], [1, 2], [1, 2]],
              dropout_rate=0.1,
              is_batch_norm=True,
              is_training=False,
              is_padding=True):

  assert len(kernel_sizes) == len(strides) == len(filters)
  pre_output = inputs
  for layer_idx in range(len(filters)):
    with tf.variable_scope("layer_%d" % layer_idx):
      pre_output = conv2d_bn_layer(pre_output, 
                    filters=filters[layer_idx], 
                    kernel_size=kernel_sizes[layer_idx], 
                    strides=strides[layer_idx], 
                    dropout_rate=dropout_rate,
                    is_batch_norm=is_batch_norm,
                    is_training=is_training,
                    is_padding=is_padding
                    )

  reduction_factor = 1
  for s in strides: 
    reduction_factor *= s[0]

  return pre_output, reduction_factor

def rnn_layer(inputs, rnn_cell, 
              rnn_hidden_size, 
              is_batch_norm,
              is_bidirectional,
              sequence_length=None, 
              is_training=False,
              time_major=False):
  
  fw_cell = rnn_cell(num_units=rnn_hidden_size,
                   name="forward")
  bw_cell = rnn_cell(num_units=rnn_hidden_size,
                   name="backward")

  if is_bidirectional:
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs, dtype=tf.float32,
    swap_memory=False,
    sequence_length=sequence_length)
    rnn_outputs = tf.concat(outputs, -1)
  else:
    rnn_outputs = tf.nn.dynamic_rnn(
    fw_cell, inputs, dtype=tf.float32, swap_memory=False,
    sequence_length=sequence_length)

  if is_batch_norm:
    rnn_outputs = sequecnce_batch_norm(rnn_outputs, time_major=time_major)
  
  return rnn_outputs

def rnn_block(inputs, 
              rnn_cell, 
              rnn_hidden_size, 
              rnn_layers, 
              is_batch_norm,
              is_bidirectional, 
              sequence_length=None,
              is_training=False,
              time_major=False):

  pre_output = inputs
  for layer_idx in range(rnn_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      pre_output = rnn_layer(pre_output, 
                rnn_cell=rnn_cell, 
                rnn_hidden_size=rnn_hidden_size[layer_idx], 
                is_batch_norm=is_batch_norm,
                is_bidirectional=is_bidirectional, 
                is_training=is_training,
                time_major=time_major,
                sequence_length=sequence_length)
  return pre_output

def sequecnce_batch_norm(inputs, 
                    time_major=False,
                    variance_epsilon=1e-5):
  input_shape = get_shape_list(inputs, expected_rank=[3])
  beta = tf.get_variable(shape=[input_shape[-1]],
                        name='beta', initializer=tf.zeros_initializer(),
                        regularizer=None, constraint=None, trainable=True)
  gamma = tf.get_variable(shape=[input_shape[-1]],
                       name='gamma', initializer=tf.ones_initializer(),
                       regularizer=None, constraint=None, trainable=True)
  mean, variance = tf.nn.moments(inputs, axes=[0, 1], keep_dims=False)
  if time_major:
    total_padded_frames = tf.cast(input_shape[0], mean.dtype)
    batch_size = tf.cast(input_shape[1], mean.dtype)
  else:
    total_padded_frames = tf.cast(input_shape[1], mean.dtype)
    batch_size = tf.cast(input_shape[0], mean.dtype)
  total_unpadded_frames_batch = tf.count_nonzero(
            inputs, axis=[0, 1], keepdims=False,
            dtype=mean.dtype
        )
  mean = (mean * total_padded_frames * batch_size) / total_unpadded_frames_batch
  variance = (variance * total_padded_frames * batch_size) / total_unpadded_frames_batch
  return tf.nn.batch_normalization(
      inputs, mean=mean, variance=variance,
      offset=beta, scale=gamma,
      variance_epsilon=1e-8
  )

def fc_layer(inputs, hidden_size, 
            dropout_rate, 
            is_training=False):
  fc_intermediate_output = tf.layers.dense(inputs, 
                      units=hidden_size)

  ffc_output = tf.nn.relu6(fc_intermediate_output)
  ffc_output = tf.nn.dropout(ffc_output, keep_prob=1-dropout_rate)
  return ffc_output

def fc_block(inputs, 
            fc_layers, 
            hidden_size, 
            dropout_rate,
            is_training=False):
  pre_output = inputs
  for layer_idx in range(fc_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      pre_output = fc_layer(pre_output, 
              hidden_size=hidden_size,
              dropout_rate=dropout_rate,
              is_training=is_training)
  return pre_output

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)

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
