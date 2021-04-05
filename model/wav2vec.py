
import tensorflow as tf
import numpy as np
from vqvae import soft_em
from model import transformer_relative_position

"""
https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/speech-to-text/wav2vec.ipynb
""" 

class Wav2VecConfig(object):
  def __init__(self,
        vocab_size,
        subsampling_filters=[144, 144],
        subsampling_kernel_sizes=[[3, 3], [3, 3]],
        subsampling_strides=[[2, 1], [2, 1]],
        subsampling_dropout=0.1,
        proj_dropout=0.1,
        ffm_expansion_factor=4,
        ffm_dropout=0.1,
        ffm_hidden_size=256,
        ffm_fc_factor=0.5,

        mha_hidden_size=256,
        mha_num_attention_heads=4,
        mha_max_relative_position=64,
        mha_num_buckets=32,
        mha_bidirectional=True,
        mha_initializer_range=0.02,
        mha_relative_position_type="relative_t5",
        mha_relative_position_embedding_type="sinusoidal_trainable",
        mha_num_hidden_layers=4,

        cnn_filters=256,
        cnn_kernel_sizes=9,
        cnn_strides=1,
        cnn_depth_multiplier=1):

    self.vocab_size = vocab_size
    self.subsampling_filters = subsampling_filters
    self.subsampling_kernel_sizes = subsampling_kernel_sizes
    self.subsampling_strides = subsampling_strides
    self.subsampling_dropout = subsampling_dropout

    self.proj_dropout = proj_dropout

    self.ffm_expansion_factor = ffm_expansion_factor
    self.ffm_dropout = ffm_dropout
    self.ffm_hidden_size = ffm_hidden_size
    self.ffm_fc_factor = ffm_fc_factor

    self.mha_hidden_size = mha_hidden_size
    self.mha_num_attention_heads = mha_num_attention_heads
    self.mha_max_relative_position = mha_max_relative_position
    self.mha_num_buckets = mha_num_buckets
    self.mha_initializer_range = mha_initializer_range
    self.mha_bidirectional = mha_bidirectional
    self.mha_relative_position_type = mha_relative_position_type
    self.mha_relative_position_embedding_type = mha_relative_position_embedding_type
    self.mha_num_hidden_layers = mha_num_hidden_layers

    self.cnn_filters = cnn_filters
    self.cnn_kernel_sizes = cnn_kernel_sizes
    self.cnn_strides = cnn_strides
    self.cnn_depth_multiplier = cnn_depth_multiplier

    self.conv_dropout_rate = conv_dropout_rate
    self.is_cnn_batch_norm = is_cnn_batch_norm
    self.is_cnn_padding = is_cnn_padding

class Wav2Vec(object):
  def __init__(self, 
              config,
              sequences, 
              is_training=False,
              sequences_mask=None):

    if not is_training:
      config.transformer_hidden_dropout_prob = 0.0
      config.transformer_attention_probs_dropout_prob = 0.0
      config.subsampling_dropout = 0.0
      config.proj_dropout = 0.0
      config.ffm_dropout = 0.0
      config.conv_dropout_rate = 0.0

    initializer = tf.truncated_normal_initializer(stddev=0.046875, dtype=tf.float32)

    with tf.variable_scope('wav2vec'):
      # since feature extraction will add extra dims on input:
      # [batch_size, time, n_dims]--->[batch_size, time, n_dims, 1]
      # so, no need for dimension expandims
      sequences_shape = get_shape_list(sequences, expected_rank=[3,4])
      if len(sequences_shape) == 4:
        # perform raw audio input
        with tf.variable_scope('conv_downsampling'):
          [self.conv_subsampling, 
          self.reduction_factor] = conv2d_block(sequences,
                                  filters=config.subsampling_filters, 
                                  kernel_sizes=config.subsampling_kernel_sizes, 
                                  strides=config.subsampling_strides,
                                  dropout_rate=config.subsampling_dropout,
                                  is_training=is_training,
                                  is_batch_norm=False,
                                  is_padding=True)

        conv_subsampling_shape = get_shape_list(self.conv_subsampling, expected_rank=[4])
        self.conv_subsampling = tf.reshape(self.conv_subsampling, 
                  shape=[conv_subsampling_shape[0], -1, 
                  conv_subsampling_shape[2] * conv_subsampling_shape[3]])

      elif len(sequences_shape) == 3:
        with tf.variable_scope('conv_downsampling'):
          [self.conv_subsampling, 
          self.reduction_factor] = conv1d_block(sequences,
                                    filters=config.subsampling_filters, 
                                    kernel_sizes=config.subsampling_kernel_sizes, 
                                    strides=config.subsampling_strides,
                                    dropout_rate=config.subsampling_dropout,
                                    is_training=is_training,
                                    is_batch_norm=False,
                                    is_padding=True)

      conv_subsampling_shape = get_shape_list(self.conv_subsampling, expected_rank=[3])
      assert len(conv_subsampling_shape) == 3

      with tf.variable_scope('linear_proj'):
        self.linear_proj = tf.layers.dense(self.conv_subsampling, 
                    units=conv_subsampling_shape[-1]
                  )
        self.linear_proj = tf.nn.dropout(self.linear_proj, 
                            keep_prob=1-config.proj_dropout)

      with tf.variable_scope('encoder'):
        mha_attention_head_size = config.mha_hidden_size // config.mha_num_attention_heads
        [self.relative_position_embeddings, 
          self.relative_position_table] = transformer_relative_position._generate_relative_positions_embeddings(
                      input_shape[1], 
                      depth=mha_attention_head_size,
                      max_relative_position=config.mha_max_relative_position, 
                      name="relative_positions_bias",
                      num_buckets=config.mha_num_buckets,
                      initializer_range=config.mha_initializer_range,
                      cache=False,
                      bidirectional=config.mha_bidirectional,
                      relative_position_type=config.mha_relative_position_type,
                      relative_position_embedding_type=config.mha_relative_position_embedding_type)

        pre_output = self.linear_proj

        self.conformer_block = conformer(pre_output,
            ffm_hidden_size=config.ffm_hidden_size,
            ffm_dropout_rate=config.ffm_dropout_rate,
            ffm_fc_factor=config.ffm_fc_factor,
            ffm_expansion_factor=config.ffm_expansion_factor,
            mha_relative_position_embeddings=self.relative_position_embeddings,
            mha_num_attention_heads=config.mha_num_attention_heads,
            mha_attention_head_size=config.mha_attention_head_size,
            mha_attention_probs_dropout_prob=config.mha_attention_probs_dropout_prob,
            mha_initializer_range=config.mha_initializer_range,
            mha_use_relative_position=config.mha_use_relative_position,
            mha_num_hidden_layers=config.mha_num_hidden_layers,
            conv_filters=config.conv_filters,
            conv_kernel_sizes=config.conv_kernel_sizes,
            conv_strides=config.conv_strides,
            cnn_depth_multiplier=config.cnn_depth_multiplier,
            conv_dropout_prob=config.conv_dropout_prob,
            relative_position_type=config.mha_relative_position_type,
            is_training=is_training)

      with tf.variable_scope('cls/predictions'):
        self.logits = tf.layers.dense(self.conformer_block[-1], config.vocab_size, 
                                kernel_initializer=initializer)

  def get_conv_downsampling_output(self):
    return self.conv_subsampling

  def get_sequence_output(self):
    return self.conformer_block

  def get_conv_reduction_factor(self):
    return self.reduction_factor

  def get_logits(self):
    return self.logits

def conformer(inputs,
            ffm_hidden_size,
            ffm_dropout_rate,
            ffm_fc_factor,
            ffm_expansion_factor,
            mha_relative_position_embeddings,
            mha_num_attention_heads,
            mha_attention_head_size,
            mha_attention_probs_dropout_prob=0.1,
            mha_initializer_range=0.02,
            mha_use_relative_position=True,
            mha_num_hidden_layers=12,
            conv_filters=256,
            conv_kernel_sizes=31,
            conv_strides=1,
            conv_dropout_prob=0.1,
            cnn_depth_multiplier=1,
            relative_position_type="relative_normal",
            is_training=False):

  input_shape = get_shape_list(inputs, expected_rank=[3])
  batch_size = input_shape[0]
  seq_length = input_shape[1]

  pre_output = inputs
  conformer_block = []

  for layer_idx in range(mha_num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      with tf.variable_scope("pre_input"):
        outputs = residual_ffm_block(pre_output, 
                      hidden_size=ffm_hidden_size, 
                      dropout_rate=ffm_dropout_rate,
                      fc_factor=ffm_fc_factor,
                      expansion_factor=ffm_expansion_factor,
                      is_training=is_training)

      with tf.variable_scope("attention"):
        with tf.variable_scope("self"):
          [attention_head, attention_probs] = transformer_relative_position.attention_layer(
                    from_tensor=outputs,
                    to_tensor=outputs,
                    attention_mask=None,
                    num_attention_heads=mha_num_attention_heads,
                    size_per_head=mha_attention_head_size,
                    attention_probs_dropout_prob=mha_attention_probs_dropout_prob,
                    initializer_range=mha_initializer_range,
                    do_return_2d_tensor=False,
                    batch_size=batch_size,
                    from_seq_length=seq_length,
                    to_seq_length=seq_length,
                    use_relative_position=mha_use_relative_position,
                    dropout_name=tf.get_variable_scope().name,
                    relative_position_type=relative_position_type,
                    relative_position_embeddings=mha_relative_position_embeddings)
        
        attention_output = layer_norm(attention_head + outputs)

        with tf.variable_scope("conformer_conv"):
          conv_output = conformer_conv(attention_output, 
                filters=conv_filters, 
                kernel_size=conv_kernel_sizes, 
                strides=conv_strides,
                depth_multiplier=depth_multiplier,
                dropout_rate=conv_dropout_prob)

        conv_attention_output = layer_norm(conv_output + attention_output)

        with tf.variable_scope("post_output"):
          outputs = residual_ffm_block(conv_attention_output, 
                      hidden_size=ffm_hidden_size, 
                      dropout_rate=ffm_dropout_rate,
                      fc_factor=ffm_fc_factor,
                      expansion_factor=ffm_expansion_factor,
                      is_training=is_training)
        conformer_block.append(outputs)
        pre_output = outputs

  return conformer_block

def glu(inputs, axis=-1):
  a, b = tf.split(inputs, 2, axis=-1)
  b = tf.nn.sigmoid(b)
  return tf.multiply(a, b)

def conformer_conv(inputs, 
            filters, 
            kernel_size, 
            strides=1,
            depth_multiplier=1,
            dropout_rate=0.1):
  # [batch, seq_len, dims]
  input_shape = get_shape_list(inputs, expected_rank=[3])
  # [batch, seq_len, 1, dims]
  outputs = tf.epxand_dims(inputs, 2)

  # [batch, seq_len, 1, filters*2]
  outputs = tf.layers.conv2d(
                  inputs=outputs, 
                  filters=filters*2, 
                  kernel_size=1, 
                  strides=1,
                  padding="valid", 
                  use_bias=True, 
                  activation=None,
                  kernel_initializer=tf.glorot_normal_initializer())

  # [batch, seq_len, 1, filters]
  outputs = glu(outputs, axis=-1)

  depthwise_filter = tf.get_variable("depthwise_filter",
                    (kernel_size, 1, filters, depth_multiplier),
                    dtype=tf.float32,
                    initializer=tf.glorot_normal_initializer())

  outputs = tf.nn.depthwise_conv2d(
    outputs, depthwise_filter, (1,1,1,1), 
    "SAME"
  )

  outputs = batch_norm(outputs, is_training=is_training)
  outputs = gelu(outputs)

  # [batch, seq_len, 1, dims]
  outputs = tf.layers.conv2d(
                  inputs=outputs, 
                  filters=input_shape[-1], 
                  kernel_size=1, 
                  strides=1,
                  padding="valid", 
                  use_bias=True, 
                  activation=None,
                  kernel_initializer=tf.glorot_normal_initializer())

  # [batch, seq_len, dims]
  outputs = tf.squeeze(outputs, axis=2)
  outputs = tf.nn.dropout(outputs, keep_prob=1-dropout_rate)
  return outputs

def residual_ffm_block(inputs, hidden_size, 
                  dropout_rate,
                  fc_factor,
                  expansion_factor,
                  is_training=False):

  if is_training:
    dropout_rate = 0.0

  input_shape = get_shape_list(inputs, expected_rank=[3])
  outputs = tf.layers.dense(outputs, 
                    units=expansion_factor*hidden_size
          )
  outputs = gelu(outputs)
  outputs = tf.nn.dropout(outputs, keep_prob=1-dropout_rate)
  outputs = tf.layers.dense(outputs, 
                    units=input_shape[-1])
  outputs = tf.nn.dropout(outputs, keep_prob=1-dropout_rate)
  outputs = inputs + fc_factor * outputs
  outputs = layer_norm(inputs)
  return outputs

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
                  layer_id,
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
              filters=[144, 144], 
              kernel_sizes=[[3, 3], [3, 3]], 
              strides=[[2, 1], [2, 1]],
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
                    kernel_size=kernel_size[layer_idx], 
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

def conv1d_bn_layer(inputs, 
                  filters, 
                  kernel_size, 
                  strides, 
                  layer_id,
                  dropout_rate,
                  is_batch_norm,
                  is_training=False,
                  is_padding=False
                  ):
  if is_padding:
    ka = kernel_size // 2
    kb = ka - 1 if kernel_size % 2 == 0 else ka
    pad = tf.zeros([tf.shape(x)[0], kb + ka, filters])
    inputs = tf.concat([pad, inputs], 1)
  inputs = tf.layers.conv1d(
                  inputs=inputs, 
                  filters=filters, 
                  kernel_size=kernel_size, 
                  strides=strides,
                  padding='valid', 
                  use_bias=False, 
                  activation=None,
                  kernel_initializer=tf.glorot_normal_initializer())

  inputs = layer_norm(inputs)
  inputs = gelu(inputs)
  inputs = tf.nn.dropout(inputs, keep_prob=1-dropout_rate)
  return inputs

def conv1d_block(inputs, 
              filters=[512,512,512,512,512,512,512], 
              kernel_sizes=[10,3,3,3,3,2,2], 
              strides=[5,2,2,2,2,2,2],
              conv_dropout_rate=0.1,
              is_batch_norm=True,
              is_training=False,
              is_padding=False):

  assert len(kernel_sizes) == len(strides) == len(filters)
  pre_output = inputs
  for layer_idx in range(len(filters)):
    with tf.variable_scope("layer_%d" % layer_idx):
      pre_output = conv1d_bn_layer(pre_output, 
                    filters=filters[layer_idx], 
                    kernel_size=kernel_size[layer_idx], 
                    strides=strides[layer_idx], 
                    dropout_rate=conv_dropout_rate,
                    is_training=is_training,
                    is_padding=is_padding,
                    is_batch_norm=is_batch_norm)

  reduction_factor = 1
  for s in strides: 
    reduction_factor *= s[0]

  return pre_output, reduction_factor

def gelu(input_tensor):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

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
