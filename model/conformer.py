
import tensorflow as tf
import numpy as np
from vqvae import soft_em
from model import transformer_relative_position
from audio_io import utils as audio_utils
import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
import os

"""
https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/speech-to-text/wav2vec.ipynb
""" 

class ConformerConfig(object):
  def __init__(self,
        char_vocab_size,
        pinyin_vocab_size=None,
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
        mha_attention_probs_dropout_prob=0.1,
        mha_hidden_dropout_prob=0.1,
        mha_use_relative_position=True,

        cnn_kernel_sizes=32,
        cnn_strides=1,
        cnn_depth_multiplier=1,
        cnn_dropout_prob=0.1,
        is_cnn_batch_norm=True,
        is_cnn_padding=True,

        fc_layers=1,
        fc_hidden_size=1,
        fc_dropout_rate=0.1,

        bottleneck_size=384,
        bottleneck_dims=256,

        vqvae_beta=0.25,
        vqvae_gamma=0.1,

        time_major=False,
        output_mode="char"):

    self.char_vocab_size = char_vocab_size
    self.pinyin_vocab_size = pinyin_vocab_size
    self.output_mode = output_mode
    if output_mode == "char":
      self.vocab_size = self.char_vocab_size
      tf.logging.info(output_mode)
      tf.logging.info(self.vocab_size)
    elif output_mode == "pinyin":
      self.vocab_size = self.pinyin_vocab_size
      tf.logging.info(output_mode)
      tf.logging.info(self.vocab_size)
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
    self.mha_attention_probs_dropout_prob = mha_attention_probs_dropout_prob
    self.mha_hidden_dropout_prob = mha_hidden_dropout_prob
    self.mha_use_relative_position = mha_use_relative_position

    self.cnn_kernel_sizes = cnn_kernel_sizes
    self.cnn_strides = cnn_strides
    self.cnn_depth_multiplier = cnn_depth_multiplier

    self.cnn_dropout_prob = cnn_dropout_prob
    self.is_cnn_batch_norm = is_cnn_batch_norm
    self.is_cnn_padding = is_cnn_padding

    self.fc_layers = fc_layers
    self.fc_hidden_size = fc_hidden_size
    self.fc_dropout_rate = fc_dropout_rate

    self.vqvae_beta = vqvae_beta
    self.vqvae_gamma = vqvae_gamma

    self.bottleneck_size = bottleneck_size
    self.bottleneck_dims = bottleneck_dims

    self.time_major = time_major

    self.reduction_factor = 1
    for s in self.subsampling_strides: 
      self.reduction_factor *= s[0]
    tf.logging.info("*** reduction_factor ***")
    tf.logging.info(self.reduction_factor)

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = ConformerConfig(char_vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
      print(key, value, '===model parameters===')
    if config.__dict__['output_mode'] == 'char':
      config.__dict__['vocab_size'] = config.__dict__['char_vocab_size']
      tf.logging.info("** output_mode is char **")
    elif config.__dict__['output_mode'] == "pinyin":
      config.__dict__['vocab_size'] = config.__dict__['pinyin_vocab_size']
      tf.logging.info("** output_mode is pinyin **")
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


class Conformer(object):
  def __init__(self, 
              config,
              sequences,
              input_length,
              is_training=False,
              is_pretraining=False,
              time_feature_mask=None,
              freq_feature_mask=None,
              target_feature_mode='linear',
              is_global_bn=False):

    config = copy.deepcopy(config)
    self.config = copy.deepcopy(config)
    for key in config.__dict__:
      print(key, "==config==", config.__dict__[key])

    if not is_training:
      config.mha_hidden_dropout_prob = 0.0
      config.mha_attention_probs_dropout_prob = 0.0
      config.subsampling_dropout = 0.0
      config.proj_dropout = 0.0
      config.ffm_dropout = 0.0
      config.cnn_dropout_prob = 0.0
      config.fc_dropout_rate = 0.0

    initializer = tf.truncated_normal_initializer(stddev=0.046875, dtype=tf.float32)

    with tf.variable_scope('conformer', reuse=tf.AUTO_REUSE):
      # since feature extraction will add extra dims on input:
      # [batch_size, time, n_dims]--->[batch_size, time, n_dims, 1]
      # so, no need for dimension expandims
      sequences_shape = get_shape_list(sequences, expected_rank=[3,4])
      if len(sequences_shape) == 4:
        tf.logging.info("*** specturm input ***")
        # perform raw audio input
        with tf.variable_scope('conv_downsampling'):
          [self.conv_subsampling, 
          self.reduction_factor] = conv2d_block(sequences,
                                  filters=config.subsampling_filters, 
                                  kernel_sizes=config.subsampling_kernel_sizes, 
                                  strides=config.subsampling_strides,
                                  dropout_rate=config.subsampling_dropout,
                                  is_training=is_training,
                                  is_batch_norm=config.is_cnn_batch_norm,
                                  is_padding=config.is_cnn_padding,
                                  is_global_bn=is_global_bn)

        conv_subsampling_shape = get_shape_list(self.conv_subsampling, expected_rank=[4])
        self.conv_subsampling = tf.reshape(self.conv_subsampling, 
                  shape=[conv_subsampling_shape[0], -1, 
                  conv_subsampling_shape[2] * conv_subsampling_shape[3]])

        tf.logging.info("*** conv down-sampling ***")
        tf.logging.info(self.conv_subsampling)

      elif len(sequences_shape) == 3:
        tf.logging.info("*** audio signal input ***")
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

        tf.logging.info("*** conv down-sampling ***")
        tf.logging.info(self.conv_subsampling)

      conv_subsampling_shape = get_shape_list(self.conv_subsampling, expected_rank=[3])
      assert len(conv_subsampling_shape) == 3

      self.unmasked_conv_subsampling = tf.identity(self.conv_subsampling)

      if is_pretraining:

        if target_feature_mode == 'soft-em':
          # apply softem
          tf.logging.info("*** apply soft em ***")
          with tf.variable_scope('discrete_bottleneck'):

            [self.code_book,
            self.code_discrete, 
            self.code_dense, 
            self.code_loss_dict] = soft_em.discrete_bottleneck(
                      self.conv_subsampling,
                      config.bottleneck_size,
                      config.bottleneck_dims,
                      beta=config.vqvae_beta,
                      gamma=config.vqvae_gamma,
                      is_training=is_training)
        elif target_feature_mode == 'linear':
          tf.logging.info("*** apply linear proj ***")
          with tf.variable_scope('pretrain_linear_proj'):
            self.code_dense = tf.layers.dense(
                        self.conv_subsampling, 
                        units=config.ffm_hidden_size
                      )
            self.code_dense = layer_norm(self.code_dense)

            self.code_loss_dict = {}
            
            self.code_discrete = tf.identity(self.code_dense)
            self.code_book = tf.identity(self.code_dense)

      with tf.variable_scope('linear_proj'):
        if is_pretraining:
          tf.logging.info("*** apply mask before linear_proj ***")
          if time_feature_mask is not None:
            tf.logging.info("*** apply time mask before linear_proj ***")
            tf.logging.info(time_feature_mask)
            time_feature_mask = tf.cast(time_feature_mask, dtype=tf.float32)
            # [B, T, 1]
            self.conv_subsampling *= tf.expand_dims(time_feature_mask, axis=-1)
          if freq_feature_mask is not None:
            tf.logging.info("***@* apply freq mask before linear_proj ***")
            tf.logging.info(freq_feature_mask)
            freq_feature_mask = tf.cast(freq_feature_mask, dtype=tf.float32)
            self.conv_subsampling *= freq_feature_mask

        self.linear_proj = tf.layers.dense(
                      self.conv_subsampling, 
                      units=config.ffm_hidden_size
                    )

        self.linear_proj = layer_norm(self.linear_proj)

        self.linear_proj = tf.nn.dropout(self.linear_proj, 
                            keep_prob=1-config.proj_dropout)

        tf.logging.info("**** linear_proj ****")
        tf.logging.info(self.linear_proj)

      if input_length is not None:
        tf.logging.info("*** generate attention mask ***")
        reduced_length = audio_utils.get_reduced_length(input_length, self.reduction_factor)
        
        tf.logging.info("*** reduced_length ***")
        tf.logging.info(reduced_length)

        sequence_mask = tf.sequence_mask(reduced_length, conv_subsampling_shape[1])
        sequence_mask = tf.cast(sequence_mask, dtype=tf.float32)
        tf.logging.info("*** sequence_mask ***")
        tf.logging.info(sequence_mask)
        if time_feature_mask is not None:
          sequence_mask *= time_feature_mask
        self.attention_mask = transformer_relative_position.create_attention_mask_from_input_mask(
                                sequence_mask,
                                sequence_mask)
      else:
        self.attention_mask = None
      tf.logging.info("*** attention_mask ***")
      tf.logging.info(self.attention_mask)

      with tf.variable_scope('encoder'):
        tf.logging.info("*** mha encoder ***")
        mha_attention_head_size = config.mha_hidden_size // config.mha_num_attention_heads
        tf.logging.info("*** mha_attention_head_size ***")
        tf.logging.info(mha_attention_head_size)

        [self.relative_position_embeddings, 
          self.relative_position_table] = transformer_relative_position._generate_relative_positions_embeddings(
                      conv_subsampling_shape[1], 
                      depth=config.mha_num_attention_heads,
                      max_relative_position=config.mha_max_relative_position, 
                      name="relative_positions_bias",
                      num_buckets=config.mha_num_buckets,
                      initializer_range=config.mha_initializer_range,
                      cache=False,
                      bidirectional=config.mha_bidirectional,
                      relative_position_type=config.mha_relative_position_type,
                      relative_position_embedding_type=config.mha_relative_position_embedding_type)
        tf.logging.info("****** relative_position_embeddings ***")
        tf.logging.info(self.relative_position_embeddings)
        pre_output = self.linear_proj

        self.conformer_block = conformer(pre_output,
            ffm_hidden_size=config.ffm_hidden_size,
            ffm_dropout_rate=config.ffm_dropout,
            ffm_fc_factor=config.ffm_fc_factor,
            ffm_expansion_factor=config.ffm_expansion_factor,
            mha_relative_position_embeddings=self.relative_position_embeddings,
            mha_num_attention_heads=config.mha_num_attention_heads,
            mha_attention_head_size=mha_attention_head_size,
            mha_attention_probs_dropout_prob=config.mha_attention_probs_dropout_prob,
            mha_hidden_dropout_prob=config.mha_hidden_dropout_prob,
            mha_initializer_range=config.mha_initializer_range,
            mha_use_relative_position=config.mha_use_relative_position,
            mha_num_hidden_layers=config.mha_num_hidden_layers,
            mha_attention_mask=self.attention_mask,
            conv_strides=config.cnn_strides,
            conv_depth_multiplier=config.cnn_depth_multiplier,
            conv_dropout_prob=config.cnn_dropout_prob,
            relative_position_type=config.mha_relative_position_type,
            is_training=is_training,
            is_global_bn=is_global_bn)

      tf.logging.info("*** conformer_block ***")
      tf.logging.info(self.conformer_block)

      if not is_pretraining:
        with tf.variable_scope('fc_module'):
          self.fc_output = fc_block(self.conformer_block[-1],
                    fc_layers=config.fc_layers, 
                    hidden_size=config.fc_hidden_size, 
                    dropout_rate=config.fc_dropout_rate,
                    is_training=is_training)

        tf.logging.info("**** fc_output ****")
        tf.logging.info(self.fc_output)

        with tf.variable_scope('cls/predictions'):
          self.logits = tf.layers.dense(self.fc_output, 
                                  config.vocab_size, 
                                  kernel_initializer=initializer)

          tf.logging.info("*** logits ***")
          tf.logging.info(self.logits)

  def get_unmasked_linear_proj(self):
    with tf.variable_scope('conformer', reuse=tf.AUTO_REUSE):
      with tf.variable_scope('linear_proj'):
        linear_proj = tf.layers.dense(
                      self.unmasked_conv_subsampling, 
                      units=self.config.ffm_hidden_size
                    )
        linear_proj = layer_norm(linear_proj)
        return linear_proj

  def get_linear_proj_encoder(self, 
                            linear_proj, 
                            input_length,
                            is_training,
                            time_feature_mask):
    conv_subsampling_shape = get_shape_list(linear_proj, expected_rank=[3])
    assert len(conv_subsampling_shape) == 3
    with tf.variable_scope('conformer', reuse=tf.AUTO_REUSE):
      if input_length is not None:
        tf.logging.info("*** generate attention mask ***")
        reduced_length = audio_utils.get_reduced_length(input_length, self.reduction_factor)
        sequence_mask = tf.sequence_mask(reduced_length, conv_subsampling_shape[1])
        tf.logging.info("*** sequence_mask ***")
        tf.logging.info(sequence_mask)
        if time_feature_mask is not None:
          sequence_mask *= time_feature_mask
        attention_mask = transformer_relative_position.create_attention_mask_from_input_mask(
                                sequence_mask,
                                sequence_mask)
      else:
        attention_mask = None

      with tf.variable_scope('encoder'):
        tf.logging.info("*** mha encoder ***")
        mha_attention_head_size = self.config.mha_hidden_size // self.config.mha_num_attention_heads
        tf.logging.info("*** mha_attention_head_size ***")
        tf.logging.info(mha_attention_head_size)

        pre_output = linear_proj

        conformer_block = conformer(pre_output,
            ffm_hidden_size=self.config.ffm_hidden_size,
            ffm_dropout_rate=self.config.ffm_dropout,
            ffm_fc_factor=self.config.ffm_fc_factor,
            ffm_expansion_factor=self.config.ffm_expansion_factor,
            mha_relative_position_embeddings=self.relative_position_embeddings,
            mha_num_attention_heads=self.config.mha_num_attention_heads,
            mha_attention_head_size=mha_attention_head_size,
            mha_attention_probs_dropout_prob=self.config.mha_attention_probs_dropout_prob,
            mha_hidden_dropout_prob=self.config.mha_hidden_dropout_prob,
            mha_initializer_range=self.config.mha_initializer_range,
            mha_use_relative_position=self.config.mha_use_relative_position,
            mha_num_hidden_layers=self.config.mha_num_hidden_layers,
            mha_attention_mask=self.attention_mask,
            conv_strides=self.config.cnn_strides,
            conv_depth_multiplier=self.config.cnn_depth_multiplier,
            conv_dropout_prob=self.config.cnn_dropout_prob,
            relative_position_type=self.config.mha_relative_position_type,
            is_training=is_training,
            is_global_bn=is_global_bn)
      with tf.variable_scope('fc_module'):
        fc_output = fc_block(conformer_block[-1],
                  fc_layers=config.fc_layers, 
                  hidden_size=config.fc_hidden_size, 
                  dropout_rate=config.fc_dropout_rate,
                  is_training=is_training)

      return conformer_block[-1], fc_output

  def get_conv_downsampling_output(self):
    return self.unmasked_conv_subsampling

  def get_sequence_output(self):
    return self.conformer_block[-1]

  def get_conv_reduction_factor(self):
    return self.reduction_factor

  def get_logits(self):
    return self.logits

  def get_fc_output(self):
    return self.fc_output

  def get_code_book(self, is_pretraining):
    if is_pretraining:
      tf.logging.info("** return code_book **")
      tf.logging.info(self.code_book)
      return self.code_book
    else:
      return None

  def get_code_discrete(self, is_pretraining):
    if is_pretraining:
      tf.logging.info("** return code_discrete **")
      tf.logging.info(self.code_discrete)
      return self.code_discrete
    else:
      return None

  def get_code_dense(self, is_pretraining):
    if is_pretraining:
      tf.logging.info("** return code_dense **")
      tf.logging.info(self.code_dense)
      return self.code_dense
    else:
      return None

  def get_code_loss(self, is_pretraining):
    if is_pretraining:
      tf.logging.info("** return code_loss_dict **")
      tf.logging.info(self.code_loss_dict)
      return self.code_loss_dict
    else:
      return None

def conformer(inputs,
            ffm_hidden_size,
            ffm_dropout_rate,
            ffm_fc_factor,
            ffm_expansion_factor,
            mha_relative_position_embeddings,
            mha_num_attention_heads,
            mha_attention_head_size,
            mha_attention_probs_dropout_prob=0.1,
            mha_hidden_dropout_prob=0.1,
            mha_initializer_range=0.02,
            mha_use_relative_position=True,
            mha_num_hidden_layers=12,
            mha_attention_mask=None,
            conv_kernel_sizes=31,
            conv_strides=1,
            conv_dropout_prob=0.1,
            conv_depth_multiplier=1,
            relative_position_type="relative_normal",
            is_training=False,
            is_global_bn=False):

  input_shape = get_shape_list(inputs, expected_rank=[3])
  batch_size = input_shape[0]
  seq_length = input_shape[1]

  pre_output = inputs
  conformer_block = []

  tf.logging.info("*** pre_output ***")
  tf.logging.info(pre_output)

  for layer_idx in range(mha_num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      with tf.variable_scope("residual_ffm_input"):
        ffm_outputs = residual_ffm_block(pre_output, 
                      hidden_size=ffm_hidden_size, 
                      dropout_rate=ffm_dropout_rate,
                      fc_factor=ffm_fc_factor,
                      expansion_factor=ffm_expansion_factor,
                      is_training=is_training)
        ffm_outputs = layer_norm(ffm_outputs)

        tf.logging.info("*** residual_ffm_input ***")
        tf.logging.info(ffm_outputs)

      with tf.variable_scope("attention"):
        with tf.variable_scope("self"):
          [attention_head, attention_scores] = transformer_relative_position.attention_layer(
                    from_tensor=ffm_outputs,
                    to_tensor=ffm_outputs,
                    attention_mask=mha_attention_mask,
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
        
          tf.logging.info("*** attention_head ***")
          tf.logging.info(attention_head)

        attention_head = tf.nn.dropout(attention_head, keep_prob=1-mha_hidden_dropout_prob)
        attention_output = layer_norm(attention_head + ffm_outputs)

        with tf.variable_scope("conformer_conv"):

          conv_output = conformer_conv(attention_output, 
                kernel_size=conv_kernel_sizes, 
                strides=conv_strides,
                depth_multiplier=conv_depth_multiplier,
                dropout_rate=conv_dropout_prob,
                is_training=is_training,
                is_global_bn=is_global_bn)

          tf.logging.info("****** conformer_conv ***")
          tf.logging.info(conv_output)

          conv_attention_output = layer_norm(conv_output + attention_output)

        with tf.variable_scope("residual_ffm_output"):
          layer_output = residual_ffm_block(conv_attention_output, 
                      hidden_size=ffm_hidden_size, 
                      dropout_rate=ffm_dropout_rate,
                      fc_factor=ffm_fc_factor,
                      expansion_factor=ffm_expansion_factor,
                      is_training=is_training)

          tf.logging.info("*** residual_ffm_output ***")
          tf.logging.info(layer_output)

          layer_output = layer_norm(layer_output)
        conformer_block.append(layer_output)
        pre_output = layer_output

  return conformer_block

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

def glu(inputs, axis=-1):
  a, b = tf.split(inputs, 2, axis=-1)
  b = tf.nn.sigmoid(b)
  return tf.multiply(a, b)

def conformer_conv(inputs, 
            kernel_size, 
            strides=1,
            depth_multiplier=1,
            dropout_rate=0.1,
            is_training=False,
            is_global_bn=False):

  # [batch, seq_len, dims]
  input_shape = get_shape_list(inputs, expected_rank=[3])
  input_dim = input_shape[-1]
  # [batch, seq_len, 1, dims]
  outputs = tf.expand_dims(inputs, 2)

  # [batch, seq_len, 1, filters*2]
  outputs = tf.layers.conv2d(
                  inputs=outputs, 
                  filters=input_dim*2, 
                  kernel_size=1, 
                  strides=1,
                  padding="valid", 
                  use_bias=True, 
                  activation=None,
                  kernel_initializer=tf.glorot_normal_initializer())

  # [batch, seq_len, 1, filters]
  outputs = glu(outputs, axis=-1)

  depthwise_filter = tf.get_variable("depthwise_filter",
                    (kernel_size, 1, input_dim, depth_multiplier),
                    dtype=tf.float32,
                    initializer=tf.glorot_normal_initializer())

  outputs = tf.nn.depthwise_conv2d(
    outputs, depthwise_filter, (1,1,1,1), 
    "SAME"
  )

  outputs = batch_norm(outputs, is_training=is_training,
                      is_global_bn=is_global_bn)

  outputs = gelu(outputs)

  # [batch, seq_len, 1, dims]
  outputs = tf.layers.conv2d(
                  inputs=outputs, 
                  filters=input_dim, 
                  kernel_size=1, 
                  strides=1,
                  padding="valid", 
                  use_bias=True, 
                  activation=None,
                  kernel_initializer=tf.glorot_normal_initializer())

  # [batch, seq_len, 1, dims]
  outputs = tf.squeeze(outputs, axis=2)

  outputs = tf.nn.dropout(outputs, keep_prob=1-dropout_rate)
  return outputs

def residual_ffm_block(inputs, hidden_size, 
                  dropout_rate,
                  fc_factor,
                  expansion_factor,
                  is_training=False):

  input_shape = get_shape_list(inputs, expected_rank=[3])
  outputs = tf.layers.dense(inputs, 
                    units=expansion_factor*hidden_size
          )
  outputs = gelu(outputs)
  outputs = tf.nn.dropout(outputs, keep_prob=1-dropout_rate)
  outputs = tf.layers.dense(outputs, 
                    units=input_shape[-1])
  outputs = tf.nn.dropout(outputs, keep_prob=1-dropout_rate)
  outputs = inputs + fc_factor * outputs
  return outputs

def batch_norm(inputs, is_training, 
              batch_norm_decay=0.997, 
              batch_norm_eps=1e-5,
              is_global_bn=False):
  try:
    from model.global_bn_utils import batch_norm as global_batch_norm

    return global_batch_norm(inputs=inputs,
            is_training=is_training, 
            batch_norm_decay=batch_norm_decay,
            batch_norm_eps=batch_norm_eps,
            is_global_bn=is_global_bn)
  except:
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
                  is_batch_norm=False,
                  is_training=False,
                  is_padding=True,
                  is_global_bn=False
                  ):

  if is_padding:
    paddings = [k_size//2 for k_size in kernel_size]
    inputs = tf.pad(
        inputs,
        [[0, 0], [paddings[0], paddings[0]], [paddings[1], paddings[1]], [0, 0]])
    tf.logging.info("** apply cnn padding **")
    tf.logging.info(paddings)
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
    inputs = batch_norm(inputs, is_training, 
                        is_global_bn=is_global_bn)
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
              is_padding=True,
              is_global_bn=False):

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
                    is_padding=is_padding,
                    is_global_bn=is_global_bn
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
