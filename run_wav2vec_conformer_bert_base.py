# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from model import conformer 
from optimizer.optimizer_utils import (
    create_optimizer, 
    create_adam_optimizer, 
    naive_create_optimizer,
    naive_create_adam_optimizer,
    naive_create_optimizer_no_global,
    naive_create_adam_optimizer_no_global,
    naive_create_adamax_optimizer,
    naive_create_adamax_optimizer_no_global,
    naive_create_adafactor_optimizer,
    naive_create_adafactor_optimizer_no_global)
import tensorflow as tf
from audio_io import audio_featurizer_tf, read_audio
from augment_io import augment_tf
from loss import ctc_loss
import json
from audio_io import utils as audio_utils
from loss import ctc_ops
from augment_io import span_mask
from loss import circle_loss_utils
from utils import log_utils
from model import modeling_relative_position  

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def check_tf_version():
  version = tf.__version__
  print("==tf version==", version)
  tf.logging.info("** version **")
  tf.logging.info(version)
  if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
    return True
  else:
    return False

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("buckets", "", "oss buckets")
## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "train_file", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("num_gpus", 8, "Total batch size for training.")
flags.DEFINE_integer("num_accumulated_batches", 1, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("weight_decay_rate", 0.9, "The initial learning rate for Adam.")
flags.DEFINE_float("warmup_proportion", 0.1, "The initial learning rate for Adam.")
flags.DEFINE_float("lr_decay_power", 1.0, "The initial learning rate for Adam.")
flags.DEFINE_float("layerwise_lr_decay_power", 0.0, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")
flags.DEFINE_float("train_examples", 1000000.0,
                   "Total number of training epochs to perform.")
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")
flags.DEFINE_integer("log_step_count_steps", 100,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("num_train_steps", 1000,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("num_warmup_steps", 1000,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("keep_checkpoint_max", 10,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("max_duration", 20,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("samples_per_second", 8000,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("transcript_seq_length", 96,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("num_predict", 75,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("min_tok", 3,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("max_tok", 10,
                     "How many steps to make in each estimator call.")
flags.DEFINE_float("mask_prob", 0.15,
                     "How many steps to make in each estimator call.")
flags.DEFINE_float("circle_margin", 0.25,
                     "How many steps to make in each estimator call.")
flags.DEFINE_float("circle_gamma", 32,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("do_distributed_training", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string(
    "ctc_loss_type", "sparse_ctc",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "audio_featurizer_config_path", "audio_featurizer_config_path",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "featurizer_aug_config_path", "featurizer_aug_config_path",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "distributed_mode", "all_reduce",
    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("if_focal_ctc", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("is_pretraining", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("monitoring", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "target_feature_mode", "soft-em",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_bool("is_global_bn", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string("blank_index", "-1",
                     "How many steps to make in each estimator call.")
flags.DEFINE_string(
    "output_mode", "char",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "optimizer_type", "adafactor",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "decoder_type", "fc",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "bert_lm_config", "fc",
    "Initial checkpoint (usually from a pre-trained BERT model).")

def create_model(model_config, 
                ipnut_features,
                is_training,
                input_transcripts,
                ctc_loss_type,
                if_calculate_loss=True,
                unique_indices=None,
                input_length=None,
                time_feature_mask=None,
                freq_feature_mask=None):
  """Creates a classification model."""
  model = conformer.Conformer(
      config=model_config,
      sequences=ipnut_features, 
      is_training=is_training,
      input_length=input_length,
      is_pretraining=FLAGS.is_pretraining,
      time_feature_mask=time_feature_mask,
      freq_feature_mask=freq_feature_mask,
      target_feature_mode=FLAGS.target_feature_mode,
      is_global_bn=FLAGS.is_global_bn,
      decoder_type=FLAGS.decoder_type)

  sequence_output = model.get_sequence_output()

  # [batch_size, time-steps, vocab_size]
  if input_length is not None:
    tf.logging.info("*** apply reduced input length ***")
    input_length = tf.identity(input_length)
  else:
    tf.logging.info("******** apply padded input length ***")
    logits_shape = shape_list(logits)
    input_length = logits_shape[1] * tf.ones_like(label_length)

  reduction_factor = model.get_conv_reduction_factor()
  reduced_length = audio_utils.get_reduced_length(input_length, reduction_factor)

  tf.logging.info("*** reduced_length ***")
  tf.logging.info(reduced_length)

  tf.logging.info("*** label_length ***")
  tf.logging.info(label_length)

  

def model_fn_builder(model_config, 
                init_checkpoint, 
                ctc_loss_type,
                learning_rate, 
                num_train_steps, 
                num_warmup_steps,
                output_dir,
                use_tpu,
                reduced_factor):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s, dtype = %s" % (name, features[name].shape, features[name].dtype))

    clean_feature = features["clean_feature"]
    noise_feature = features["noise_feature"]
    noise_aug_feature = features["noise_aug_feature"]
    clean_aug_feature = features["clean_aug_feature"]
    feature_seq_length = features["feature_seq_length"]

    transcript_id = features["transcript_id"]
    gender_id = features["gender_id"]
    dialect_id = features["dialect_id"]
    speaker_id = features["speaker_id"]
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (clean_aug_loss, 
    clean_aug_per_example_loss, 
    clean_aug_logits,
    clean_aug_audio_embedding,
    clean_valid_loss_mask) = create_model(
        model_config=model_config,
        ipnut_features=clean_aug_feature,
        input_transcripts=transcript_id,
        is_training=is_training,
        ctc_loss_type=ctc_loss_type,
        # unique_indices=(features['unique_labels'],
        #                 features['unique_indices']),
        unique_indices=None,
        if_calculate_loss=True,
        input_length=feature_seq_length)

    (noise_aug_loss, 
    noise_aug_per_example_loss, 
    noise_aug_logits,
    noise_aug_audio_embedding,
    noise_valid_loss_mask) = create_model(
        model_config=model_config,
        ipnut_features=noise_aug_feature,
        input_transcripts=transcript_id,
        is_training=is_training,
        ctc_loss_type=ctc_loss_type,
        # unique_indices=(features['unique_labels'],
        #                 features['unique_indices']),
        unique_indices=None,
        if_calculate_loss=True,
        input_length=feature_seq_length)

    total_loss = (clean_aug_loss + noise_aug_loss)
    # total_loss = (noise_loss + clean_loss)
    total_loss = total_loss / 2.0

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = conformer.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      print("===update_ops===", update_ops)
      
      encoder_params = []
      for scope in ['conformer/conv_downsampling',
                    'conformer/linear_proj',
                    "conformer/encoder"]:
        encoder_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

      for params in encoder_params:
        tf.logging.info("** encoder_params **")
        tf.logging.info(params)

      decoder_params = []
      for scope in ['conformer/fc_module',
                    'conformer/decoder',
                    'conformer/cls']:
        decoder_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

      for params in decoder_params:
        tf.logging.info("** decoder_params **")
        tf.logging.info(params)

      if FLAGS.optimizer_type == 'adafactor':
        optimizer_fn = naive_create_adafactor_optimizer_no_global
        tf.logging.info("** adafactor **")
        tf.logging.info(optimizer_fn)
      elif FLAGS.optimizer_type == "adamax":
        optimizer_fn = naive_create_adamax_optimizer_no_global
        tf.logging.info("** adamax **")
        tf.logging.info(optimizer_fn)
      elif FLAGS.optimizer_type == 'adam_decay':
        optimizer_fn = naive_create_optimizer_no_global
        tf.logging.info(optimizer_fn)

      global_step = tf.train.get_or_create_global_step()
      with tf.control_dependencies(update_ops):

        [train_enc_op, 
        enc_learning_rate] = optimizer_fn(
          total_loss, 
          learning_rate, 
          num_train_steps, 
          weight_decay_rate=FLAGS.weight_decay_rate,
          use_tpu=use_tpu,
          warmup_steps=num_warmup_steps,
          lr_decay_power=FLAGS.lr_decay_power,
          layerwise_lr_decay_power=FLAGS.layerwise_lr_decay_power,
          tvars=encoder_params
          )

        [train_dec_op, 
        dec_learning_rate] = optimizer_fn(
          total_loss, 
          learning_rate if FLAGS.decoder_type == "fc" else learning_rate*2.0, 
          num_train_steps, 
          weight_decay_rate=FLAGS.weight_decay_rate,
          use_tpu=use_tpu,
          warmup_steps=num_warmup_steps,
          lr_decay_power=FLAGS.lr_decay_power,
          layerwise_lr_decay_power=FLAGS.layerwise_lr_decay_power,
          tvars=decoder_params
          )

        new_global_step = global_step + 1
        with tf.control_dependencies([train_enc_op, train_dec_op]):
          train_op = global_step.assign(new_global_step)

      hook_dict = {}
      hook_dict['noise_loss'] = noise_aug_loss
      hook_dict['clean_loss'] = clean_aug_loss
      reduced_length = audio_utils.get_reduced_length(feature_seq_length, reduced_factor)
      
      tf.logging.info("*** reduced_length ***")
      tf.logging.info(reduced_length)

      tf.logging.info("*** clean_valid_loss_mask ***")
      tf.logging.info(clean_valid_loss_mask)

      hook_dict['seq_length'] = tf.reduce_mean(reduced_length)
      hook_dict['avg_valid_num'] = tf.reduce_sum(clean_valid_loss_mask)

      hook_dict['dec_learning_rate'] = dec_learning_rate
      hook_dict['enc_learning_rate'] = enc_learning_rate

      if FLAGS.monitoring and hook_dict:
        host_call = log_utils.construct_scalar_host_call_v1(
                                    monitor_dict=hook_dict,
                                    model_dir=output_dir,
                                    prefix="train/")
      else:
        host_call = None

      print(host_call, "====host_call=====")

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          host_call=host_call
          )
        
    return output_spec

  return model_fn

def check_tf_version():
  version = tf.__version__
  print("==tf version==", version)
  if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
    return True
  else:
    return False

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(input_file, 
                    is_training, 
                    drop_remainder,
                    audio_featurizer, 
                    feature_augmenter,
                    max_duration,
                    transcript_seq_length,
                    samples_per_second=8000,
                    batch_size=32,
                    use_tpu=False,
                    num_predict=76,
                    mask_prob=0.15,
                    min_tok=3,
                    max_tok=10
                    ):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  def input_fn(params):
    name_to_features = {
      "clean_audio_resample": tf.FixedLenFeature([], tf.string),
      "noise_audio_resample": tf.FixedLenFeature([], tf.string),
      "speaker_id": tf.FixedLenFeature([], tf.int64),
      "noise_id": tf.FixedLenFeature([], tf.int64),
      "gender_id": tf.FixedLenFeature([], tf.int64),
      "dialect_id": tf.FixedLenFeature([], tf.int64),
      "transcript_id": tf.FixedLenFeature([transcript_seq_length], tf.int64),
      "transcript_pinyin_id": tf.FixedLenFeature([transcript_seq_length], tf.int64)
    }
    batch_size = params["batch_size"]
    def _decode_record(record, name_to_features):
      """Decodes a   record to a TensorFlow example."""
      example = tf.parse_single_example(record, name_to_features)

      # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
      # So cast all int64 to int32.
      # with tf.device("/CPU:0"):
      noise_audio = read_audio.tf_read_raw_audio(example['noise_audio_resample'], 
                        samples_per_second=samples_per_second,
                          use_tpu=use_tpu)
      noise_feature = audio_featurizer.tf_extract(noise_audio)
      noise_aug_feature = feature_augmenter.after.augment(noise_feature)

      clean_audio = read_audio.tf_read_raw_audio(example['clean_audio_resample'], 
                        samples_per_second=samples_per_second,
                          use_tpu=use_tpu)
      clean_feature = audio_featurizer.tf_extract(clean_audio)
      clean_aug_feature = feature_augmenter.after.augment(clean_feature)
      
      feature_length = int(audio_featurizer.max_length)

      # noise_feature = noise_feature[:feature_length, :, :]
      # noise_aug_feature = noise_aug_feature[:feature_length, :, :]
      # clean_feature = clean_feature[:feature_length, :, :]
      # clean_aug_feature = clean_aug_feature[:feature_length, :, :]
      
      # [T, D, 1] 
      output_examples = {}

      if FLAGS.output_mode == 'char':
        output_examples['transcript_id'] = tf.cast(example['transcript_id'], dtype=tf.int32)
        tf.logging.info("*** apply char transcript_id ***")
      elif FLAGS.output_mode == 'pinyin':
        output_examples['transcript_id'] = tf.cast(example['transcript_pinyin_id'], dtype=tf.int32)
        tf.logging.info("*** apply pinyin transcript_id ***")

      output_examples['clean_feature'] = tf.cast(clean_feature, dtype=tf.float32)
      output_examples['noise_feature'] = tf.cast(noise_feature, dtype=tf.float32)
      output_examples['clean_aug_feature'] = tf.cast(clean_aug_feature, dtype=tf.float32)
      output_examples['noise_aug_feature'] = tf.cast(noise_aug_feature, dtype=tf.float32)
      # output_examples['clean_audio'] = tf.cast(clean_audio, dtype=tf.float32)
      # output_examples['noise_audio'] = tf.cast(noise_audio, dtype=tf.float32)
      output_examples['speaker_id'] = tf.cast(example['speaker_id'], dtype=tf.int32)
      output_examples['gender_id'] = tf.cast(example['gender_id'], dtype=tf.int32)
      output_examples['dialect_id'] = tf.cast(example['dialect_id'], dtype=tf.int32)
      feature_shape = shape_list(clean_feature)
      # [T, V, 1]
      output_examples['feature_seq_length'] = tf.cast(feature_shape[0], dtype=tf.int32)
      # [unique_labels, 
      # unique_indices] = ctc_ops.ctc_unique_labels_single(
      #         tf.cast(example['transcript_id'], dtype=tf.int32)
      #         )
      # indices_padded_values = unique_indices[-1]
      
      # unique_label_shape = shape_list(unique_labels)
      # unique_indices_shape = shape_list(unique_indices)

      # unique_labels = tf.expand_dims(unique_labels, axis=0)
      # unique_indices = tf.expand_dims(unique_indices, axis=0)
      
      # unique_labels = tf.pad(unique_labels, [[0,0],[0,transcript_seq_length-unique_label_shape[0]]])
      # unique_indices = tf.pad(unique_indices, [[0,0],[0,transcript_seq_length-unique_indices_shape[0]]], constant_values=indices_padded_values)
      
      if check_tf_version():
        [unique_labels, 
        unique_indices] = tf.nn.ctc_unique_labels(
              tf.cast(tf.expand_dims(output_examples['transcript_id'], axis=0), dtype=tf.int32)
              )
        tf.logging.info("** apply original ctc unique labels **")
      else:
        [unique_labels, 
        unique_indices] = ctc_ops.ctc_unique_labels(
                tf.cast(tf.expand_dims(output_examples['transcript_id'], axis=0), dtype=tf.int32)
                )
        tf.logging.info("** apply re-implemented ctc unique labels **")

      unique_labels = tf.squeeze(unique_labels, axis=0)
      unique_indices = tf.squeeze(unique_indices, axis=0)

      output_examples['unique_labels'] = tf.cast(unique_labels, dtype=tf.int32)
      output_examples['unique_indices'] = tf.cast(unique_indices, dtype=tf.int32)
      
      reduced_length = audio_utils.get_reduced_length(feature_shape[0], audio_featurizer.get_reduced_factor())

      print(reduced_length, audio_featurizer.get_reduced_length(), "====")
      inputs = tf.sequence_mask(reduced_length, audio_featurizer.get_reduced_length())
      inputs = tf.cast(inputs, dtype=tf.int32)

      tf.logging.info("*** inputs ***")
      tf.logging.info(inputs)

      tf.logging.info("*** get_reduced_length ***")
      tf.logging.info(audio_featurizer.get_reduced_length())

      # span_mask_examples = span_mask.mask_generator(inputs, 
      #             audio_featurizer.get_reduced_length(), 
      #             num_predict=num_predict,
      #             mask_prob=mask_prob,
      #             stride=1, 
      #             min_tok=min_tok, 
      #             max_tok=max_tok)
      # output_examples['masked_mask'] = 1.0 - span_mask_examples['masked_mask']
      # output_examples['masked_positions'] = span_mask_examples['masked_positions']
      # output_examples['masked_weights'] = span_mask_examples['masked_weights']
      
      # add in utterance negative sample
      for name in list(output_examples.keys()):
        t = output_examples[name]
        if t.dtype == tf.int64:
          t = tf.to_int32(t)
        output_examples[name] = t
      return output_examples

    """The actual input function."""

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.map(lambda record: _decode_record(record, name_to_features))

    d = d.padded_batch(
              batch_size=batch_size,
              padded_shapes={
                "clean_feature":tf.TensorShape(audio_featurizer.shape),
                "noise_feature":tf.TensorShape(audio_featurizer.shape),
                "clean_aug_feature":tf.TensorShape(audio_featurizer.shape),
                "noise_aug_feature":tf.TensorShape(audio_featurizer.shape),
                # "clean_audio":tf.TensorShape([max_duration*samples_per_second]),
                # "noise_audio":tf.TensorShape([max_duration*samples_per_second]),
                "speaker_id":tf.TensorShape([]),
                "transcript_id":tf.TensorShape([transcript_seq_length]),
                "gender_id":tf.TensorShape([]),
                "dialect_id":tf.TensorShape([]),
                "unique_labels":tf.TensorShape([transcript_seq_length]),
                "unique_indices":tf.TensorShape([transcript_seq_length]),
                "feature_seq_length":tf.TensorShape([]),
                # "masked_positions":tf.TensorShape([num_predict]),
                # "masked_weights":tf.TensorShape([num_predict]),
                # "masked_mask":tf.TensorShape([audio_featurizer.get_reduced_length()]),
              },
              padding_values={
                "clean_feature":0.0,
                "noise_feature":0.0,
                "clean_aug_feature":0.0,
                "noise_aug_feature":0.0,
                # "clean_audio":0.0,
                # "noise_audio":0.0,
                "speaker_id":0,
                "transcript_id":0,
                "gender_id":0,
                "dialect_id":0,
                "unique_labels":0,
                "unique_indices":0,
                "feature_seq_length":0,
                # "masked_positions":0,
                # "masked_weights":0.0,
                # "masked_mask":0.0
              },
              drop_remainder=drop_remainder
          )

    d = d.prefetch(batch_size*10)
    d = d.apply(tf.data.experimental.ignore_errors())
    return d
  return input_fn

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  model_config = conformer.ConformerConfig.from_json_file(FLAGS.bert_config_file)

  tf_version = check_tf_version()

  if int(FLAGS.blank_index) != 0:
    model_config.__dict__['vocab_size'] += 1
    tf.logging.info("** blank_index is added to the vocab-size")
    tf.logging.info(model_config.__dict__['vocab_size'])

  config_name = FLAGS.bert_config_file.split("/")[-1]
  import os
  output_dir = os.path.join(FLAGS.buckets, FLAGS.output_dir)

  tf.gfile.MakeDirs(output_dir)
  if FLAGS.init_checkpoint:
    init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
  else:
    init_checkpoint = None

  tf.logging.info("** init_checkpoint **")
  tf.logging.info(init_checkpoint)
  
  import os
  with tf.gfile.GFile(os.path.join(FLAGS.buckets, FLAGS.output_dir, config_name), "w") as fwobj:
    fwobj.write(model_config.to_json_string()+"\n")

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_file = os.path.join(FLAGS.buckets, FLAGS.train_file)
  print("*** train file ***", train_file)
  input_files = []
  
  with tf.gfile.GFile(train_file, "r") as reader:
    for index, line in enumerate(reader):
      content = line.strip()
      if 'tfrecord' in content:
        train_file_path = os.path.join(FLAGS.buckets, FLAGS.data_dir, content)
        # print(train_file_path, "====train_file_path====")
        input_files.append(train_file_path)
  print("==total input files==", len(input_files))
  import random
  random.shuffle(input_files)

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  audio_featurizer_config_path = os.path.join(FLAGS.buckets, FLAGS.audio_featurizer_config_path)
  with tf.gfile.Open(audio_featurizer_config_path, "r") as frobj:
    audio_featurizer_config = json.load(frobj)

  audio_featurizer = audio_featurizer_tf.TFSpeechFeaturizer(audio_featurizer_config)
  max_feature_length = audio_featurizer.get_length_from_duration(FLAGS.max_duration)
  audio_featurizer.update_length(max_feature_length)

  audio_featurizer.update_reduced_factor(model_config.reduction_factor)

  featurizer_aug_config_path = os.path.join(FLAGS.buckets, FLAGS.featurizer_aug_config_path)
  with tf.gfile.Open(featurizer_aug_config_path, "r") as frobj:
    featurizer_aug_config = json.load(frobj)
    tf.logging.info("** featurizer_aug_config **")
    tf.logging.info(featurizer_aug_config)
  feature_augmenter = augment_tf.Augmentation(featurizer_aug_config, 
                                            use_tf=True)

  model_fn = model_fn_builder(
      model_config=model_config,
      init_checkpoint=init_checkpoint,
      ctc_loss_type=FLAGS.ctc_loss_type,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      output_dir=output_dir,
      use_tpu=FLAGS.use_tpu,
      reduced_factor=audio_featurizer.get_reduced_factor())

  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  train_input_fn = input_fn_builder(
        input_file=input_files,
        is_training=True,
        drop_remainder=True,
        audio_featurizer=audio_featurizer,
        feature_augmenter=feature_augmenter,
        max_duration=FLAGS.max_duration,
        samples_per_second=FLAGS.samples_per_second,
        transcript_seq_length=FLAGS.transcript_seq_length,
        batch_size=FLAGS.train_batch_size,
        use_tpu=FLAGS.use_tpu,
        num_predict=FLAGS.num_predict,
        mask_prob=FLAGS.mask_prob,
        min_tok=FLAGS.min_tok,
        max_tok=FLAGS.max_tok
        )

  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

if __name__ == "__main__":
  tf.app.run()

