# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#.   sdfhj
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
from optimizer.optimizer_utils import create_optimizer, create_adam_optimizer
import tensorflow as tf
from audio_io import audio_featurizer_tf, read_audio
from augment_io import augment_tf
from loss import ctc_loss
import json
from audio_io import utils as audio_utils
from loss import ctc_ops
from augment_io import span_mask
from loss import circle_loss_utils

# import subprocess
# subprocess.call(['sh', './deepspeech/install_warpctc.sh'])

# from tensorflow_binding import warpctc_tensorflow

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("buckets", "", "oss buckets")
flags.DEFINE_integer("task_index", 0, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")

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

flags.DEFINE_float("alpha", 0.5,
                     "How many steps to make in each estimator call.")
flags.DEFINE_float("gamma", 2.0,
                     "How many steps to make in each estimator call.")
flags.DEFINE_string(
    "target_feature_mode", "soft-em",
    "Initial checkpoint (usually from a pre-trained BERT model).")

# with tf.gfile.GFile(os.path.join(FLAGS.buckets, FLAGS.output_dir, 'evn.txt'), "w") as fwobj:
#   tf_src_path = "/".join(tf.sysconfig.get_include().split("/")[:-2]+['tensorflow'])
#   fwobj.write(tf_src_path+"\n")
#   print(tf_src_path, "==src path==")
#   print(tf.sysconfig.get_lib(), "==lib path==")
#   fwobj.write(tf.sysconfig.get_lib()+"\n")
#   fwobj.write(tf.sysconfig.get_include()+"\n")
#   print(tf.sysconfig.get_include(), "==include path==")

# import subprocess
# subprocess.call(['sh', './deepspeech/install_warpctc.sh'])

# try:
#   from loss.warp_ctc_loss import warpctc_loss
# except:
#   warpctc_loss = None

# from loss.warp_ctc_loss import warpctc_loss

def create_model(model_config, 
                ipnut_features,
                is_training,
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
      target_feature_mode=FLAGS.target_feature_mode)

  sequence_output = model.get_sequence_output()
  code_discrete = model.get_code_discrete(True)
  code_dense = model.get_code_dense(True)
  code_loss_dict = model.get_code_loss(True)

  return (sequence_output, 
        code_discrete, 
        code_dense,
        code_loss_dict)

def model_fn_builder(model_config, 
                init_checkpoint, 
                learning_rate, 
                num_train_steps, 
                num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    clean_feature = features["clean_feature"]
    noise_feature = features["noise_feature"]
    feature_seq_length = features["feature_seq_length"]

    time_feature_mask = features["masked_mask"]
    masked_positions = features["masked_positions"]
    masked_weights = features["masked_weights"]

    gender_id = features["gender_id"]
    dialect_id = features["dialect_id"]
    speaker_id = features["speaker_id"]
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (clean_sequence_output, 
    clean_code_discrete, 
    clean_code_dense,
    clean_code_loss_dict) = create_model(
        model_config=model_config,
        ipnut_features=clean_feature,
        is_training=is_training,
        input_length=feature_seq_length,
        time_feature_mask=time_feature_mask)

    (clean_loss, 
    clean_per_example_loss) = get_masked_lm_output(
            clean_sequence_output, 
            masked_positions,
            clean_code_dense,
            masked_weights,
            margin=FLAGS.circle_margin,
            gamma=FLAGS.circle_gamma)

    (noise_sequence_output, 
    noise_code_discrete, 
    noise_code_dense,
    noise_code_loss_dict) = create_model(
        model_config=model_config,
        ipnut_features=noise_feature,
        is_training=is_training,
        input_length=feature_seq_length,
        time_feature_mask=time_feature_mask)

    (noise_loss, 
    noise_per_example_loss) = get_masked_lm_output(
            noise_sequence_output, 
            masked_positions,
            noise_code_dense,
            masked_weights,
            margin=FLAGS.circle_margin,
            gamma=FLAGS.circle_gamma)

    total_loss = (noise_loss+clean_loss) / 2.0
    for key in noise_code_loss_dict:
      tf.logging.info(key)
      tf.logging.info(noise_code_loss_dict[key])
      total_loss += noise_code_loss_dict[key]

    for key in clean_code_loss_dict:
      tf.logging.info(key)
      tf.logging.info(noise_code_loss_dict[key])
      total_loss += clean_code_loss_dict[key]

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = deepspeech.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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
      print("==update_ops==", update_ops)
      with tf.control_dependencies(update_ops):
        train_op, output_learning_rate = create_optimizer(
            total_loss, learning_rate, num_train_steps, 
            weight_decay_rate=FLAGS.weight_decay_rate,
            warmup_steps=num_warmup_steps, 
            use_tpu=FLAGS.use_tpu,
            warmup_proportion=FLAGS.warmup_proportion,
            lr_decay_power=FLAGS.lr_decay_power,
            layerwise_lr_decay_power=FLAGS.layerwise_lr_decay_power,
            n_transformer_layers=model_config.num_hidden_layers,
            task_layers=[])
    
      hook_dict = {}
      hook_dict['noise_loss'] = noise_loss
      hook_dict['clean_loss'] = clean_loss
      for key in noise_code_loss_dict:
        hook_dict["noise_{}".format(key)] = noise_code_loss_dict[key]
      for key in clean_code_loss_dict:
        hook_dict["clean_{}".format(key)] = clean_code_loss_dict[key]
      
      hook_dict['learning_rate'] = output_learning_rate
      logging_hook = tf.train.LoggingTensorHook(
        hook_dict, every_n_iter=100)
      training_hooks = []
      training_hooks.append(logging_hook)
        
      output_spec = tf.estimator.EstimatorSpec(mode=mode, 
          loss=total_loss, 
          train_op=train_op,
          training_hooks=training_hooks)
    return output_spec

  return model_fn

def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = shape_list(sequence_tensor)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

def get_masked_lm_output(
            input_tensor, 
            positions,
            code_dense,
            label_weights,
            margin=0.25,
            gamma=32):
  """Get loss and log probs for the masked LM."""
  
  # [batch_size*num_predict, dims]
  masked_input_tensor = gather_indexes(input_tensor, positions)
  unmasked_code_tensor = gather_indexes(code_dense, positions)

  masked_input_tensor = tf.nn.l2_normalize(masked_input_tensor, axis=-1)
  unmasked_code_tensor = tf.nn.l2_normalize(unmasked_code_tensor, axis=-1)

  masked_tensor_shape = shape_list(masked_input_tensor)
  # [batch_size*num_predict]
  label_weights = tf.reshape(label_weights, [-1])
  # [batch_size*num_predict, 1]
  label_mask = tf.expand_dims(label_weights, axis=-1)

  # [batch_size*num_predict, batch_size*num_predict]
  similarity_matrix = tf.matmul(
            masked_input_tensor, 
            unmasked_code_tensor,
            transpose_b=True)

  print(similarity_matrix)

  pos_label_mask = tf.eye(masked_tensor_shape[0]) * label_mask
  neg_label_mask = (1.0 - pos_label_mask) * label_mask

  neg_sample_prob = tf.random_uniform(
              [masked_tensor_shape[0], masked_tensor_shape[0]],
              minval=1e-5,
              maxval=0.999,
              dtype=tf.float32)

  neg_sample_label = tf.cast(tf.greater_equal(neg_sample_prob, 0.4), dtype=tf.float32)
  neg_label_mask *= neg_sample_label

  print(similarity_matrix, pos_label_mask, neg_label_mask)

  per_example_loss = circle_loss_utils.circle_loss(
                similarity_matrix, 
                pos_label_mask, 
                neg_label_mask,
                margin=0.25,
                gamma=32)

  loss = tf.reduce_sum(per_example_loss*label_weights) / (1e-10+tf.reduce_sum(label_weights))

  return (loss, per_example_loss)

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
                    worker_count=None,
                    task_index=0,
                    distributed_mode='all_reduce',
                    num_predict=76,
                    mask_prob=0.15,
                    min_tok=3,
                    max_tok=10
                    ):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "clean_audio_resample": tf.FixedLenFeature([], tf.string),
      "noise_audio_resample": tf.FixedLenFeature([], tf.string),
      "speaker_id": tf.FixedLenFeature([], tf.int64),
      "noise_id": tf.FixedLenFeature([], tf.int64),
      "gender_id": tf.FixedLenFeature([], tf.int64),
      "dialect_id": tf.FixedLenFeature([], tf.int64),
      "transcript_id": tf.FixedLenFeature([transcript_seq_length], tf.int64)
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
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

    # [T, D, 1]
    output_examples = {}

    output_examples['clean_feature'] = tf.cast(clean_feature, dtype=tf.float32)
    output_examples['noise_feature'] = tf.cast(noise_feature, dtype=tf.float32)
    output_examples['clean_aug_feature'] = tf.cast(clean_aug_feature, dtype=tf.float32)
    output_examples['noise_aug_feature'] = tf.cast(noise_aug_feature, dtype=tf.float32)
    output_examples['clean_audio'] = tf.cast(clean_audio, dtype=tf.float32)
    output_examples['noise_audio'] = tf.cast(noise_audio, dtype=tf.float32)
    output_examples['speaker_id'] = tf.cast(example['speaker_id'], dtype=tf.int32)
    output_examples['transcript_id'] = tf.cast(example['transcript_id'], dtype=tf.int32)
    output_examples['gender_id'] = tf.cast(example['gender_id'], dtype=tf.int32)
    output_examples['dialect_id'] = tf.cast(example['dialect_id'], dtype=tf.int32)
    feature_shape = shape_list(noise_feature)
    # [T, V, 1]
    output_examples['feature_seq_length'] = tf.cast(feature_shape[0], dtype=tf.int32)
    [unique_labels, 
    unique_indices] = ctc_ops.ctc_unique_labels(
            tf.cast(example['transcript_id'], dtype=tf.int32)
            )
    output_examples['unique_labels'] = tf.cast(unique_labels, dtype=tf.int32)
    output_examples['unique_indices'] = tf.cast(unique_indices, dtype=tf.int32)
    
    reduced_length = audio_utils.get_reduced_length(feature_shape[0], audio_featurizer.get_reduced_factor())

    print(reduced_length, audio_featurizer.get_reduced_length(), "====")
    inputs = tf.sequence_mask(reduced_length, audio_featurizer.get_reduced_length())
    inputs = tf.cast(inputs, dtype=tf.int32)
    span_mask_examples = span_mask.mask_generator(inputs, 
                audio_featurizer.get_reduced_length(), 
                num_predict=num_predict,
                mask_prob=mask_prob,
                stride=1, 
                min_tok=min_tok, 
                max_tok=max_tok)
    output_examples['masked_mask'] = 1 - span_mask_examples['masked_mask']
    output_examples['masked_positions'] = span_mask_examples['masked_positions']
    output_examples['masked_weights'] = span_mask_examples['masked_weights']
    
    print(output_examples['masked_mask'], output_examples['masked_positions'], output_examples['masked_weights'])

    # random_prob = tf.random_uniform(
    #           [reduced_length, reduced_length],
    #           minval=1e-5,
    #           maxval=0.999,
    #           dtype=tf.float32)
    # invalid_mask = 1.0 - tf.eye(reduced_length)
    # invalid_mask *= (1 - span_mask_examples['masked_mask'])

    # neg_mask = tf.cast(tf.great(random_prob, 0.3), dtype=tf.float32)
    # output_examples['neg_mask'] = neg_mask

    return output_examples

  """The actual input function."""

  # For training, we want a lot of parallel reading and shuffling.
  # For eval, we want no shuffling and parallel reading doesn't matter.
  d = tf.data.TFRecordDataset(input_file)
  if is_training:
    d = d.repeat()
    d = d.shuffle(buffer_size=100)

  if distributed_mode == 'collective_reduce':
    if worker_count and task_index:
      d = d.shard(worker_count, task_index)

  d = d.map(lambda record: _decode_record(record, name_to_features))

  d = d.padded_batch(
            batch_size=batch_size,
            padded_shapes={
              "clean_feature":tf.TensorShape(audio_featurizer.shape),
              "noise_feature":tf.TensorShape(audio_featurizer.shape),
              "clean_aug_feature":tf.TensorShape(audio_featurizer.shape),
              "noise_aug_feature":tf.TensorShape(audio_featurizer.shape),
              "clean_audio":tf.TensorShape([max_duration*samples_per_second]),
              "noise_audio":tf.TensorShape([max_duration*samples_per_second]),
              "speaker_id":tf.TensorShape([]),
              "transcript_id":tf.TensorShape([transcript_seq_length]),
              "gender_id":tf.TensorShape([]),
              "dialect_id":tf.TensorShape([]),
              "unique_labels":tf.TensorShape([transcript_seq_length]),
              "unique_indices":tf.TensorShape([transcript_seq_length]),
              "feature_seq_length":tf.TensorShape([]),
              "masked_positions":tf.TensorShape([num_predict]),
              "masked_weights":tf.TensorShape([num_predict]),
              "masked_mask":tf.TensorShape([audio_featurizer.get_reduced_length()]),
            },
            padding_values={
              "clean_feature":0.0,
              "noise_feature":0.0,
              "clean_aug_feature":0.0,
              "noise_aug_feature":0.0,
              "clean_audio":0.0,
              "noise_audio":0.0,
              "speaker_id":-1,
              "transcript_id":0,
              "gender_id":-1,
              "dialect_id":-1,
              "unique_labels":0,
              "unique_indices":0,
              "feature_seq_length":0,
              "masked_positions":0,
              "masked_weights":0.0,
              "masked_mask":0.0
            },
            drop_remainder=drop_remainder
        )

  d = d.prefetch(batch_size*10)
  d = d.apply(tf.data.experimental.ignore_errors())
  return d

def make_distributed_info_without_evaluator():
  worker_hosts = FLAGS.worker_hosts.split(",")
  if len(worker_hosts) > 1:
    cluster = {"chief": [worker_hosts[0]],
         "worker": worker_hosts[1:]}
  else:
    cluster = {"chief": [worker_hosts[0]]}

  if FLAGS.task_index == 0:
    task_type = "chief"
    task_index = 0
  else:
    task_type = "worker"
    task_index = FLAGS.task_index - 1
  return cluster, task_type, task_index

def dump_into_tf_config(cluster, task_type, task_index):
  os.environ['TF_CONFIG'] = json.dumps(
    {'cluster': cluster,
     'task': {'type': task_type, 'index': task_index}})

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  model_config = conformer.ConformerConfig.from_json_file(FLAGS.bert_config_file)

  config_name = FLAGS.bert_config_file.split("/")[-1]
  import os
  output_dir = os.path.join(FLAGS.buckets, FLAGS.output_dir)

  tf.gfile.MakeDirs(output_dir)
  if FLAGS.init_checkpoint:
    init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
  else:
    init_checkpoint = None
  sess_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)

  if FLAGS.do_distributed_training:
    if 'TF_CONFIG' in os.environ:
        del os.environ['TF_CONFIG']
    if "pai" in tf.__version__.lower() and FLAGS.distributed_mode == 'all_reduce':
      from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
      cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps('nccl')
      print(tf.__version__, "==tf version all reduce==")
      distribution = tf.contrib.distribute.MirroredStrategy(
                  num_gpus=FLAGS.num_gpus,
                  cross_tower_ops=cross_tower_ops,
                  all_dense=True,
                  iter_size=FLAGS.num_accumulated_batches)
      print("===distribution===", distribution)
      worker_count = None
      task_index = 0
      worker_gpus = FLAGS.num_gpus
      print(worker_gpus, "==worker_count==")
    elif 'pai' in tf.__version__.lower() and FLAGS.distributed_mode == 'collective_reduce':
      print(tf.__version__, "==tf version collective reduce==")
      cluster, task_type, task_index = make_distributed_info_without_evaluator()
      dump_into_tf_config(cluster, task_type, task_index)
      distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(
                  num_gpus_per_worker=FLAGS.num_gpus,
                  cross_tower_ops_type='horovod', # default, horovod
                  all_dense=True,
                  iter_size=FLAGS.num_accumulated_batches)

      worker_hosts = FLAGS.worker_hosts.split(",")
      worker_count = len(worker_hosts)
      worker_gpus = worker_count*FLAGS.num_gpus
      print(worker_gpus, "==worker_count==")

  if task_index == 0:
    import os
    with tf.gfile.GFile(os.path.join(FLAGS.buckets, FLAGS.output_dir, config_name), "w") as fwobj:
      fwobj.write(model_config.to_json_string()+"\n")

  global_batch_size = FLAGS.train_batch_size * worker_gpus * FLAGS.num_accumulated_batches
  local_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulated_batches

  run_config = tf.estimator.RunConfig(
                          keep_checkpoint_max=FLAGS.keep_checkpoint_max,
                          train_distribute=distribution, # tf 1.8
                          session_config=sess_config,
                          save_checkpoints_secs=None,
                          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                          log_step_count_steps=FLAGS.log_step_count_steps)

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

  train_examples = None
  num_warmup_steps = None

  train_examples = FLAGS.train_examples
  num_train_steps = int(
        FLAGS.train_examples / (global_batch_size) * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      model_config=model_config,
      init_checkpoint=init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)

  # if FLAGS.distributed_mode == "collective_reduce":
  #   output_dir = output_dir if task_index == 0 else None

  output_dir = output_dir

  estimator = tf.estimator.Estimator(
                model_fn=model_fn,
                model_dir=output_dir,
                config=run_config)

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
  feature_augmenter = augment_tf.Augmentation(featurizer_aug_config, 
                                            use_tf=True)

  train_input_fn = lambda: input_fn_builder(
        input_file=input_files,
        is_training=True,
        drop_remainder=True,
        audio_featurizer=audio_featurizer,
        feature_augmenter=feature_augmenter,
        max_duration=FLAGS.max_duration,
        samples_per_second=FLAGS.samples_per_second,
        transcript_seq_length=FLAGS.transcript_seq_length,
        batch_size=local_batch_size,
        use_tpu=FLAGS.use_tpu,
        worker_count=worker_count,
        task_index=task_index,
        distributed_mode=FLAGS.distributed_mode,
        num_predict=FLAGS.num_predict,
        mask_prob=FLAGS.mask_prob,
        min_tok=FLAGS.min_tok,
        max_tok=FLAGS.max_tok
        )

  if FLAGS.distributed_mode == 'collective_reduce':

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  elif FLAGS.distributed_mode == 'all_reduce':
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

if __name__ == "__main__":
  tf.app.run()

