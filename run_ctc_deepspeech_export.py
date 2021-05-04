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
from model import deepspeech 
from optimizer.optimizer_utils import create_optimizer, create_adam_optimizer
import tensorflow as tf
from audio_io import audio_featurizer_tf, read_audio
from augment_io import augment_tf
from loss import ctc_loss
import json
from audio_io import utils as audio_utils
from loss import ctc_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import sparse_tensor

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
flags.DEFINE_integer("beam_width", 100,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("top_paths", 100,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("blank_index", 100,
                     "How many steps to make in each estimator call.")

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

from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import sparse_tensor
def _get_dim(tensor, i):
  """Get value of tensor shape[i] preferring static value if available."""
  return tensor_shape.dimension_value(
      tensor.shape[i]) or array_ops.shape(tensor)[i]

def create_model(model_config, 
                ipnut_features,
                is_training,
                beam_width=100,
                top_paths=100,
                input_length=None,
                blank_index=0):
  """Creates a classification model."""
  model = deepspeech.DeepSpeech(
      config=model_config,
      sequences=ipnut_features, 
      is_training=is_training,
      input_length=input_length)

  logits = model.get_logits()

  # [batch_size, time-steps, vocab_size]
  input_length = tf.identity(input_length)
  
  reduction_factor = model.get_conv_reduction_factor()
  reduced_length = audio_utils.get_reduced_length(input_length, reduction_factor)

  # from [batch_size, reduced_length, dims]
  # to [reduced_length, batch_size, dims]
  decoded_logits = tf.transpose(logits, [1,0,2])
  
  if blank_index < 0:
    blank_index += _get_dim(decoded_logits, 2)

  if blank_index != _get_dim(decoded_logits, 2) - 1:
    decoded_logits = array_ops.concat([
        decoded_logits[:, :, :blank_index],
        decoded_logits[:, :, blank_index + 1:],
        decoded_logits[:, :, blank_index:blank_index + 1],
    ],
                              axis=2)
    tf.logging.info("****** modify blank index **")
    tf.logging.info(blank_index)
    tf.logging.info( _get_dim(decoded_logits, 2) - 1)

  (decoded_path, 
    log_probability) = tf.nn.ctc_beam_search_decoder(
    decoded_logits, reduced_length, 
    beam_width=beam_width, 
    top_paths=top_paths, 
    merge_repeated=True
  )

  decoded = tf.to_int32(decoded_path[0])
  decoded_path = tf.sparse_tensor_to_dense(decoded)
  decoded_path += 1

  return (decoded_path, log_probability, logits)

def model_fn_builder(model_config, 
                init_checkpoint,
                audio_featurizer, 
                beam_width=100,
                top_paths=100,
                blank_index=0):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    # with tf.device("/CPU:0"):
    #   audio = read_audio.tf_read_raw_audio(features['raw_audio'], 
    #                     samples_per_second=FLAGS.samples_per_second,
    #                       use_tpu=False)
    #   audio_feature = audio_featurizer.tf_extract(audio)
    #   # [1, T, V, 1]
    #   audio_feature = tf.expand_dims(audio_feature, axis=0)

    #   feature_shape = shape_list(audio_feature)
    #   feature_seq_length = tf.cast(feature_shape[1], dtype=tf.int32)
    #   feature_seq_length = tf.expand_dims(feature_seq_length, axis=0)
    
    audio_feature = features['audio_feature']
    feature_seq_length = features['feature_seq_length']
    is_training = False

    (decoded_path, 
    log_probability, 
    logits) = create_model(model_config, 
                audio_feature,
                is_training,
                beam_width=beam_width,
                top_paths=top_paths,
                input_length=feature_seq_length,
                blank_index=blank_index)

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

    output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"decoded_path": decoded_path,
                      "log_probability": log_probability,
                      "logits":logits},
          export_outputs={
              "output":tf.estimator.export.PredictOutput(
                          {"decoded_path": decoded_path,
                          "log_probability": log_probability,
                          "logits":logits}
                      )
          })
    return output_spec

  return model_fn

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  model_config = deepspeech.DeepSpeechConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.blank_index != 0:
    model_config.__dict__['vocab_size'] += 1
    tf.logging.info("** blank_index is added to the vocab-size")
    tf.logging.info(model_config.__dict__['vocab_size'])

  config_name = FLAGS.bert_config_file.split("/")[-1]
  import os
  output_dir = os.path.join(FLAGS.buckets, FLAGS.output_dir)

  if FLAGS.init_checkpoint:
    init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
  else:
    init_checkpoint = None
  sess_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)

  audio_featurizer_config_path = os.path.join(FLAGS.buckets, FLAGS.audio_featurizer_config_path)
  with tf.gfile.Open(audio_featurizer_config_path, "r") as frobj:
    audio_featurizer_config = json.load(frobj)

  audio_featurizer = audio_featurizer_tf.TFSpeechFeaturizer(audio_featurizer_config)
  max_feature_length = audio_featurizer.get_length_from_duration(FLAGS.max_duration)
  audio_featurizer.update_length(max_feature_length)

  # receiver_features = {
  #   "raw_audio":tf.placeholder(tf.string, name='raw_audio')
  # }

  receiver_features = {
    "audio_feature":tf.placeholder(tf.float32, [None, 2001, 80, 1], name='audio_feature'),
    "feature_seq_length":tf.placeholder(tf.float32, [None], name='feature_seq_length'),
  }

  def serving_input_receiver_fn():
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(receiver_features)()
    return input_fn

  model_fn = model_fn_builder(
      model_config=model_config,
      init_checkpoint=init_checkpoint,
      audio_featurizer=audio_featurizer,
      beam_width=FLAGS.beam_width,
      top_paths=FLAGS.top_paths,
      blank_index=FLAGS.blank_index)

  estimator = tf.estimator.Estimator(
              model_fn=model_fn,
              model_dir=output_dir)

  import os
  input_export_dir = os.path.join(output_dir, 'export_dir')

  export_dir = estimator.export_savedmodel(input_export_dir, 
                      serving_input_receiver_fn,
                      checkpoint_path=init_checkpoint)

  print("===Succeeded in exporting saved model==={}".format(export_dir))

if __name__ == "__main__":
  tf.app.run()
