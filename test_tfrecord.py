
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from audio_io import read_audio
import json, os
from collections import OrderedDict
import unicodedata
import sys, csv
import numpy as np
import re, time
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""

CH_PUNCTUATION = u"[',\\\\n!#$%&\'()*+-/:;<=>.?@[\\]^_`{|}~'＂＃＄％＆＇，：；＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。]"
ch_pattern = re.compile(u"[\u4e00-\u9fa50-9a-zA-Z]+")

flags = tf.flags

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string("buckets", "", "oss buckets")
flags.DEFINE_string("tables", "", "oss buckets")
flags.DEFINE_integer("task_index", 0, "Worker or server index")
flags.DEFINE_string("worker_hosts", "", "worker hosts")

# id, features
flags.DEFINE_string(
  "outputs", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "input_path", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "input_meta_path", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "init_checkpoint", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "output_path", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "transcript_path", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "audio_featurizer_config_path", "audio_featurizer_config_path",
  "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
  "sample_rate", 16000,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
  "target_sample_rate", 8000,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
  "max_duration", 20,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "pinyin_vocab", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "input_transcript_path", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "data_type", "",
  "Input TF example files (can be a glob or comma separated).")

import tensorflow as tf

def _decode_record(record, name_to_features, **kargs):
  example = tf.parse_single_example(record, name_to_features)
  noise_audio = read_audio.tf_read_raw_audio(example['clean_audio_resample'], 
            samples_per_second=8000,
            use_tpu=True)
  example['noise_audio'] = noise_audio
  return example

name_to_features = {
  "clean_audio_resample": tf.FixedLenFeature([], tf.string),
  "noise_audio_resample": tf.FixedLenFeature([], tf.string),
  "speaker_id":tf.FixedLenFeature([], tf.int64),
  "transcript_id":tf.FixedLenFeature([81], tf.int64),
  "transcript_pinyin_id":tf.FixedLenFeature([81], tf.int64),
}

def train_input_fn(input_file, _parse_fn, name_to_features):

  dataset = tf.data.TFRecordDataset(input_file, buffer_size=4096)
  dataset = dataset.map(lambda x:_decode_record(x, name_to_features))
  dataset = dataset.batch(1)
  
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  return features

sess = tf.Session()

task_id = FLAGS.task_index
oss_path = os.path.join(FLAGS.buckets, FLAGS.input_path, "chinese_asr_{}.tfrecord".format(FLAGS.task_index))
print(oss_path, "==oss_path==")
features = train_input_fn(oss_path, '', name_to_features
         )

sess.run(tf.group(tf.global_variables_initializer(), tf.tables_initializer()))

ppp = []
count = 0
while True:
  # try:
  resp_features = sess.run(features)
  count += 1
  # except:
  #   break

print("==total data==", count)