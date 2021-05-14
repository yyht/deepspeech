# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

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

from inference.saved_model_inference import SavedModelInfer
import os

from audio_io import read_audio
from audio_io import utils
import random, librosa, os
from audio_io import audio_featurizer_tf

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class TFFeat(object):
  def __init__(self, config):
    self.config = config
    self.graph = tf.Graph()

  def load_sess(self):

    with self.graph.as_default():

      self.audio_featurizer = audio_featurizer_tf.TFSpeechFeaturizer(speech_config)
      self.max_feature_length = self.audio_featurizer.get_length_from_duration(20)
      self.audio_featurizer.update_length(self.max_feature_length)
    
      self.audio_tensor =  tf.placeholder(tf.float32, [None], name='audio')
      self.audio_feature = self.audio_featurizer.tf_extract(self.audio_tensor)

      feature_shape = shape_list(self.audio_feature)
      self.feature_seq_length = tf.cast(feature_shape[0], dtype=tf.int32)
      self.audio_feature = tf.pad(self.audio_feature, [
            [0, self.max_feature_length-self.feature_seq_length],
            [0,0],
            [0,0]])
      self.sess = tf.Session()

  def feat_sess(self, input_audio):
    with self.graph.as_default():
        [audio_feature, 
        feature_seq_length] = self.sess.run(
          [
            self.audio_feature,
            self.feature_seq_length
          ],
          feed_dict={
            self.audio_tensor:input_audio
          }
          )
    return [audio_feature, 
            feature_seq_length]

from dataset import tokenization
from pypinyin import pinyin, lazy_pinyin, Style
pinyin_dict = {
  "pinyin2id":{},
  "id2pinyin":{}
}
with tf.gfile.Open(os.path.join(FLAGS.buckets, FLAGS.pinyin_vocab), "r") as frobj:
  for index, line in enumerate(frobj):
    content = line.strip()
    content = tokenization.convert_to_unicode(content)
    pinyin_dict['pinyin2id'][content] = index
    pinyin_dict['id2pinyin'][index] = content

tf.logging.info("** succeeded in loading pinyin dict **")

with tf.gfile.Open(FLAGS.audio_featurizer_config_path, "r") as frobj:
  audio_featurizer_config = json.load(frobj)

feat_api = TFFeat(audio_featurizer_config)
feat_api.load_sess()

model_config = {
  "model":os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
}

model_api = SavedModelInfer(model_config)

num_workers = len(FLAGS.worker_hosts.split(","))
print(FLAGS.task_index, "====task index====", FLAGS.worker_hosts)

input_meta_path = os.path.join(FLAGS.buckets, FLAGS.input_meta_path)

print(input_meta_path, "===input_meta_path===")

input_meta_lst = []
with tf.gfile.Open(input_meta_path, "r") as f:
  for line in f:
    content = line.strip().split("\t")
    file_name = content[0]
    file_path = content[1]
    input_meta_lst.append({
      "file_name":file_name,
      "file_path":file_path
      })

total_batch = int(len(input_meta_lst)/num_workers)
start_index = FLAGS.task_index * total_batch

if FLAGS.task_index == num_workers - 1:
  end_index = len(input_meta_lst)
else:
  end_index = (FLAGS.task_index + 1) * total_batch
current_meta_lst = input_meta_lst[start_index:end_index]
random.shuffle(current_meta_lst)

transcript_path = os.path.join(FLAGS.buckets, FLAGS.input_transcript_path)

print(transcript_path, "===transcript_path===")

transcript_dict = {}
with tf.gfile.Open(transcript_path, "r") as csvfile:
  readCSV = csv.reader(csvfile, delimiter='\t')
  for index, content in enumerate(readCSV):
    if index == 0:
      key = content
      continue
    item_dict = dict(zip(key, content))
    transcript_dict[item_dict['UtteranceID']]= item_dict 

output_path = os.path.join(FLAGS.buckets, FLAGS.output_path, 'am_{}.json'.format(FLAGS.task_index))
fwobj = tf.gfile.Open(output_path, "w")

for index, item_dict in enumerate(current_meta_lst):
  item_name = item_dict['file_name']
  item_path = item_dict['file_path']

  data_path = os.path.join(FLAGS.buckets, FLAGS.input_path, item_path)  
  wave = read_audio.read_raw_audio(data_path, sample_rate=FLAGS.sample_rate)

  if FLAGS.target_sample_rate != FLAGS.sample_rate:
    wave = librosa.resample(wave, FLAGS.sample_rate, FLAGS.target_sample_rate)
  
  [audio_feature, 
  feature_seq_length] = feat_api.feat_sess(wave)

  am_resp = model_api.infer(
        {
            "audio_feature":[audio_feature],
            "feature_seq_length":[feature_seq_length]
        }
  )

  resp_lst = model_api.ctc_beam_decode(am_resp, pinyin_dict['id2pinyin'])

  utterance_id = item_name
  pinyin_transcript = transcript_dict[utterance_id]['pinyin_transcription']
  pinyin_transcript = tokenization.convert_to_unicode(pinyin_transcript).lower()

  fwobj.write(
    json.dumps({
      "hyp":resp_lst,
      "ref":pinyin_transcript
      })+"\n"
    )

fwobj.close()
