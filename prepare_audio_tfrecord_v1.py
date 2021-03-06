# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

import json, os
from collections import OrderedDict
import unicodedata
import sys, csv
from dataset import tokenization
import numpy as np
import re, time
from dataset import pai_write_tfrecord_with_noise_mixup

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
  "input_transcript_path", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "input_speaker_meta_path", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "noise_path", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "noise_meta_path", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "output_path", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "pinyin_vocab", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "gender_vocab", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "spk_vocab", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "dialect_vocab", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "char_vocab", "",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
  "sample_rate", 16000,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
  "target_sample_rate", 8000,
  "Input TF example files (can be a glob or comma separated).")

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

print(len(pinyin_dict['pinyin2id']), "==sum of pinyin2id==")

char_dict = {
  "char2id":{},
  "id2char":{}
}

with tf.gfile.Open(os.path.join(FLAGS.buckets, FLAGS.char_vocab), "r") as frobj:
  for index, line in enumerate(frobj):
    content = line.strip()
    content = tokenization.convert_to_unicode(content)
    char_dict['char2id'][content] = index
    char_dict['id2char'][index] = content

print(len(char_dict['char2id']), "==sum of char2id==")

gender_dict = {
  "gender2id":{},
  "id2gender":{}
}
with tf.gfile.Open(os.path.join(FLAGS.buckets, FLAGS.gender_vocab), "r") as frobj:
  for index, line in enumerate(frobj):
    content = line.strip()
    content = tokenization.convert_to_unicode(content)
    gender_dict['gender2id'][content] = index
    gender_dict['id2gender'][index] = content

print(len(gender_dict['gender2id']), "==sum of gender2id==")

spk_dict = {
  "spk2id":{},
  "id2spk":{}
}
with tf.gfile.Open(os.path.join(FLAGS.buckets, FLAGS.spk_vocab), "r") as frobj:
  for index, line in enumerate(frobj):
    content = line.strip()
    content = tokenization.convert_to_unicode(content)
    spk_dict['spk2id'][content] = index
    spk_dict['id2spk'][index] = content

print(len(spk_dict['spk2id']), "==sum of spk2id==")

dialect_dict = {
  "dialect2id":{},
  "id2dialect":{}
}
with tf.gfile.Open(os.path.join(FLAGS.buckets, FLAGS.dialect_vocab), "r") as frobj:
  for index, line in enumerate(frobj):
    content = line.strip()
    content = tokenization.convert_to_unicode(content)
    dialect_dict['dialect2id'][content] = index
    dialect_dict['id2dialect'][index] = content

print(len(dialect_dict['dialect2id']), "==sum of dialect2id==")

def tokenize(input_text, input_pinyin_text, max_length=81):
  input_text = tokenization.convert_to_unicode(input_text).lower()
  input_pinyin_text = tokenization.convert_to_unicode(input_pinyin_text).lower()

  input_tokens = input_text.split()
  input_pinyin_tokens = input_pinyin_text.split()
  input_pinyin_ids = []
  input_token_ids = []
  for item in input_pinyin_tokens:
    if item not in pinyin_dict['pinyin2id']:
      continue
    if pinyin_dict['pinyin2id'][item] == 0:
      continue
    input_pinyin_ids.append(pinyin_dict['pinyin2id'][item])
  for item in input_tokens:
    if item not in char_dict['char2id']:
      continue
    if char_dict['char2id'][item] == 0:
      continue
    input_token_ids.append(char_dict['char2id'][item])

  if len(input_pinyin_ids) > max_length or len(input_pinyin_ids) > max_length:
    print("===max length===", input_text)
    return "", ""

  if len(input_pinyin_ids) == 0 or len(input_token_ids) == 0:
    print("===zero length===", input_text)
    return "", ""

  input_token_ids += [0]*(max_length-len(input_token_ids))
  input_pinyin_ids += [0]*(max_length-len(input_pinyin_ids))
  
  if max(input_token_ids) <= len(char_dict['char2id'])-1 and max(input_pinyin_ids) <= len(pinyin_dict['pinyin2id'])-1:
    return input_token_ids, input_pinyin_ids
  else:
    return "", ""

num_workers = len(FLAGS.worker_hosts.split(","))
print(FLAGS.task_index, "====task index====", FLAGS.worker_hosts)

noise_meta_path = os.path.join(FLAGS.buckets, FLAGS.noise_meta_path)
print(noise_meta_path, "===noise_meta_path===")

noise_meta_lst = []
import csv
with tf.gfile.Open(noise_meta_path, "r") as csvfile:
  readCSV = csv.reader(csvfile, delimiter=',')
  for index, content in enumerate(readCSV):
    if index == 0:
      key = content
      continue
    item_dict = dict(zip(key, content))
    noise_meta_lst.append(item_dict)
    
print(noise_meta_lst)

from collections import OrderedDict

input_speaker_meta_path = os.path.join(FLAGS.buckets, FLAGS.input_speaker_meta_path)

print(input_speaker_meta_path, "===input_speaker_meta_path===")

input_speaker_meta_dict = OrderedDict({})
with tf.gfile.Open(input_speaker_meta_path, "r") as csvfile:
  readCSV = csv.reader(csvfile, delimiter='\t')
  for index, content in enumerate(readCSV):
    if index == 0:
      key = content
      continue
    item_dict = dict(zip(key, content))
    input_speaker_meta_dict[item_dict['SPKID']]= item_dict

import json

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

input_meta_path = os.path.join(FLAGS.buckets, FLAGS.input_meta_path)

print(input_meta_path, "===input_meta_path===")

input_meta_lst = []
with tf.gfile.Open(input_meta_path, "r") as f:
  for index, line in enumerate(f):
    # print(index, line, "==index==")
    content = line.strip().split("\t")
    file_name = content[0]
    file_path = content[1]
    input_meta_lst.append({
      "file_name":file_name,
      "file_path":file_path
      })
    
print("==input_meta_lst==", len(input_meta_lst))

total_batch = int(len(input_meta_lst)/num_workers)

start_index = FLAGS.task_index * total_batch

import random
# random.shuffle(input_meta_lst)

if FLAGS.task_index == num_workers - 1:
  end_index = len(input_meta_lst)
else:
  end_index = (FLAGS.task_index + 1) * total_batch
current_meta_lst = input_meta_lst[start_index:end_index]
random.shuffle(current_meta_lst)

tf_sess = tf.Session()

output_path = os.path.join(FLAGS.buckets, FLAGS.output_path, 'chinese_asr_{}.tfrecord'.format(FLAGS.task_index))
tfrecord_writer = tf.python_io.TFRecordWriter(output_path)

meta_output_path = os.path.join(FLAGS.buckets, FLAGS.output_path, 'chinese_asr_{}.meta'.format(FLAGS.task_index))
fwobj = tf.gfile.Open(meta_output_path, "w")

max_resample_shape = []

import gc
print(gc.get_threshold(), "==before gc count==")
gc.set_threshold(100000, 10, 10)
print(gc.get_threshold(), "==after gc count==")

import time
start_time = time.time()
feature_time = 0
write_time = 0

from tensorflow.python.ops import gen_audio_ops as contrib_audio

class TFAPI(object):
  def __init__(self, sample_rate):
    self.sample_rate = sample_rate
    self.audio_array = tf.placeholder(tf.float32, [None])
    self.audio_tensor = tf.expand_dims(self.audio_array, axis=-1)
    self.audio_string_tensor = contrib_audio.encode_wav(self.audio_tensor, sample_rate=sample_rate)

  def run(self, audio_array):
    audio_string = tf_sess.run(self.audio_string_tensor, 
            feed_dict={
        self.audio_array:audio_array
    })
    return audio_string

tf_string_api = TFAPI(FLAGS.target_sample_rate)

def get_length(token_id_list):
  valid_item = [item for item in token_id_list if item != 0]
  return len(valid_item)

for index, item_dict in enumerate(current_meta_lst):
  item_name = item_dict['file_name']
  item_path = item_dict['file_path']

  utterance_id = item_name
  if utterance_id not in transcript_dict:
    print(utterance_id, item_path)
    continue
  token_transcript = transcript_dict[utterance_id]['token_transcription']
  token_transcript = tokenization.convert_to_unicode(token_transcript).lower()

  if not transcript_dict[utterance_id].get("pinyin_transcription", {}):
    continue
  pinyin_transcript = transcript_dict[utterance_id]['pinyin_transcription']
  pinyin_transcript = tokenization.convert_to_unicode(pinyin_transcript).lower()

  speaker_string = transcript_dict[item_name]['SpeakerID']
  speaker_id = 0 #spk_dict["spk2id"][speaker_string]
  
  gender_id = 0 #gender_dict["gender2id"][input_speaker_meta_dict[speaker_string]['Gender'].lower()]
  dialect_id = 0 #dialect_dict["dialect2id"][input_speaker_meta_dict[speaker_string]['Dialect']]
    
  clean_path = os.path.join(FLAGS.buckets, FLAGS.input_path, item_path)
  
  noise_id = np.random.choice(np.arange(0, len(noise_meta_lst)), p=[0.05, 0.05, 0.00, 0.3, 0.3, 0.3])

  noise_sample_name = noise_meta_lst[noise_id]['fname']
  noise_sample_path = os.path.join(FLAGS.buckets, FLAGS.noise_path, noise_sample_name)

  [transcript_id, 
  pinyin_id] = tokenize(token_transcript, 
                        pinyin_transcript, 
                        max_length=81)

  if not transcript_id or not pinyin_id:
    item_dict['transcript'] = token_transcript
    item_dict['clean_path'] = clean_path
    fwobj.write(
      json.dumps(
          item_dict,
          ensure_ascii=False
        )+"\n"
      )
    tf.logging.info("==invalid-meta of transcript_id==")
    tf.logging.info(item_dict)
    continue

  if not tf.gfile.Exists(clean_path):
    item_dict['transcript'] = token_transcript
    item_dict['clean_path'] = clean_path
    fwobj.write(
      json.dumps(
          item_dict,
          ensure_ascii=False
        )+"\n"
      )
    tf.logging.info("==clean_path not exists==")
    tf.logging.info(item_dict)
    continue
  
  valid_flag = False
  feat_tmp_time = time.time()
  if get_length(transcript_id) <= 2 and get_length(pinyin_id) <= 2:
    continue
  if get_length(transcript_id) != get_length(pinyin_id):
    continue
  
  tf.logging.info("** check length of transcript_id and pinyin_id")
  tf.logging.info(get_length(transcript_id))
  tf.logging.info(get_length(pinyin_id))

  for t in range(10):
    try:
      [example, ori_shape, resample_shape] = pai_write_tfrecord_with_noise_mixup.noise_synthesizer(
                      clean_path, 
                      noise_sample_path,
                      speaker_id,
                      noise_id,
                      gender_id,
                      dialect_id,
                      transcript_id,
                      tf_sess,
                      sample_rate=FLAGS.sample_rate,
                      target_sample_rate=FLAGS.target_sample_rate,
                      pinyin_id=pinyin_id,
                      tf_string_api=tf_string_api)
      valid_flag = True
      feature_time += (time.time() - feat_tmp_time)
      max_resample_shape.append(resample_shape)
      break
    except:
      time.sleep(0.01)
      continue

  valid_write_flag = False
  write_tmp_time = time.time()
  if valid_flag:
    time.sleep(0.005)
    tfrecord_writer.write(example.SerializeToString())
    valid_write_flag = True
    write_time += (time.time() - write_tmp_time)
    
  if not valid_flag or not valid_write_flag:
    item_dict['transcript'] = token_transcript
    item_dict['clean_path'] = clean_path
    fwobj.write(
      json.dumps(
          item_dict,
          ensure_ascii=False
        )+"\n"
      )
    if not valid_flag:
      tf.logging.info("==invalid-process of clean_path==")
      tf.logging.info(item_dict)

    if not valid_write_flag:
      tf.logging.info("==invalid-write process of clean_path==")
      tf.logging.info(item_dict)
    continue
  fwobj.write(
      json.dumps(
          {
            "ori_shape":ori_shape,
            "resample_shape":resample_shape
          }
        )+"\n"
    )

  if np.mod(index, 100) == 0 and index != 0:
    print(ori_shape, "==mode 100==", resample_shape, item_dict)
    print(time.time()-start_time, write_time, feature_time)
    print(transcript_id, pinyin_id)
    start_time = time.time()
    write_time = 0
    feature_time = 0
  if np.mod(index, 1000) == 0:
    gc.collect()
    print("==gc collect==")
  # if index == 10:
  #   break

tfrecord_writer.close()
fwobj.close()

print(max(max_resample_shape), "==max resample-shape==")