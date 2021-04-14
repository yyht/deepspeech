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
    input_pinyin_ids.append(pinyin_dict['pinyin2id'][item])
  for item in input_tokens:
    input_token_ids.append(char_dict['char2id'][item])
  
  if len(input_pinyin_ids) > max_length or len(input_pinyin_ids) > max_length:
    return "", ""

  input_token_ids += [0]*(max_length-len(input_token_ids))
  input_pinyin_ids += [0]*(max_length-len(input_pinyin_ids))
  return input_token_ids, input_pinyin_ids

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
  for line in f:
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

final_output_path = os.path.join(FLAGS.buckets, FLAGS.output_path)

try:
  filenames = tf.gfile.ListDirectory(final_output_path)
except tf.errors.NotFoundError:
  filenames = []
  tf.logging.info("`checkpoint_dir` does not exist yet...")

left_meta_lst = []
import json
for filename in filenames:
  if filename.endswith(".meta"):
    meta_path = os.path.join(final_output_path, filename)
    with tf.gfile.Open(meta_path) as frobj:
      for line in frobj:
        content = tokenization.convert_to_unicode(line.strip())
        content = json.loads(content)
        if 'file_name' in content:
          left_meta_lst.append(content)

print("==total left meta list==", len(left_meta_lst))

if FLAGS.task_index == num_workers - 1:
  end_index = len(left_meta_lst)
else:
  end_index = (FLAGS.task_index + 1) * total_batch
current_meta_lst = left_meta_lst[start_index:end_index]
random.shuffle(current_meta_lst)

tf_sess = tf.Session()

output_path = os.path.join(FLAGS.buckets, FLAGS.output_path, 'chinese_asr_last_{}.tfrecord'.format(FLAGS.task_index))
tfrecord_writer = tf.python_io.TFRecordWriter(output_path)

meta_output_path = os.path.join(FLAGS.buckets, FLAGS.output_path, 'chinese_asr_last_{}.meta'.format(FLAGS.task_index))
fwobj = tf.gfile.Open(meta_output_path, "w")

max_resample_shape = []

import gc
print(gc.get_threshold(), "==before gc count==")
gc.set_threshold(100000, 10, 10)
print(gc.get_threshold(), "==after gc count==")

for index, item_dict in enumerate(current_meta_lst):
  item_name = item_dict['file_name']
  item_path = item_dict['file_path']

  utterance_id = item_name
  if utterance_id not in transcript_dict:
    print(utterance_id, item_path)
    continue
  token_transcript = transcript_dict[utterance_id]['token_transcription']
  token_transcript = tokenization.convert_to_unicode(token_transcript).lower()

  pinyin_transcript = transcript_dict[utterance_id]['pinyin_transcription']
  pinyin_transcript = tokenization.convert_to_unicode(pinyin_transcript).lower()

  speaker_string = transcript_dict[item_name]['SpeakerID']
  speaker_id = spk_dict["spk2id"][speaker_string]
  
  gender_id = gender_dict["gender2id"][input_speaker_meta_dict[speaker_string]['Gender'].lower()]
  dialect_id = dialect_dict["dialect2id"][input_speaker_meta_dict[speaker_string]['Dialect']]
    
  clean_path = item_path
  
  noise_id = np.random.choice(np.arange(0, len(noise_meta_lst)), p=[0.8, 0.1, 0.1])

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
    print(item_dict, "==invalid-meta of transcript_id==")
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
    print(item_dict, "==invalid-meta of clean_path==")
    continue
  
  valid_flag = False
  for t in range(10):
    try:
      [tfrecord_writer, ori_shape, resample_shape] = pai_write_tfrecord_with_noise_mixup.noise_synthesizer(
                      clean_path, 
                      noise_sample_path,
                      speaker_id,
                      noise_id,
                      gender_id,
                      dialect_id,
                      transcript_id,
                      tfrecord_writer,
                      tf_sess,
                      sample_rate=FLAGS.sample_rate,
                      target_sample_rate=FLAGS.target_sample_rate,
                      pinyin_id=pinyin_id)
      valid_flag = True
      max_resample_shape.append(resample_shape)
      break
    except:
      time.sleep(0.1)
      continue

  if not valid_flag:
    item_dict['transcript'] = token_transcript
    item_dict['clean_path'] = clean_path
    fwobj.write(
      json.dumps(
          item_dict,
          ensure_ascii=False
        )+"\n"
      )
    print(item_dict, "==invalid-process of clean_path==")
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
  if np.mod(index, 1000) == 0:
    gc.collect()
    print("==gc collect==")
  # if index == 10:
  #   break

tfrecord_writer.close()
fwobj.close()

print(max(max_resample_shape), "==max resample-shape==")