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

flags.DEFINE_integer(
  "sample_rate", 16000,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
  "target_sample_rate", 8000,
  "Input TF example files (can be a glob or comma separated).")

# from tokenizers import (ByteLevelBPETokenizer,
#               CharBPETokenizer,
#               SentencePieceBPETokenizer,
#               BertWordPieceTokenizer)
# 
# bpe_tokenizer = BertWordPieceTokenizer(
#               "./deepspeech/data/chinese_vocab/alphabet4k.txt", 
#                lowercase=True
#               )
# bpe_tokenizer.enable_padding('right', 
#                  length=96)
# bpe_tokenizer.enable_truncation(max_length=96)


tokenizer = tokenization.FullTokenizer(
            "./deepspeech/data/chinese_vocab/alphabet4k.txt", 
            do_lower_case=True
            )

def tokenize(input_text, max_length=96):
  input_text = tokenization.convert_to_unicode(input_text).lower()
  input_text = re.sub("[fil]", "", input_text)
  input_text = re.sub("[spk]", "", input_text)
  ori_text = input_text
  input_text = re.sub(CH_PUNCTUATION, "", input_text)
  utt_lst = re.findall(ch_pattern, input_text)
  utts = []
  for item in utt_lst:
    utts.extend(item)
  utts = " ".join(utts)

  input_tokens = tokenizer.tokenize(utts)
  try:
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    if len(input_ids) < max_length:
      input_ids += [0]*(max_length-len(input_ids))
    assert len(input_ids) == max_length
    return input_ids
  except:
    print("==not in vocab==", input_tokens, input_text)
    return ''

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

meta_dict = OrderedDict({})
for spk_id in input_speaker_meta_dict:
  item = input_speaker_meta_dict[spk_id]
  for key in item:
    if key in meta_dict:
      meta_dict[key].append(item[key])
    else:
      meta_dict[key] = [item[key]]
    
speaker_id_dict = OrderedDict({})
speaker_ini_id = 0
for item in meta_dict['SPKID']:
  if item in speaker_id_dict:
    continue
  speaker_id_dict[item] = speaker_ini_id
  speaker_ini_id += 1

gender_id_dict = OrderedDict({})
gender_ini_id = 0
for item in meta_dict['Gender']:
  if item in gender_id_dict:
    continue
  gender_id_dict[item] = gender_ini_id
  gender_ini_id += 1

dialect_id_dict = OrderedDict({})
dialect_ini_id = 0
for item in meta_dict['Dialect']:
  if item in dialect_id_dict:
    continue
  dialect_id_dict[item] = dialect_ini_id
  dialect_ini_id += 1

import json
if FLAGS.task_index == 0:
  meta_json = os.path.join(FLAGS.buckets, FLAGS.output_path, "meta.json")
  with tf.gfile.Open(meta_json, "w") as fwobj:
    json.dump(
       {
        "speaker_id":speaker_id_dict,
        "gender_id":gender_id_dict,
        "dialect_id":dialect_id_dict
      },
      fwobj
      )

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

for index, item_dict in enumerate(current_meta_lst):
  item_name = item_dict['file_name']
  item_path = item_dict['file_path']

  item_path_true = "/".join(item_path.split("/")[1:])
  utterance_id = item_path.split("/")[-1]
  transcript = transcript_dict[utterance_id]['Transcription']
  transcript = tokenization.convert_to_unicode(transcript).lower()

  speaker_string = item_path.split("/")[2]
  speaker_id = speaker_id_dict[speaker_string]
  
  gender_id = gender_id_dict[input_speaker_meta_dict[speaker_string]['Gender']]
  dialect_id = dialect_id_dict[input_speaker_meta_dict[speaker_string]['Dialect']]
    
  clean_path = os.path.join(FLAGS.buckets, FLAGS.input_path, item_path_true)
  
  noise_id = np.random.choice(np.arange(0, len(noise_meta_lst)), p=[0.8, 0.1, 0.1])

  noise_sample_name = noise_meta_lst[noise_id]['fname']
  noise_sample_path = os.path.join(FLAGS.buckets, FLAGS.noise_path, noise_sample_name)

  transcript_id = tokenize(transcript)
  if not transcript_id:
    item_dict['transcript'] = transcript
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
    item_dict['transcript'] = transcript
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
                      target_sample_rate=FLAGS.target_sample_rate)
      valid_flag = True
      break
    except:
      time.sleep(0.01)
      continue

  if not valid_flag:
    item_dict['transcript'] = transcript
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

  # if index == 10:
  #   break

tfrecord_writer.close()
fwobj.close()