
from dataset.data_generator import DataGenerator
from augment_io import augment_np
from pypinyin import pinyin, lazy_pinyin, Style
from dataset import tokenization
import numpy as np
import tensorflow as tf

class AMDataGenerator(object):
  def __init__(self, config, data_path_dict):
    self.config = config
    self.load_meta = False
    if data_path_dict:
      self.load_data_meta(data_path_dict)
      self.load_meta = True

  def load_transcript_meta(self, meta_path):
    input_meta_lst = []
    with tf.gfile.Open(meta_path, "r") as f:
      for line in f:
        content = line.strip().split("\t")
        file_name = content[0]
        file_path = content[1]
        input_meta_lst.append({
          "file_name":file_name,
          "file_path":file_path
          })
    return input_meta_lst

  def load_transcript(self, transcript_path):
    transcript_dict = {}
    with tf.gfile.Open(transcript_path, "r") as csvfile:
      readCSV = csv.reader(csvfile, delimiter='\t')
      for index, content in enumerate(readCSV):
        if index == 0:
          key = content
          continue
        item_dict = dict(zip(key, content))
        transcript_dict[item_dict['UtteranceID']]= item_dict 
    return transcript_dict

  def load_data_meta(self, data_path_dict):
    self.meta_dict = {}
    for key in data_path_dict:
      data_dict = data_path_dict[key]
      meta_lst = self.load_transcript_meta(data_dict['meta_path'])
      transcript_dict = self.load_transcript(data_dict['transcript_path'])
      random.shuffle(meta_lst)
      self.meta_dict[key] = {
        "data_path": data_dict['data_path'],
        "meta": meta_lst,
        "transcript": transcript_dict,
        "meta_num": len(meta_lst)
      }

  def load_char(self, char_path):
    self.char_dict = {
      "char2id":{},
      "id2char":{}
    }
    with tf.gfile.Open(char_path, "r") as frobj:
      for index, line in enumerate(frobj):
        content = line.strip()
        content = tokenization.convert_to_unicode(content)
        self.char_dict['char2id'][content] = index
        self.char_dict['id2char'][index] = content
    tf.logging.info("** sum of char2id **")
    tf.logging.info(len(self.char_dict['char2id']))

  def load_pinyin(self, pinyin_path):
    self.pinyin_dict = {
      "pinyin2id":{},
      "id2pinyin":{}
    }
    with tf.gfile.Open(pinyin_path, "r") as frobj:
      for index, line in enumerate(frobj):
        content = line.strip()
        content = tokenization.convert_to_unicode(content)
        self.pinyin_dict['pinyin2id'][content] = index
        self.pinyin_dict['id2pinyin'][index] = content
    tf.logging.info("** sum of pinyin2id **")
    tf.logging.info(len(self.pinyin_dict['pinyin2id']))

  def load_gender(self, gender_path):
    self.gender_dict = {
      "gender2id":{},
      "id2gender":{}
    }
    with tf.gfile.Open(gender_path, "r") as frobj:
      for index, line in enumerate(frobj):
        content = line.strip()
        content = tokenization.convert_to_unicode(content)
        self.gender_dict['gender2id'][content] = index
        self.gender_dict['id2gender'][index] = content
    tf.logging.info("** sum of gender2id **")
    tf.logging.info(len(self.gender_dict['gender2id']))

  def load_spk(self, spk_path):
    self.spk_dict = {
      "spk2id":{},
      "id2spk":{}
    }
    with tf.gfile.Open(spk_path, "r") as frobj:
      for index, line in enumerate(frobj):
        content = line.strip()
        content = tokenization.convert_to_unicode(content)
        self.spk_dict['spk2id'][content] = index
        self.spk_dict['id2spk'][index] = content

    tf.logging.info("** sum of spk2id **")
    tf.logging.info(len(self.spk_dict['spk2id']))

  def load_dialect(self, dialect_path):
    self.dialect_dict = {
      "dialect2id":{},
      "id2dialect":{}
    }
    with tf.gfile.Open(dialect_path, "r") as frobj:
      for index, line in enumerate(frobj):
        content = line.strip()
        content = tokenization.convert_to_unicode(content)
        self.dialect_dict['dialect2id'][content] = index
        self.dialect_dict['id2dialect'][index] = content

    tf.logging.info("** sum of dialect2id **")
    tf.logging.info(len(self.dialect_dict['dialect2id']))

  

  def 

  def preprocess(self, )

  def iteration(self, data_path_dict):
    
    key_lst = list(self.meta_dict.keys())
    selected_key = np.random.choice(key_lst)

    meta_num = self.meta_dict[selected_key]['meta_num']
    meta_dict = self.meta_dict[selected_key]['meta'][np.random.choice(meta_num)]
    





      


