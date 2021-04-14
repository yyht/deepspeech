from collections import OrderedDict
from pypinyin import pinyin, lazy_pinyin, Style
import tensorflow as tf
import csv

import opencc
cc = opencc.OpenCC('t2s')

transcript_dict = OrderedDict({})

with tf.gfile.Open("/data/albert/asr/test/trans_st_cmds.txt", "r") as csvfile:
  readCSV = csv.reader(csvfile, delimiter='\t')
  for index, content in enumerate(readCSV):
  if index == 0:
    key = content
    continue
  item_dict = OrderedDict(zip(key, content))
  transcript_dict[item_dict['UtteranceID']]= item_dict
  transcript_dict[item_dict['UtteranceID']]['src'] = 'st_cmds'
with tf.gfile.Open("/data/albert/asr/test/trans_aitangdata.txt", "r") as csvfile:
  readCSV = csv.reader(csvfile, delimiter='\t')
  for index, content in enumerate(readCSV):
  if index == 0:
    key = content
    continue
  item_dict = OrderedDict(zip(key, content))
  transcript_dict[item_dict['UtteranceID']]= item_dict
  transcript_dict[item_dict['UtteranceID']]['src'] = 'aitangdata'
with tf.gfile.Open("/data/albert/asr/test/trans_magic_data.txt", "r") as csvfile:
  readCSV = csv.reader(csvfile, delimiter='\t')
  for index, content in enumerate(readCSV):
  if index == 0:
    key = content
    continue
  item_dict = OrderedDict(zip(key, content))
  transcript_dict[item_dict['UtteranceID']]= item_dict
  transcript_dict[item_dict['UtteranceID']]['src'] = 'magic_data'
  
with tf.gfile.Open("/data/albert/asr/test/trans_aishell.txt", "r") as csvfile:
  readCSV = csv.reader(csvfile, delimiter='\t')
  for index, content in enumerate(readCSV):
  if index == 0:
    key = content
    continue
  item_dict = OrderedDict(zip(key, content))
  transcript_dict[item_dict['UtteranceID']]= item_dict
  transcript_dict[item_dict['UtteranceID']]['src'] = 'aishell'
  
from tokenizers import (ByteLevelBPETokenizer,
        CharBPETokenizer,
        SentencePieceBPETokenizer,
        BertWordPieceTokenizer)

bpe_tokenizer = BertWordPieceTokenizer(
        "/data/albert/asr/test/vocab.txt", 
         lowercase=True
        )

import re
import jieba_fast as jieba

CH_PUNCTUATION = u"['\\\\,\\!#$%&\'()*+-/:￥;<=>.?\\n@[\\]^▽_`{|}~'－＂＃＄％＆＇，：；＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。]"
ch_pattern = re.compile(u"[\u4e00-\u9fa5]+")
from collections import OrderedDict

# non_ch = u"[^0-9a-zA-Z一-龥㖏\s\t]+"

non_ch = u"[^一-龥㖏\s\t]+"

flag_count = 0

from collections import Counter

vocab = Counter()
pinyin_vocab = Counter()

fwobj = open("/data/albert/asr/test/invalid.txt", "w")

for key in transcript_dict:
  
  flag = False
  
  initial_utt = transcript_dict[key]['Transcription'].strip().lower()
  initial_utt = cc.convert(initial_utt)
  utt = re.sub("(\[fil\])", "", initial_utt)
  utt = re.sub("(\[spk\])", "", utt)
  find_=  re.findall(non_ch, utt)
  if len(find_):
    print(find_)
    fwobj.write("&".join(find_)+"\n")
  utt = re.sub(non_ch, "", utt)
  
  cutted_lst = list(jieba.cut(utt))
  
  token_lst = []
  token_pinyin_lst = []
  token_string = ""
  for item in cutted_lst:
#         if re.findall(ch_pattern, item):
#             token_string += item
#         else:
#             if item not in [' ', '']:
#                 bpe_token = bpe_tokenizer.encode(item).tokens[1:-1]
#                 token_string += " "
#                 token_string += " ".join(bpe_token)
#                 token_string += " "
#                 flag_count += 1
#                 flag = True
    if re.findall(u"[0-9a-z#]+", item):
      if item not in [' ', '']:
        bpe_token = bpe_tokenizer.encode(item).tokens[1:-1]
        token_string += " "
        token_string += " ".join(bpe_token)
        token_string += " "
        flag_count += 1
        flag = True
    else:
      token_string += item

  for item in token_string.split():
    if re.findall(u"[0-9a-z#]+", item):
#             bpe_token = bpe_tokenizer.encode(item).tokens[1:-1]
#             print(item, bpe_token, token_string, initial_utt)
      token_lst.append(item)
      token_pinyin_lst.append(item)
    else:
      token_lst.extend(item)
      pinyin_lst = lazy_pinyin(item, style=Style.TONE3, neutral_tone_with_five=True)
      token_pinyin_lst.extend(pinyin_lst)
      
  for item in token_pinyin_lst:
#         if item not in pinyin_vocab:
    if item == "#":
      print(token_pinyin_lst, initial_utt)
      fwobj.write("&".join(token_pinyin_lst)+"\n")
    pinyin_vocab[item] += 1
      
  for item in token_lst:
#             if item not in vocab:
    vocab[item] += 1
        
  transcript_dict[key]['token_transcription'] = " ".join(token_lst)
  transcript_dict[key]['pinyin_transcription'] = " ".join(token_pinyin_lst)
  key_lst = list(transcript_dict[key].keys())
#     if flag_count <= 100 and flag == True:
#         print(token_lst, token_pinyin_lst, token_string)
#     if flag_count > 10000:
#         break

fwobj.close()

pinyin_thresh = 5
char_thresh = 5

fwobj1 = open("/data/albert/asr/test/chinese_char5k_all_count.txt", "w")
with open("/data/albert/asr/test/chinese_char5k_all.txt", "w") as fwobj:
  fwobj.write(" "+"\n")
  for key in vocab:
    key_count = "\t".join([key, str(vocab[key])])
    if vocab[key] >= char_thresh:
      fwobj.write(key+"\n")
    fwobj1.write(key_count+"\n")
fwobj1.close()

fwobj1 = open("/data/albert/asr/test/chinese_syllable5k_all_count.txt", "w")
with open("/data/albert/asr/test/chinese_syllable5k_all.txt", "w") as fwobj:
  fwobj.write(" "+"\n")
  for key in pinyin_vocab:
    key_count = "\t".join([key, str(pinyin_vocab[key])])
    if pinyin_vocab[key] >= pinyin_thresh:
      fwobj.write(key+"\n")
    fwobj1.write(key_count+"\n")
fwobj1.close()
  
with tf.gfile.Open("/data/albert/asr/test/trans_st_cmds_v1.txt", "w") as fwobj:
  fwobj.write("\t".join([item for item in key_lst if item not in ['src']])+"\n")
  for utterance_id in transcript_dict:
    content = [transcript_dict[utterance_id][key] for key in key_lst if key not in  ['src']]
    if transcript_dict[utterance_id]['src'] == 'st_cmds' and 'token_transcription' in transcript_dict[utterance_id]:
      token_script = transcript_dict[utterance_id]['token_transcription'].split()
      pinyin_script = transcript_dict[utterance_id]['pinyin_transcription'].split()
      token_invliad_flag = False
      pinyin_invliad_flag = False
      for token in token_script:
        if vocab[token] < char_thresh:
          token_invliad_flag = True
          break
      for token in pinyin_script:
        if  pinyin_vocab[token] < pinyin_thresh:
          pinyin_invliad_flag = True
          break
      if pinyin_invliad_flag or token_invliad_flag:
        print(token_script)
        continue
          
      fwobj.write("\t".join(content)+"\n")
      
with tf.gfile.Open("/data/albert/asr/test/trans_aitangdata_v1.txt", "w") as fwobj:
  fwobj.write("\t".join([item for item in key_lst if item not in ['src']])+"\n")
  for utterance_id in transcript_dict:
    content = [transcript_dict[utterance_id][key] for key in key_lst if key not in  ['src']]
    if transcript_dict[utterance_id]['src'] == 'aitangdata' and 'token_transcription' in transcript_dict[utterance_id]:
      token_script = transcript_dict[utterance_id]['token_transcription'].split()
      pinyin_script = transcript_dict[utterance_id]['pinyin_transcription'].split()
      token_invliad_flag = False
      pinyin_invliad_flag = False
      for token in token_script:
        if vocab[token] < char_thresh:
          token_invliad_flag = True
          break
      for token in pinyin_script:
        if  pinyin_vocab[token] < pinyin_thresh:
          pinyin_invliad_flag = True
          break
      if pinyin_invliad_flag or token_invliad_flag:
        print(token_script)
        continue
      fwobj.write("\t".join(content)+"\n")
      
with tf.gfile.Open("/data/albert/asr/test/trans_magic_data_v1.txt", "w") as fwobj:
  fwobj.write("\t".join([item for item in key_lst if item not in ['src']])+"\n")
  for utterance_id in transcript_dict:
    content = [transcript_dict[utterance_id][key] for key in key_lst if key not in  ['src']]
    if transcript_dict[utterance_id]['src'] == 'magic_data' and 'token_transcription' in transcript_dict[utterance_id]:
      token_script = transcript_dict[utterance_id]['token_transcription'].split()
      pinyin_script = transcript_dict[utterance_id]['pinyin_transcription'].split()
      token_invliad_flag = False
      pinyin_invliad_flag = False
      for token in token_script:
        if vocab[token] < char_thresh:
          token_invliad_flag = True
          break
      for token in pinyin_script:
        if  pinyin_vocab[token] < pinyin_thresh:
          pinyin_invliad_flag = True
          break
      if pinyin_invliad_flag or token_invliad_flag:
        print(token_script)
        continue
      fwobj.write("\t".join(content)+"\n")
      
with tf.gfile.Open("/data/albert/asr/test/trans_aishell_v1.txt", "w") as fwobj:
  fwobj.write("\t".join([item for item in key_lst if item not in ['src']])+"\n")
  for utterance_id in transcript_dict:
    content = [transcript_dict[utterance_id][key] for key in key_lst if key not in  ['src']]
    if transcript_dict[utterance_id]['src'] == 'aishell' and 'token_transcription' in transcript_dict[utterance_id]:
      token_script = transcript_dict[utterance_id]['token_transcription'].split()
      pinyin_script = transcript_dict[utterance_id]['pinyin_transcription'].split()
      token_invliad_flag = False
      pinyin_invliad_flag = False
      for token in token_script:
        if vocab[token] < char_thresh:
          token_invliad_flag = True
          break
      for token in pinyin_script:
        if  pinyin_vocab[token] < pinyin_thresh:
          pinyin_invliad_flag = True
          break
      if pinyin_invliad_flag or token_invliad_flag:
        print(token_script)
        continue
      fwobj.write("\t".join(content)+"\n")