# -*- coding: utf-8 -*-

import envoy # pip install envoy
import os
import codecs
import collections
import re
import unicodedata
import six
from xpinyin import Pinyin

p = Pinyin()

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


# root_path = '/data/albert.xht/aidatatang_200zh/corpus'
# output_root_path = '/data/albert.xht/aidatatang_200zh_raw'

# for partition_path in ['train', 'test', 'dev']:
#   father_path = os.path.join(root_path, partition_path)
#   output_father_path = os.path.join(output_root_path, partition_path)
#   if not os.path.exists(output_father_path):
#     os.mkdir(output_father_path)
#   for fi in os.listdir(father_path):
#     temp_file = os.path.join(father_path, fi)
#     print(temp_file, "==temp_file==")
#     if (temp_file.endswith("tar.gz")):
#       envoy.run("tar xzf %s -C %s" % (temp_file, output_father_path))

spk_info = {}
all_scp = {}
trans = {}

mapping = {
    u"女":"female",
    u"男":"male"

}

root_path = '/data/albert.xht/st_cmds/ST-CMDS/ST-CMDS-20170001_1-OS/'

# /data/albert.xht/ST-CMDS-20170001_1-OS
for fi in os.listdir(root_path):
  fi_path = os.path.join(root_path, fi)
  
  if (fi_path.endswith("metadata")):
    tmp_dict = {}

    with open(fi_path) as frobj:
      for line in frobj:
        line = convert_to_unicode(line)
        content = line.strip().split()
        if len(content) >= 2:
          tmp_dict[content[0]] = "".join(content[1])
    print(tmp_dict, fi_path)

    if 'SCD' in tmp_dict:
      if tmp_dict['SCD'] not in spk_info:
        spk_info[tmp_dict['SCD']] = {

        }
        spk_info[tmp_dict['SCD']]['SPKID'] = tmp_dict['SCD']
        spk_info[tmp_dict['SCD']]['Gender'] = mapping[tmp_dict.get('SEX', u'男')]
        spk_info[tmp_dict['SCD']]['Age'] = tmp_dict['AGE']
        spk_info[tmp_dict['SCD']]['Dialect'] = p.get_pinyin(tmp_dict.get('BIR', u"北京"), ' ')
    
    if "DIR" in tmp_dict:
      utterance_id = fi_path.split("/")[-1].split(".")[0]
      utterance_id_path = utterance_id+".wav"
      print(utterance_id, "==utt==", utterance_id_path)
      all_scp[utterance_id+".wav"] = utterance_id_path
    
      label_path = os.path.join(root_path, utterance_id+".txt")
      print(label_path, "==label_path==")
      with open(label_path) as frobj:
        for line in frobj:
          content = line.strip()

      trans[utterance_id+".wav"] = {
        "UtteranceID":utterance_id+".wav",
        "SpeakerID":tmp_dict['SCD'],
        "Transcription":content
      }

root_path = "/data/albert.xht/st_cmds"

with open(os.path.join(root_path, 'metadata', "trans.txt"), "w") as fwobj:
  fwobj.write("\t".join(['UtteranceID', 'SpeakerID', 'Transcription'])+"\n")
  for utterance_id in trans:
    fwobj.write("\t".join([trans[utterance_id]['UtteranceID'], 
                          trans[utterance_id]['SpeakerID'], 
                          trans[utterance_id]['Transcription']])+"\n")

with open(os.path.join(root_path, 'metadata', "spk_info.txt"), "w") as fwobj:
  fwobj.write("\t".join(['SPKID', 'Age', 'Gender', "Dialect"])+"\n")
  for spk_id in spk_info:
    tmp_dict = spk_info[spk_id]
    fwobj.write("\t".join([tmp_dict['SPKID'], 
                          tmp_dict['Age'], 
                          tmp_dict['Gender'],
                          tmp_dict['Dialect']])+"\n")

with open(os.path.join(root_path, 'metadata', "all.scp"), "w") as fwobj:
  for utterance_id in all_scp:
    print(utterance_id, all_scp[utterance_id],"\t".join([utterance_id, all_scp[utterance_id]]))
    fwobj.write("\t".join([utterance_id, all_scp[utterance_id]])+"\n")
