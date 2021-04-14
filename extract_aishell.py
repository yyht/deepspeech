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


# root_path = '/data/albert.xht/aishell/data_aishell/wav'
# output_root_path = '/data/albert.xht/aishell/datasshell_raw'

# if not os.path.exists(output_root_path):
#   os.mkdir(output_root_path)

# for partition_path in os.listdir(root_path):
#   father_path = os.path.join(root_path, partition_path)
#   file_name = father_path.split("/")[-1].split(".")[0]
#   print(file_name)
#   output_father_path = os.path.join(output_root_path, file_name)
#   if not os.path.exists(output_father_path):
#     os.mkdir(output_father_path)
  
#   temp_file = father_path
#   print(temp_file, "==temp_file==", file_name)
#   if (temp_file.endswith("tar.gz")):
#     envoy.run("tar xzf %s -C %s" % (temp_file, output_father_path))

spk_info = {}
all_scp = {}
trans = {}

mapping = {
    u"F":"female",
    u"M":"male"

}

spk_info = '/data/albert.xht/aishell/resource_aishell/speaker.info'

spk_dict = {}
with open(spk_info, "r") as frobj:
  for line in frobj:
    content = line.strip().split()
    spk_dict[content[0]] = {
      "SPKID":"S"+content[0],
      "Gender":mapping[content[1]],
      "Age":"27",
      "Dialect":p.get_pinyin(u"北京", ' ')
    }

transcript_path = '/data/albert.xht/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt'
trans = {}
with open(transcript_path, "r") as frobj:
  for line in frobj:
    content = line.strip().split()
    utterance_id = content[0]
    transcript = " ".join(content[1:])
    trans[utterance_id+".wav"] = {
      "UtteranceID":utterance_id+".wav",
      "SpeakerID":"",
      "Transcription":transcript
    }

# # /data/albert.xht/ST-CMDS-20170001_1-OS
train_scp = {}
root_path = '/data/albert.xht/aishell/datasshell_raw'
total_count = 0
for f1 in os.listdir(root_path):
  # /data/albert.xht/aishell/datasshell_raw/S0002
  spk_id = f1
  f1_path = os.path.join(root_path, f1)
  for f2 in os.listdir(f1_path):
    # /data/albert.xht/aishell/datasshell_raw/S0002/train
    f2_path = os.path.join(f1_path, f2)
    for f3 in os.listdir(f2_path):
      # /data/albert.xht/aishell/datasshell_raw/S0002/train/S0002
      f3_path = os.path.join(f2_path, f3)
      for f4 in os.listdir(f3_path):
        # /data/albert.xht/aishell/datasshell_raw/S0002/train/S0002/BAC009S0002W0
        utterance_id = f4
        total_count += 1
        if utterance_id in trans:
          trans[utterance_id]['SpeakerID'] = spk_id
        train_scp[utterance_id] = os.path.join(f1, f2, f3, f4)
        print(train_scp[utterance_id])

root_path = "/data/albert.xht/aishell"

with open(os.path.join(root_path, 'metadata', "trans.txt"), "w") as fwobj:
  fwobj.write("\t".join(['UtteranceID', 'SpeakerID', 'Transcription'])+"\n")
  for utterance_id in trans:
    fwobj.write("\t".join([trans[utterance_id]['UtteranceID'], 
                          trans[utterance_id]['SpeakerID'], 
                          trans[utterance_id]['Transcription']])+"\n")

with open(os.path.join(root_path, 'metadata', "spk_info.txt"), "w") as fwobj:
  fwobj.write("\t".join(['SPKID', 'Age', 'Gender', "Dialect"])+"\n")
  for spk_id in spk_dict:
    tmp_dict = spk_dict[spk_id]
    fwobj.write("\t".join([tmp_dict['SPKID'], 
                          tmp_dict['Age'], 
                          tmp_dict['Gender'],
                          tmp_dict['Dialect']])+"\n")

with open(os.path.join(root_path, 'metadata', "train.scp"), "w") as fwobj:
  for utterance_id in train_scp:
    print(utterance_id, train_scp[utterance_id],"\t".join([utterance_id, train_scp[utterance_id]]))
    fwobj.write("\t".join([utterance_id, train_scp[utterance_id]])+"\n")
