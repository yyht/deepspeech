import envoy # pip install envoy
import os

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
train_scp = {}
dev_scp = {}
test_scp = {}
trans = {}

root_path = '/data/albert.xht/aidatatang_200zh_raw'
for partition_path in ['train', 'test', 'dev']:
  # /data/albert.xht/aidatatang_200zh_raw/train
  father_path = os.path.join(root_path, partition_path)
  for fi in os.listdir(father_path):
    # /data/albert.xht/aidatatang_200zh_raw/train/G0002
    fi_root_path = os.path.join(father_path, fi)
    
    for fi_path in os.listdir(fi_root_path):
      sub_file = os.path.join(fi_root_path, fi_path)
      if (sub_file.endswith("metadata")):
        tmp_dict = {}
        with open(sub_file) as frobj:
          for line in frobj:
            content = line.strip().split("\t")
            if len(content) == 2:
              tmp_dict[content[0]] = content[1]
        if 'SCD' in tmp_dict:
          if tmp_dict['SCD'] not in spk_info:
            spk_info[tmp_dict['SCD']] = {

            }
            spk_info[tmp_dict['SCD']]['SPKID'] = tmp_dict['SCD']
            spk_info[tmp_dict['SCD']]['Gender'] = tmp_dict['SEX']
            spk_info[tmp_dict['SCD']]['Age'] = tmp_dict['AGE']
            spk_info[tmp_dict['SCD']]['Dialect'] = tmp_dict['BIR']
        
        if "DIR" in tmp_dict:
          utterance_id = fi_path.split(".")[0]
          utterance_id_path = os.path.join([partition_path, fi, utterance_id+".wav"])
          print(utterance_id, utterance_id_path)
          if partition_path == "train":
            train_scp[utterance_id+".wav"] = utterance_id_path
          if partition_path == "test":
            test_scp[utterance_id+".wav"] = utterance_id_path
          if partition_path == "dev":
            dev_scp[utterance_id+".wav"] = utterance_id_path
        
        if "LBD" in tmp_dict:
          label_path = os.path.join(fi_root_path, utterance_id+".txt")
          print(label_path, "==label_path==")
          with open(label_path) as frobj:
            for line in frobj:
              content = line.strip()

          trans[utterance_id+".wav"] = {
            "UtteranceID":utterance_id+".wav",
            "SpeakerID":tmp_dict['SCD'],
            "Transcription":content
          }

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


with open(os.path.join(root_path, 'metadata', "train.scp"), "w") as fwobj:
  for utterance_id in train_scp:
    # print(utterance_id, train_scp[utterance_id],"\t".join([utterance_id, train_scp[utterance_id]]))
    fwobj.write("\t".join([utterance_id, "/".join(train_scp[utterance_id])])+"\n")

with open(os.path.join(root_path, 'metadata', "dev.scp"), "w") as fwobj:
  for utterance_id in dev_scp:
    fwobj.write("\t".join([utterance_id, "/".join(dev_scp[utterance_id])])+"\n")

with open(os.path.join(root_path, 'metadata', "test.scp"), "w") as fwobj:
  for utterance_id in test_scp:
    fwobj.write("\t".join([utterance_id, "/".join(test_scp[utterance_id])])+"\n")