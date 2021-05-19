def _edit_distance(s1, s2):
  if len(s1) > len(s2):
    s1, s2 = s2, s1

  distances = range(len(s1) + 1)
  for i2, c2 in enumerate(s2):
    distances_ = [i2+1]
    for i1, c1 in enumerate(s1):
      if c1 == c2:
        distances_.append(distances[i1])
      else:
        distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
    distances = distances_
  return distances[-1]

def _get_num_words_in_utt(utt_lst):
  return len(utt_lst)

def wer_score(ref, hyp):
  hyp = hyp.split()
  
  edit_dist = _edit_distance(hyp, ref)
  num_ref_words = _get_num_words_in_utt(ref)
  return edit_dist, num_ref_words

"""
data_path = '/data/albert/asr/test/oslr/oslr_test/'
import os, json

edit_dist, num_ref_words = 0, 0

list_dir =os.listdir(data_path)
data_num = 0
for i in range(0,len(list_dir)):
  path = os.path.join(data_path, list_dir[i])
  
  with open(path, "r") as frobj:
    for line in frobj:
      data = json.loads(line.strip())
      ref = data['ref']
      hyp = data['hyp'][0][0]['hyp_0']['text']
      
      edit_dist_, num_ref_words_ = wer_score(ref, hyp)
      edit_dist += edit_dist_
      num_ref_words += num_ref_words_
      data_num += 1
            
wer = edit_dist/num_ref_words
print("==per==", edit_dist/num_ref_words*100)
"""