# -*- coding: utf-8 -*-

import multiprocessing
import re
import jieba_fast as jieba
from pydub import AudioSegment
from pypinyin import pinyin, lazy_pinyin, Style
import soundfile as sf
import io, json, os
import numpy as np

id2pinyin = {}
pinyin2id = {}
with open("/data/albert.xht/gaode/chinese_syllable5k_all.txt", "r") as frobj:
    for index, line in enumerate(frobj):
        word = line.strip()
        id2pinyin[index] = word
        pinyin2id[word] = index
print("===pinyin2id==", len(pinyin2id))

CH_PUNCTUATION = u"['\\\\,\\!#$%&\'()*+-/:￥;<=>.?\\n@[\\]^▽_`{|}~'－＂＃＄％＆＇，：；＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡]"
ch_pattern = re.compile(u"[\u4e00-\u9fa5]+")
from collections import OrderedDict

non_ch = u"[^0-9a-zA-Z一-龥㖏\\s\t]+"

num_en = u"[0-9a-z]+"

def char2pinyin(input_text):
  utt = re.sub("(\\[fil\\])", "", input_text)
  utt = re.sub("(\\[spk\\])", "", utt)

  utt = re.sub(non_ch, "", utt)
  cutted_lst = list(jieba.cut(utt))
  token_pinyin_lst = []
  skip_flag = False
  for item in cutted_lst:
    if re.findall(num_en, item):
      skip_flag = True
      continue
    if skip_flag:
      break
    pinyin_lst = lazy_pinyin(item, style=Style.TONE3, neutral_tone_with_five=True)
    token_pinyin_lst.extend(pinyin_lst)
  return token_pinyin_lst
    
def audiowrite(data, fs, destpath, norm=False):
    if norm:
        rms = (data ** 2).mean() ** 0.5
        scalar = 10 ** (-25 / 10) / (rms+eps)
        data = data * scalar
        if max(abs(data))>=1:
            data = data/max(abs(data), eps)
    
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    sf.write(destpath, data, fs)
    return

def read_wave(audio_path):
    import soundfile as sf
    import io
    with open(audio_path, "rb") as frobj:
        audio = frobj.read()
        wave, sr = sf.read(io.BytesIO(audio))
    return wave, sr

from audio2numpy import open_audio
def read_audio(audio_path):
    wave, sr = open_audio(audio_path)
    return wave, sr

def wave2segment(segment_path, wave_path, scp_path, trans_path, chunk_index, dict_list, noise_path, error_path):
    fwobj_scp = open(scp_path, "w")
    fwobj_tran = open(trans_path, "w")
    fwobj_error = open(error_path, "w")
    fwobj_tran.write("\t".join(['UtteranceID','SpeakerID', 'Transcription', 'token_transcription', 'pinyin_transcription'])+"\n")
    
    for index in chunk_index:
        content = dict_list[index]
       
        order_id = content['ordered_id']
        wave_path_ = os.path.join(wave_path, order_id+".mp3")
        wave, sr = read_audio(wave_path_)
        wave_length = wave.shape[0]
        prev_start, prev_end = 0, 0
        noise_index = 0
        
        for segment_index, item in enumerate(content['transcript']):
            start = int(item['start'])
            end = int(item['end'])
            transcript = item['transcript']
            
            if start > prev_end:
                noise_start = 0
                noise_end = start - 1
                noise_wave = wave[noise_start*sr:noise_end*sr]
                noise_utterance_id = order_id+"_"+str(noise_index)
                target_path = os.path.join(noise_path, noise_utterance_id+".wav")
                # audiowrite(noise_wave, sr, target_path, norm=False)
                noise_index += 1
                prev_end = end
                
            if wave_length >= start*sr and wave_length >= end*sr:
                
                flag = True

                token_transcription = " ".join(transcript)
                pinyin_transcription = (char2pinyin(transcript))
                for p in pinyin_transcription:
                    if p not in pinyin2id:
                        flag = False
                        continue
                if not flag:
                    fwobj_error.write("\t".join([order_id, transcript, str(start), str(end)])+"\n")
                    continue
                pinyin_transcription = " ".join(pinyin_transcription)

                segment_wave = wave[start*sr:end*sr]
                utterance_id = order_id+"_"+str(start)+"_"+str(end)
                target_path = os.path.join(segment_path, utterance_id+".wav")
                audiowrite(segment_wave, sr, target_path, norm=False)
                segment_name = segment_path.split("/")[-1]
                fwobj_scp.write("\t".join([utterance_id, segment_name+"/"+utterance_id+".wav"])+"\n")
                fwobj_tran.write("\t".join([utterance_id, order_id, transcript, token_transcription, pinyin_transcription])+"\n")
    fwobj_scp.close()
    fwobj_tran.close()
    fwobj_error.close()
    
def build_index_chunk(num_of_documents, process_num):
    chunk_size = int(num_of_documents/process_num)

    index_chunk = {}
    random_index = np.random.permutation(range(num_of_documents)).tolist()
    for i_index in range(process_num):
        start = i_index * chunk_size
        end = (i_index+1) * chunk_size
        if i_index in index_chunk:
            index_chunk[i_index].extend(random_index[start:end])
        else:
            index_chunk[i_index] = random_index[start:end]
    return index_chunk
            
def multi_process(
                    process_num, segment_path, wav_path, scp_path, trans_path, meta_path, 
                    noise_path, error_path):

    chunk_num = process_num - 1
    content_list = []

    with open(meta_path, "r") as frobj:
        for line in frobj:
            content = json.loads(line.strip())
            content_list.append(content)
    num_of_documents = len(content_list)
    
    chunks = build_index_chunk(num_of_documents, process_num)
    pool = multiprocessing.Pool(processes=process_num)
    
    # chunk_key = 0
    # wave2segment(segment_path, wav_path, scp_path+"_"+str(chunk_key)+".txt", 
    #                  trans_path+"_"+str(chunk_key)+".txt",
    #                  chunks[chunk_key],
    #                  content_list, noise_path,
    #                  error_path+"_"+str(chunk_key)+".txt")

    for chunk_id, chunk_key in enumerate(chunks):
        pool.apply_async(wave2segment,
            args=((segment_path, wav_path, scp_path+"_"+str(chunk_key)+".txt", 
                     trans_path+"_"+str(chunk_key)+".txt",
                     chunks[chunk_key],
                     content_list, noise_path,
                     error_path+"_"+str(chunk_key)+".txt"))) # apply_async
    pool.close()
    pool.join()

# scp_path = '/data/albert.xht/gaode/10000_20210519_wav_segment/train_scp'
# trans_path = '/data/albert.xht/gaode/10000_20210519_wav_segment/trans'
# error_path = '/data/albert.xht/gaode/10000_20210519_wav_segment/error'
# segment_path = '/data/albert.xht/gaode/10000_20210519_wav_segment'
# wav_path = '/data/albert.xht/gaode/10000_20210519'
# meta_path = "/data/albert.xht/gaode/10000_20210519.json"

scp_path = '/data/albert.xht/gaode/fanbinshixiaogou_20210518_wav_segment/train_scp'
trans_path = '/data/albert.xht/gaode/fanbinshixiaogou_20210518_wav_segment/trans'
error_path = '/data/albert.xht/gaode/fanbinshixiaogou_20210518_wav_segment/error'
segment_path = '/data/albert.xht/gaode/fanbinshixiaogou_20210518_wav_segment'
wav_path = '/data/albert.xht/gaode/fanbinshixiaogou_20210518'
meta_path = "/data/albert.xht/gaode/fanbinshixiaogou_20210518.json"

noise_path = "/data/albert.xht/gaode/noise_wave_segment"

multi_process(20, segment_path, wav_path, scp_path, 
                  trans_path, meta_path, noise_path,
                 error_path)