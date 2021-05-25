import multiprocessing
from os import listdir
from os.path import isfile, join
from pydub import AudioSegment
import numpy as np
import os

def convertmp32wav(file_path, wav_path, mp3_list, file_name_list):
    for index in file_name_list:
        file_name = mp3_list[index]
        file = join(file_path, file_name)
        sound = AudioSegment.from_mp3(file)
        file_name = "".join(file_name.split(".")[:-1])
        sound.export(os.path.join(wav_path, file_name+".wav"), format="wav")

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
                    process_num, mp3_path, wav_path):

    chunk_num = process_num - 1

    mp3_list = listdir(mp3_path)
    num_of_documents = len(mp3_list)
    
    chunks = build_index_chunk(num_of_documents, process_num)
    pool = multiprocessing.Pool(processes=process_num)
    
#     convertmp32wav(mp3_path, wav_path, mp3_list,
#                     chunks[0])


    for chunk_id, chunk_key in enumerate(chunks):
        
        pool.apply_async(convertmp32wav,
            args=(mp3_path, wav_path, mp3_list,
                    chunks[chunk_id])) # apply_async
    pool.close()
    pool.join()

# mp3_path = '/data/albert.xht/gaode/fanbinshixiaogou_20210518'
# wav_path = '/data/albert.xht/gaode/fanbinshixiaogou_20210518_wav'

mp3_path = '/data/albert.xht/gaode/10000_20210519'
wav_path = '/data/albert.xht/gaode/10000_20210519_wav'

multi_process(20, mp3_path, wav_path)