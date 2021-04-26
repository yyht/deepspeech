
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:54:05 2019
@author: chkarada
"""
import soundfile as sf
import os
import numpy as np

from audio_io import audio_featurizer_tf
from augment_io import augment
from audio_io import read_audio
import matplotlib.pylab as pylab

eps = 1e-10

# Function to read audio
def audioread(path, norm = False, start=0, stop=None):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        x, sr = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(x.shape) == 1:  # mono
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr
    else:  # multi-channel
        x = x.T
        x = x.sum(axis=0)/x.shape[0]
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr
    
# Funtion to write audio    
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

# Function to mix clean speech and noise at various SNR levels
def snr_mixer(clean, noise, snr):
    # Normalizing to -25 dB FS
    rmsclean = (clean**2).mean()**0.5
    scalarclean = 10 ** (-25 / 20) / rmsclean
    clean = clean * scalarclean
    rmsclean = (clean**2).mean()**0.5

    rmsnoise = (noise**2).mean()**0.5
    scalarnoise = 10 ** (-25 / 20) /rmsnoise
    noise = noise * scalarnoise
    rmsnoise = (noise**2).mean()**0.5
    
    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return clean, noisenewlevel, noisyspeech

import tensorflow as tf
import tensorflow

def get_reduced_length(length, reduction_factor):
  return tf.cast(tf.math.ceil(tf.divide(length, tf.cast(reduction_factor, dtype=length.dtype))), dtype=tf.int32)

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def _decode_record(record, name_to_features, **kargs):
    example = tf.parse_single_example(record, name_to_features)
    noise_audio = read_audio.tf_read_raw_audio(example['clean_audio_resample'], 
                      samples_per_second=8000,
                        use_tpu=True)
    print(noise_audio)

    noise_feature = audio_featurizer.tf_extract(noise_audio)
    noise_aug_feature = augment_api.after.augment(noise_feature)

#     feature_shape = shape_list(noise_feature)
#     feature_seq_length = tf.cast(feature_shape[0], dtype=tf.int32)
    
#     reduced_length = get_reduced_length(feature_seq_length, 4)

#     inputs = tf.sequence_mask(reduced_length, 501)
# #     inputs = tf.ones(reduced_length)
    
#     input_shape = shape_list(inputs)
    
#     print(inputs, "==inputs==", input_shape)
#     inputs = tf.cast(inputs, dtype=tf.int32)
#     inputs = tf.ones((501))
#     span_mask_examples = mask_generator(inputs, 
#                 501, 
#                 num_predict=150,
#                 mask_prob=0.2,
#                 stride=1, 
#                 min_tok=10, 
#                 max_tok=10)
#     print(span_mask_examples)
    
#     inputs = tf.ones((feature_seq_length))
#     time_mask = batch_time_mask(tf.expand_dims(inputs, axis=-1), 
#                 num_masks=10, 
#                 mask_factor=100, 
#                 p_upperbound=0.065)
    
    return {
        "noise_audio":noise_audio,
        "noise_feature":noise_feature,
        "noise_aug_feature":noise_aug_feature,
#         "feature_seq_length":feature_seq_length,
        "transcript_id":example['transcript_id'],
#         "masked_mask":span_mask_examples["masked_mask"],
#         "time_mask":time_mask,
#         "masked_positions":span_mask_examples['masked_positions'],
#         "masked_weights":span_mask_examples['masked_weights'],
#         "transcript_pinyin_id":example["transcript_pinyin_id"]
    }

name_to_features = {
    "clean_audio_resample": tf.FixedLenFeature([], tf.string),
    "noise_audio_resample": tf.FixedLenFeature([], tf.string),
    "speaker_id":tf.FixedLenFeature([], tf.int64),
    "transcript_id":tf.FixedLenFeature([128], tf.int64),
    "transcript_pinyin_id":tf.FixedLenFeature([128], tf.int64),
}

def train_input_fn(input_file, _parse_fn, name_to_features):

    dataset = tf.data.TFRecordDataset(input_file, buffer_size=4096)
    dataset = dataset.shuffle(2048)
    dataset = dataset.map(lambda x:_decode_record(x, name_to_features))
    dataset = dataset.batch(1)
    
#     dataset = dataset.padded_batch(
#             batch_size=1,
#             padded_shapes={
#         'noise_audio':tf.TensorShape([160000]),
#         'noise_feature':tf.TensorShape([2001, 80]),
#         },
#          padding_values={
#              "noise_audio":0.0,
#              "noise_feature":0.0
#          }   
#         )
    
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features


sess = tf.Session()


features = train_input_fn(['gs://yyht_source/pretrain/aishell_8000/train/chinese_asr_0.tfrecord'], '', name_to_features
               )

sess.run(tf.group(tf.global_variables_initializer(), tf.tables_initializer()))

ppp = []
while True:
#     try:
    resp_features = sess.run(features)
    break

import matplotlib.pylab as pylab
pylab.imshow(resp_features['noise_feature'][0, :, :, 0].T, cmap="hot", origin='lower', aspect='auto', interpolation='nearest')