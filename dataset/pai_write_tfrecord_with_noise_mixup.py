
import tensorflow as tf
import numpy as np
from audio_io import read_audio
from audio_io import utils
import random, librosa

import time

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

def noise_synthesizer(clean_path, 
                    noise_path,
                    speaker_id,
                    noise_id,
                    gender_id,
                    dialect_id,
                    transcript_id,
                    tf_sess,
                    sample_rate=16000,
                    target_sample_rate=8000,
                    pinyin_id=None,
                    tf_string_api=None):
  clean_speech = read_audio.read_raw_audio(clean_path, sample_rate=target_sample_rate)
  noise_wave = read_audio.read_raw_audio(noise_path, sample_rate=target_sample_rate)

  clean_speech_len = clean_speech.shape[0]
  noise_wave_len = noise_wave.shape[0]

  if clean_speech_len > noise_wave_len:
    noise_wave_segment = noise_wave
    while True:
      noise_wave_segment_len = noise_wave_segment.shape[0]
      if noise_wave_segment_len == clean_speech_len:
        break
      if noise_wave_segment_len < clean_speech_len:
        delta_len = clean_speech_len - noise_wave_segment_len

        if delta_len >= noise_wave_len:
          noise_part = noise_wave
        else:
          random_start = np.random.randint(0, noise_wave_len-delta_len)
          noise_part = noise_wave[random_start:random_start+delta_len]
        noise_wave_segment = np.concatenate([noise_wave_segment, noise_part])
  else:
    random_start = np.random.randint(0, noise_wave_len-clean_speech_len)
    noise_wave_segment = noise_wave[random_start:clean_speech_len+random_start]

  SNR = np.linspace(0, 40, 5)

  # snr_level = np.random.randint(0, 5)
  snr_level = np.random.choice(np.arange(0, 5), p=[0.7,0.1,0.1,0.05,0.05])

  clean, noisenewlevel, noisy_speech = snr_mixer(clean_speech, noise_wave_segment, SNR[snr_level])

  start = time.time()

  if sample_rate != target_sample_rate:
    resample_clean_speech = librosa.resample(clean_speech, sample_rate, target_sample_rate)
    resample_noisy_speech = librosa.resample(noisy_speech, sample_rate, target_sample_rate)
  else:
    resample_clean_speech = clean_speech
    resample_noisy_speech = noisy_speech
  # print("==resample time==", time.time()-start)

  resample_clean_speech = resample_clean_speech.astype(np.float32)
  resample_noisy_speech = resample_noisy_speech.astype(np.float32)

  # clean_speech = clean_speech.astype(np.float32)
  # noisy_speech = noisy_speech.astype(np.float32)

  # clean_speech_string = read_audio.tf_encode_raw_audio(clean_speech, target_sample_rate)
  # noisy_speech_string = read_audio.tf_encode_raw_audio(noisy_speech, target_sample_rate)

  start = time.time()

  if not tf_string_api:
    resample_clean_speech_string = read_audio.tf_encode_raw_audio(resample_clean_speech, target_sample_rate)
    resample_noisy_speech_string = read_audio.tf_encode_raw_audio(resample_noisy_speech, target_sample_rate)

    [
    # clean_speech_b, 
    # noisy_speech_b, 
    resample_clean_speech_b,
    resample_noisy_speech_b
    ] = tf_sess.run([
                  # clean_speech_string, 
                  # noisy_speech_string, 
                  resample_clean_speech_string,
                  resample_noisy_speech_string
                  ])
  else:
    resample_clean_speech_b = tf_string_api.run(resample_clean_speech)
    resample_noisy_speech_b = tf_string_api.run(resample_noisy_speech)

  # print("==tf sess run time==", time.time()-start)

  feature = {
    # "clean_audio": utils.bytestring_feature([clean_speech_b]),
    # "noise_audio": utils.bytestring_feature([noisy_speech_b]),
    "clean_audio_resample": utils.bytestring_feature([resample_clean_speech_b]),
    "noise_audio_resample": utils.bytestring_feature([resample_noisy_speech_b]),
    "speaker_id": utils.int64_feature([speaker_id]),
    "noise_id": utils.int64_feature([noise_id]),
    "gender_id": utils.int64_feature([gender_id]),
    "dialect_id": utils.int64_feature([dialect_id]),
    "transcript_id": utils.int64_feature(transcript_id)
  }
  if pinyin_id is not None:
    feature['transcript_pinyin_id'] = utils.int64_feature(pinyin_id)
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  return example, resample_clean_speech.shape[0], resample_clean_speech.shape[0]
