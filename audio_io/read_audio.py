
try:
  import soundfile as wavfile
  soundfile_flag = True
except:
  import scipy.io.wavfile as wavfile
  soundfile_flag = False
try:
  import librosa
except:
  librosa = None
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as contrib_audio
import io
import numpy as np

def read_bytes(path):
  with tf.gfile.Open(path, "rb") as f:
    content = f.read()
  return tf.convert_to_tensor(content, dtype=tf.string)

def read_bytes_v1(path):
  content = tf.io.read_file(audio_path)
  return content

def read_raw_audio(audio_path, sample_rate=8000):
  """
  wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate, mono=True)
  """
  with tf.gfile.Open(audio_path, "rb") as frobj:
    audio = frobj.read()
  if soundfile_flag:
    wave, sr = wavfile.read(io.BytesIO(audio))
  else:
    # normalize to [-1,1]
    sr, wave = wavfile.read(io.BytesIO(audio))
    wave = wave.astype(np.float32)
    wave /= 32768.0
  if wave.ndim > 1: wave = np.mean(wave, axis=-1)
  if sr != sample_rate: 
    wave = librosa.resample(wave, sr, sample_rate)
  return wave

def tf_read_raw_audio_from_path(audio_path, use_tpu=False):
  content_tensor = read_bytes(audio_path)
  wave, rate = contrib_audio.decode_wav(content_tensor, desired_channels=1)
  # wave shape: [None, 1]
  return tf.reshape(wave, shape=[-1])

def tf_read_raw_audio(audio_tensor, samples_per_second=8000,
                        use_tpu=False):
  # if use_tpu:
  wave, rate = contrib_audio.decode_wav(audio_tensor, desired_channels=1)
  # else:
  #   wave = tf.contrib.ffmpeg(
  #     audio_tensor,
  #     file_format='wav',
  #     samples_per_second=samples_per_second,
  #     channel_count=1
  #     )
  return tf.reshape(wave, shape=[-1])

def tf_encode_raw_audio(audio_array, samples_per_second):
  audio_tensor = tf.convert_to_tensor(audio_array)
  audio_tensor = tf.expand_dims(audio_array, axis=-1)
  audio_string_tensor = contrib_audio.encode_wav(audio_tensor, sample_rate=samples_per_second)
  return audio_string_tensor





