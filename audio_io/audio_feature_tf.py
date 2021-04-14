
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as contrib_audio
from tensorflow.python.ops import gen_spectral_ops as contrib_spectral
from tensorflow.contrib import signal as contrib_signal
import os
import io
import abc
import six
import math
import numpy as np
from audio_io.gammatone_tf import fft_weights

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def normalize_audio_feature(audio_feature, per_feature=False):
  """
  TF Mean and variance features normalization
  Args:
    audio_feature: tf.Tensor with shape [T, F]

  Returns:
    normalized audio features with shape [T, F]
  """
  axis = 0 if per_feature else None
  mean = tf.reduce_mean(audio_feature, axis=axis)
  std_dev = tf.math.reduce_std(audio_feature, axis=axis) + 1e-9
  return (audio_feature - mean) / std_dev


def stft(signal, nfft, frame_step, frame_length, center):
  if center: signal = tf.pad(signal, [[nfft // 2, nfft // 2]], mode="REFLECT")
  window = contrib_signal.hann_window(frame_length, periodic=True)
  left_pad = (nfft - frame_length) // 2
  right_pad = nfft - frame_length - left_pad
  window = tf.pad(window, [[left_pad, right_pad]])
  framed_signals = contrib_signal.frame(signal, frame_length=nfft, frame_step=frame_step)
  framed_signals *= window
  return tf.square(tf.abs(contrib_spectral.rfft(framed_signals, [nfft])))


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
  if amin <= 0:
    raise ValueError('amin must be strictly positive')

  magnitude = S

  if six.callable(ref):
    # User supplied a function to calculate reference power
    ref_value = ref(magnitude)
  else:
    ref_value = np.abs(ref)

  log_spec = 10.0 * log10(tf.maximum(amin, magnitude))
  log_spec -= 10.0 * log10(tf.maximum(amin, ref_value))

  if top_db is not None:
    if top_db < 0:
      raise ValueError('top_db must be non-negative')
    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

  return log_spec

def compute_spectrogram(signal, nfft, frame_step, frame_length, center,
          num_feature_bins,
          ref=1.0, amin=1e-10, top_db=80.0):
  S = stft(signal, nfft, frame_step, frame_length, center)
  spectrogram = power_to_db(S, ref, amin, top_db)
  return spectrogram[:, :num_feature_bins]

def compute_log_mel_spectrogram(signal, nfft, frame_step, frame_length, center,
                                sample_rate, num_feature_bins,
                                  ref=1.0, amin=1e-10, top_db=80.0):
  spectrogram = stft(signal, nfft, frame_step, frame_length, center)
  linear_to_weight_matrix = contrib_signal.linear_to_mel_weight_matrix(
    num_mel_bins=num_feature_bins,
    num_spectrogram_bins=spectrogram.shape[-1],
    sample_rate=sample_rate,
    lower_edge_hertz=0.0, 
    upper_edge_hertz=(sample_rate / 2)
  )
  mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
  return power_to_db(mel_spectrogram, ref, amin, top_db)

def compute_mfcc(signal, nfft, frame_step, frame_length, center,
          sample_rate, num_feature_bins,
          ref=1.0, amin=1e-10, top_db=80.0):
  log_mel_spectrogram = compute_log_mel_spectrogram(signal, nfft, frame_step, frame_length, center,
          sample_rate, num_feature_bins,
          ref, amin, top_db)
  return contrib_signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

def compute_log_gammatone_spectrogram(signal, nfft, frame_step, frame_length, center,
          sample_rate, num_feature_bins,
          ref=1.0, amin=1e-10, top_db=80.0):
  S = stft(signal, nfft, frame_step, frame_length, center)

  gammatone = fft_weights(nfft, sample_rate,
              num_feature_bins, width=1.0,
              fmin=0, fmax=int(sample_rate / 2),
              maxlen=(nfft / 2 + 1))

  gammatone_spectrogram = tf.tensordot(S, gammatone, 1)

  return power_to_db(gammatone_spectrogram, ref, amin, top_db)

def compute_logfbank_feature(signal, nfft, frame_step, frame_length, center,
          sample_rate, num_feature_bins,
          ref=1.0, amin=1e-10, top_db=80.0):

  spectrogram = stft(signal, nfft, frame_step, frame_length, center)
  linear_to_weight_matrix = contrib_signal.linear_to_mel_weight_matrix(
    num_mel_bins=num_feature_bins,
    num_spectrogram_bins=spectrogram.shape[-1],
    sample_rate=sample_rate,
    lower_edge_hertz=0.0, upper_edge_hertz=(sample_rate / 2)
  )
  mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
  return tf.log(mel_spectrogram + 1e-20)

def normalize_signal(signal):
  """
  TF Normailize signal to [-1, 1] range
  Args:
    signal: tf.Tensor with shape [None]

  Returns:
    normalized signal with shape [None]
  """
  gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
  return signal * gain

def preemphasis(signal, coeff=0.97):
  """
  TF Pre-emphasis
  Args:
      signal: tf.Tensor with shape [None]
      coeff: Float that indicates the preemphasis coefficient

  Returns:
      pre-emphasized signal with shape [None]
  """
  if not coeff or coeff <= 0.0: return signal
  s0 = tf.expand_dims(signal[0], axis=-1)
  s1 = signal[1:] - coeff * signal[:-1]
  return tf.concat([s0, s1], axis=-1)

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops
def reduce_variance(input_tensor, axis=None, keepdims=False, name=None):
  name = name if name else "reduce_variance"
  with ops.name_scope(name):
    means = tf.reduce_mean(input_tensor, axis=axis, keepdims=True)
    squared_deviations = gen_math_ops.square(input_tensor - means)
    return tf.reduce_mean(squared_deviations, axis=axis, keepdims=keepdims)

def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
  name = name if name else "reduce_std"
  with ops.name_scope(name):
    variance = reduce_variance(input_tensor, axis=axis, keepdims=keepdims)
    return gen_math_ops.sqrt(variance)

def normalize_audio_feature(audio_feature, per_feature=False):
  """
  TF Mean and variance features normalization
  Args:
      audio_feature: tf.Tensor with shape [T, F]

  Returns:
      normalized audio features with shape [T, F]
  """
  axis = 0 if per_feature else None
  mean = tf.reduce_mean(audio_feature, axis=axis)
  std_dev = reduce_std(audio_feature, axis=axis) + 1e-9
  return (audio_feature - mean) / std_dev



