
import tensorflow as tf
from audio_io import audio_feature_np
import numpy as np
import math

class SpeechFeaturizer(object):
  def __init__(self, speech_config):
    """
    We should use TFSpeechFeaturizer for training to avoid differences
    between tf and librosa when converting to tflite in post-training stage
    speech_config = {
      "sample_rate": int,
      "frame_ms": int,
      "stride_ms": int,
      "num_feature_bins": int,
      "feature_type": str,
      "delta": bool,
      "delta_delta": bool,
      "pitch": bool,
      "normalize_signal": bool,
      "normalize_feature": bool,
      "normalize_per_feature": bool
    }
    """
    # Samples
    self.sample_rate = speech_config.get("sample_rate", 16000)
    self.frame_length = int(self.sample_rate * (speech_config.get("frame_ms", 25) / 1000))
    self.frame_step = int(self.sample_rate * (speech_config.get("stride_ms", 10) / 1000))
    # Features
    self.num_feature_bins = speech_config.get("num_feature_bins", 80)
    self.feature_type = speech_config.get("feature_type", "log_mel_spectrogram")
    self.preemphasis = speech_config.get("preemphasis", None)
    # Normalization
    self.normalize_signal = speech_config.get("normalize_signal", True)
    self.normalize_feature = speech_config.get("normalize_feature", True)
    self.normalize_per_feature = speech_config.get("normalize_per_feature", False)
    self.center = speech_config.get("center", True)
    # Length
    self.max_length = 0
    print(self.frame_length, self.frame_step, self.num_feature_bins, self.nfft)

  @property
  def nfft(self):
    """ Number of FFT """
    return 2 ** (self.frame_length - 1).bit_length()

  @property
  def shape(self):
    """ The shape of extracted features """
    raise NotImplementedError()

  def get_length_from_duration(self, duration):
    nsamples = math.ceil(float(duration) * self.sample_rate)
    if self.center: nsamples += self.nfft
    return 1 + (nsamples - self.nfft) // self.frame_step  # https://www.tensorflow.org/api_docs/python/tf/signal/frame

  def update_length(self, length: int):
    self.max_length = max(self.max_length, length)

  def reset_length(self):
    self.max_length = 0

  def extract(self, signal):
    """ Function to perform feature extraction """
    raise NotImplementedError()

class NPSpeechFeaturizer(SpeechFeaturizer):
  @property
  def shape(self):
    length = self.max_length if self.max_length > 0 else None
    return [length, self.num_feature_bins, 1]

  def extract(self, signal):
    """
    Extract speech features from signals (for using in tflite)
    Args:
      signal: tf.Tensor with shape [None]

    Returns:
      features: tf.Tensor with shape [T, F, 1]
    """
    if self.normalize_signal:
      tf.logging.info("*** normalize_signal ***")
      signal = audio_feature_np.normalize_signal(signal)

    tf.logging.info("*** preemphasis ***")
    signal = audio_feature_np.preemphasis(signal, self.preemphasis)

    if self.feature_type == "spectrogram":
      tf.logging.info("*** spectrogram ***")
      features = audio_feature_np.compute_spectrogram(
          signal, 
          self.nfft, 
          self.frame_step, 
          self.frame_length, 
          self.center,
          self.num_feature_bins,
          ref=1.0, amin=1e-10, top_db=80.0)
    elif self.feature_type == "stft":
      tf.logging.info("*** stft ***")
      features = audio_feature_np.stft(
          signal, 
          self.nfft, 
          self.frame_step, 
          self.frame_length, 
          self.center)
    elif self.feature_type == "log_mel_spectrogram":
      tf.logging.info("*** log_mel_spectrogram ***")
      features = audio_feature_np.compute_log_mel_spectrogram(
          signal, 
          self.nfft, 
          self.frame_step, 
          self.frame_length, 
          self.center,
          self.sample_rate, 
          self.num_feature_bins,
          ref=1.0, amin=1e-10, top_db=80.0)
    elif self.feature_type == "mfcc":
      tf.logging.info("*** mfcc ***")
      features = audio_feature_np.compute_mfcc(
          signal, 
          self.nfft, 
          self.frame_step, 
          self.frame_length, 
          self.center,
          self.sample_rate, 
          self.num_feature_bins,
          ref=1.0, amin=1e-10, top_db=80.0)
    elif self.feature_type == "log_gammatone_spectrogram":
      tf.logging.info("*** log_gammatone_spectrogram ***")
      features = audio_feature_np.compute_log_gammatone_spectrogram(
          signal, 
          self.nfft, 
          self.frame_step, 
          self.frame_length, 
          self.center,
          self.sample_rate, 
          self.num_feature_bins,
          ref=1.0, amin=1e-10, top_db=80.0)
    elif self.feature_type == "log_logfbank_spectrogram":
      tf.logging.info("*** log_logfbank_spectrogram ***")
      features = audio_feature_np.compute_logfbank_feature(
          signal, 
          self.nfft, 
          self.frame_step, 
          self.frame_length, 
          self.center,
          self.sample_rate, 
          self.num_feature_bins,
          ref=1.0, amin=1e-10, top_db=80.0)
    else:
      raise ValueError("feature_type must be either 'mfcc', 'log_mel_spectrogram' or 'spectrogram'")

    if self.normalize_feature:
      tf.logging.info("*** normalize_feature ***")
      features = audio_feature_np.normalize_audio_feature(features, per_feature=self.normalize_per_feature)

    features = np.expand_dims(features, axis=-1)
    return features