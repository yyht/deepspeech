
import numpy as np
import librosa
from audio_io.gammatone_np import fft_weights

"""
this is compatiable to audio_feature_tf
for in graph feature extraction
"""

def normalize_audio_feature(audio_feature, per_feature=False):
  """ Mean and variance normalization """
  axis = 0 if per_feature else None
  mean = np.mean(audio_feature, axis=axis)
  std_dev = np.std(audio_feature, axis=axis) + 1e-9
  normalized = (audio_feature - mean) / std_dev
  return normalized

def compute_pitch(signal, nfft, frame_step, frame_length,
                sample_rate, num_feature_bins):
  pitches, _ = librosa.core.piptrack(
      y=signal, sr=sample_rate,
      n_fft=nfft, hop_length=frame_step,
      fmin=0.0, fmax=int(sample_rate / 2), win_length=frame_length, center=False
  )

  pitches = pitches.T

  assert num_feature_bins <= frame_length // 2 + 1, \
      "num_features for spectrogram should \
  be <= (sample_rate * window_size // 2 + 1)"

  return pitches[:, :num_feature_bins]

def stft(signal, nfft, frame_step, frame_length, center):
  return np.square(
    np.abs(librosa.core.stft(signal, n_fft=nfft, 
          hop_length=frame_step,
          win_length=frame_length, 
          center=center, window="hann")))

def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
  return librosa.power_to_db(S, ref=ref, amin=amin, top_db=top_db)

def compute_mfcc(signal, nfft, frame_step, frame_length, center,
                  sample_rate, num_feature_bins,
                  ref=1.0, amin=1e-10, top_db=80.0):
  S = stft(signal, nfft, frame_step, frame_length, center)

  mel = librosa.filters.mel(sample_rate, nfft,
                n_mels=num_feature_bins,
                fmin=0.0, fmax=int(sample_rate / 2),
                htk=True, norm=None)

  mel_spectrogram = np.dot(S.T, mel.T)

  mfcc = librosa.feature.mfcc(sr=sample_rate,
                S=power_to_db(mel_spectrogram, ref, amin, top_db).T,
                n_mfcc=num_feature_bins)

  # this compatiable to tf-mfcc calculation
  mfcc /= np.sqrt(num_feature_bins*2)

  return mfcc.T

def compute_log_mel_spectrogram(signal, nfft, frame_step, frame_length, center,
                          sample_rate, num_feature_bins,
                          ref=1.0, amin=1e-10, top_db=80.0):
  S = stft(signal, nfft, frame_step, frame_length, center)

  mel = librosa.filters.mel(sample_rate, nfft,
                n_mels=num_feature_bins,
                fmin=0.0, fmax=int(sample_rate / 2),
                htk=True, norm=None)

  mel_spectrogram = np.dot(S.T, mel.T)

  return power_to_db(mel_spectrogram, ref, amin, top_db)

def compute_logfbank_feature(signal, nfft, frame_step, frame_length, center,
                          sample_rate, num_feature_bins,
                          ref=1.0, amin=1e-10, top_db=80.0):
  S = stft(signal, nfft, frame_step, frame_length, center)

  mel = librosa.filters.mel(sample_rate, nfft,
                n_mels=num_feature_bins,
                fmin=0.0, fmax=int(sample_rate / 2),
                htk=True, norm=None)

  mel_spectrogram = np.dot(S.T, mel.T)

  return np.log(mel_spectrogram + 1e-20).T


def compute_log_gammatone_spectrogram(signal, nfft, frame_step, frame_length, center,
                          sample_rate, num_feature_bins,
                          ref=1.0, amin=1e-10, top_db=80.0):
  S = stft(signal, nfft, frame_step, frame_length, center)

  gammatone = fft_weights(nfft, sample_rate,
              num_feature_bins, width=1.0,
              fmin=0, fmax=int(sample_rate / 2),
              maxlen=(nfft / 2 + 1))

  gammatone_spectrogram = np.dot(S.T, gammatone.T)

  return power_to_db(gammatone_spectrogram, ref, amin, top_db)

def pitch_delta_delta(signal, features, 
                nfft, frame_step, frame_length,
                sample_rate, num_feature_bins,
                if_delta, if_delta_delta, if_pitch,
                normalize_feature, normalize_per_feature=False):
  original_features = features.copy()
  if normalize_feature:
     features = normalize_audio_feature(features, per_feature=normalize_per_feature)
  features = np.expand_dims(features, axis=-1)
  if if_delta:
    delta = librosa.feature.delta(original_features.T).T
    if normalize_feature:
      delta = normalize_audio_feature(delta, per_feature=normalize_per_feature)
    features = np.concatenate([features, np.expand_dims(delta, axis=-1)], axis=-1)

  if if_delta_delta:
    delta_delta = librosa.feature.delta(original_features.T, order=2).T
    if normalize_feature:
        delta_delta = normalize_audio_feature(
            delta_delta, per_feature=normalize_per_feature)
    features = np.concatenate([features, np.expand_dims(delta_delta, axis=-1)], axis=-1)

  if if_pitch:
    pitches = compute_pitch(signal, nfft, frame_step, frame_length,
                sample_rate, num_feature_bins)
    if normalize_feature:
        pitches = normalize_audio_feature(
            pitches, per_feature=normalize_per_feature)
    features = np.concatenate([features, np.expand_dims(pitches, axis=-1)], axis=-1)
  return features

def normalize_signal(signal):
  """ Normailize signal to [-1, 1] range """
  gain = 1.0 / (np.max(np.abs(signal)) + 1e-9)
  return signal * gain

def preemphasis(signal, coeff=0.97):
  if not coeff or coeff <= 0.0:
      return signal
  return np.append(signal[0], signal[1:] - coeff * signal[:-1])
