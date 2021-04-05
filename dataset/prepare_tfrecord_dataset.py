
from audio_io import audio_feature_np
from audio_io import audio_feature_tf
from audio_io import audio_featurizer_tf
from audio_io import audio_featurizer_np
from audio_io import read_audio
from augment_io import augment
from audio_io import maptext2id
import tensorflow as tf

class AudioTfrecordDataset(object):
  def __init__(self, config):
    self.alphabet = []
    with tf.gfile.Open(config['alphabet']) as frobj:
      for line in frobj:
        self.alphabet.append(line.strip())

    self.blank_id = config.get('blank_id', 0)
    self.sample_rate = config.sample_rate
    self.has_tpu = config.get('has_tpu', False)
    self.augmentations = Augmentation
    self.vocab_list = []
    with tf.gfile.Open(config.get('vocab_path', "")) as frobj:
      for line in frobj:
        self.vocab_list.append(line.strip())
    self.vocab_tensor_table = maptext2id.build_index_table(self.vocab_list)
    self.delimiter = config.get('delimiter', '&')

  def tf_spec_preprocess(self, audio, indices):
    with tf.device("/CPU:0"):
      signal = read_audio.tf_read_raw_audio(audio, self.sample_rate, self.has_tpu)
      features = self.speech_featurizer.tf_extract(signal)

      augment_features = self.augmentations.after.augment(features)
      
      label = maptext2id.full_onehot_process_line_as_1d_input(
                    indices, 
                    self.vocab_tensor_table,
                    delimiter=self.delimiter)
      label_length = tf.cast(tf.shape(label)[0], tf.int32)

      augment_features = tf.convert_to_tensor(augment_features, tf.float32)
      input_length = tf.cast(tf.shape(augment_features)[0], tf.int32)

      return features, augment_features, input_length, label, label_length
