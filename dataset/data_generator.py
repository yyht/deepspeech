
from tokenizer.snippets import is_string, is_py2
import numpy as np
import tensorflow as tf

class DataGenerator(object):
  """template for DataGenerator
  """
  def __init__(self, batch_size=32, buffer_size=None):
    self.batch_size = batch_size
    self.buffer_size = buffer_size or batch_size * 1000

  def iteration(self, data_path_dict):
    raise NotImplementedError

  def to_dict(self, data_path_dict, types, shapes, names=None):
    if names is None:
      for d in self.iteration(data_path_dict):
        yield d
    else:
      def warps(key, value):
        output_dict = {}
        for key_name, value_name in zip(key, value):
          output_dict[key_name] = value_name
        return output_dict

      for d in self.iter(data_path_dict):
        yield d

  def _fixup_shape(self, record, shapes):
    for key in record:
      record[key].set_shape(shapes[key])
    return record

  def to_dataset(self, data_path_dict, types, shapes, names=None, padded_batch=False,
              is_training=False):
    """
    """
    if names is None:
      def generator():
        for d in self.iteration(data_path_dict):
          yield d
    else:

      def warps(key, value):
        output_dict = {}
        for key_name, value_name in zip(key, value):
          output_dict[key_name] = value_name
        return output_dict

      def generator():
        for d in self.iteration(data_path_dict):
          yield d

      types = warps(names, types)
      
      shapes = warps(names, shapes)

    if padded_batch:
      dataset = tf.data.Dataset.from_generator(
        generator, output_types=types
      )
      if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self.buffer_size)
      dataset = dataset.map(lambda record: self._fixup_shape(record, shapes))
      dataset = dataset.padded_batch(self.batch_size, 
                              padded_shapes=shapes,
                              drop_remainder=True)
    else:
      dataset = tf.data.Dataset.from_generator(
        generator, output_types=types, output_shapes=shapes
      )
      if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self.buffer_size)
      dataset = dataset.map(lambda record: self._fixup_shape(record, shapes))
      dataset = dataset.batch(self.batch_size, drop_remainder=True)
      
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    return dataset
