
import tensorflow as tf

def tf_cer(decode, target):
  """Tensorflwo Charactor Error rate

  Args:
    decoder (tf.Tensor): tensor shape [B]
    target (tf.Tensor): tensor shape [B]

  Returns:
    tuple: a tuple of tf.Tensor of (edit distances, number of characters) of each text
  """
  distances = tf.edit_distance(decode.to_sparse(), target.to_sparse(), normalize=False)  # [B]
  lengths = tf.cast(target.row_lengths(axis=1), dtype=tf.float32)  # [B]
  return tf.reduce_sum(distances), tf.reduce_sum(lengths)