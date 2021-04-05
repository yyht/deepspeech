
import tensorflow as tf
from warpctc_tensorflow import ctc

def warpctc_loss(y_true, y_pred, input_length, 
                    label_length, time_major=True):

  # [batch_size, time_steps, vocab]-->
  # [time_steps, batch_size, vocab]
  y_pred = tf.transpose(y_pred, [1,0,2])

  y_true = tf.cast(y_true, tf.int32)
  y_pred = tf.cast(y_pred, tf.float32)
  input_length = tf.cast(input_length, tf.int32)
  label_length = tf.cast(label_length, tf.int32)

  costs = ctc(activations=y_pred,
              flat_labels=y_true,
              label_lengths=label_length,
              input_lengths=input_length)
  return costs