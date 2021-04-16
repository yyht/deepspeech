
import tensorflow as tf
# from loss.ctc_ops import ctc_loss_v2, ctc_label_dense_to_sparse, ctc_unique_labels
from loss import ctc_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import sparse_tensor

"""
https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/ctc.py
"""

def _get_dim(tensor, i):
  """Get value of tensor shape[i] preferring static value if available."""
  return tensor_shape.dimension_value(
      tensor.shape[i]) or array_ops.shape(tensor)[i]

def sparse_ctc_loss(y_true, y_pred, input_length, 
                    label_length, 
                    time_major=False,
                    blank_index=0):
    
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    input_length = tf.cast(input_length, tf.int32)
    label_length = tf.cast(label_length, tf.int32)
    if time_major:
      y_pred = tf.transpose(y_pred, [1,0,2])

    labels = ctc_ops.ctc_label_dense_to_sparse(y_true, label_length)
    logits = y_pred

    if blank_index < 0:
      blank_index += _get_dim(logits, 2)

    if blank_index != _get_dim(logits, 2) - 1:
      logits = array_ops.concat([
          logits[:, :, :blank_index],
          logits[:, :, blank_index + 1:],
          logits[:, :, blank_index:blank_index + 1],
      ],
                                axis=2)
    
      labels = sparse_tensor.SparseTensor(
          labels.indices,
          array_ops.where(labels.values < blank_index, labels.values,
                          labels.values - 1), labels.dense_shape)
    tf.logging.info("** blank_index **")
    tf.logging.info(blank_index)
    
    return tf.nn.ctc_loss(
        labels=labels,
        inputs=logits,
        sequence_length=input_length,
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=True,
        time_major=time_major
    )

def dense_ctc_loss(y_true, y_pred, 
                  input_length, 
                  label_length, 
                  indices=None,
                  blank_index=0,
                  time_major=False):
    
    if time_major:
      y_pred = tf.transpose(y_pred, [1,0,2])

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    input_length = tf.cast(input_length, tf.int32)
    label_length = tf.cast(label_length, tf.int32)

    tf.logging.info("*** y_true ***")
    tf.logging.info(y_true)

    tf.logging.info("*** y_pred ***")
    tf.logging.info(y_pred)

    tf.logging.info("*** input_length ***")
    tf.logging.info(input_length)

    tf.logging.info("*** label_length ***")
    tf.logging.info(label_length)

    print(indices, "===indices===")
    if blank_index < 0:
      blank_index += _get_dim(logits, 2)

    tf.logging.info("** blank_index **")
    tf.logging.info(blank_index)

    return ctc_ops.ctc_loss_v2(
        labels=y_true,
        logit_length=input_length,
        logits=y_pred,
        label_length=label_length,
        logits_time_major=time_major,
        unique=indices,
        blank_index=blank_index
    )