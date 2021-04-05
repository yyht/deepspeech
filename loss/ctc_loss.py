
import tensorflow as tf
# from loss.ctc_ops import ctc_loss_v2, ctc_label_dense_to_sparse, ctc_unique_labels
from loss import ctc_ops

"""
https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/ctc.py
"""

def sparse_ctc_loss(y_true, y_pred, input_length, 
                    label_length, time_major=False):
    
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    input_length = tf.cast(input_length, tf.int32)
    label_length = tf.cast(label_length, tf.int32)
    if time_major:
      y_pred = tf.transpose(y_pred, [1,0,2])

    y_true_sparse = ctc_ops.ctc_label_dense_to_sparse(y_true, label_length)

    return tf.nn.ctc_loss(
        labels=y_true_sparse,
        inputs=y_pred,
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

    return ctc_ops.ctc_loss_v2(
        labels=y_true,
        logit_length=input_length,
        logits=y_pred,
        label_length=label_length,
        logits_time_major=time_major,
        unique=indices,
        blank_index=blank_index
    )