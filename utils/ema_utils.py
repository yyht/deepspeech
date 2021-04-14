
import tensorflow as tf
import numpy as np

"""
https://stackoverflow.com/questions/49147961/how-to-use-exponential-moving-average-in-tensorflow
https://www.coder.work/article/1251863
"""

class EMA(object):
  def __init__(self, average_decay, start_step):
    with tf.variable_scope('ema', reuse=tf.AUTO_REUSE):
      self.global_step = tf.get_variable(
                        "global_step",
                        dtype=tf.int64,
                        initializer=tf.constant(1, dtype=tf.int64),
                        reuse=tf.AUTO_REUSE)
    self.start_step = start_step
    self.init_average_decay = tf.constant(average_decay, dtype=tf.float32)
    self._get_decay()
    self.ema = tf.train.ExponentialMovingAverage(decay=self.average_decay)

  def _get_decay(self):
    _step = tf.cast(self.global_step, dtype=tf.float32)
    _start_step = tf.constant(self.start_step, dtype=tf.float32)
    cond_fn = tf.less(_step, _start_step)
    step_count = _step - _start_step
    dynamic_decay_rate = tf.minimum(self.init_average_decay, (1.0+step_count)/(10.0+step_count))
    self.average_decay = tf.cond(cond_fn,
                  lambda:tf.constant(value=0.0, shape=[], dtype=tf.float32, name="first_stage_decay"),
                  lambda:dynamic_decay_rate)

  def _restore_vars(self, var_lst=None):
    if not var_lst:
      ema_variables = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
    else:
      ema_variables = var_lst
    return tf.group(*[tf.assign(x, self.ema.average(x)) for x in ema_variables])

  def _apply_ema(self, var_lst):
    ema_op = self.ema.apply(var_lst)
    return ema_op

  def _get_ema_op(self, var_lst, is_training=False):
    if is_training:
      return self._apply_ema(var_lst)
    else:
      return self._restore_vars(var_lst)

  def ema_getter(self, getter, name, *args, **kwargs):
    var = getter(name, *args, **kwargs)
    ema_var = self.ema.average(var)
    return ema_var if ema_var else var

class RestoreParametersAverageValues(tf.train.SessionRunHook):
  """
  Replace parameters with their moving averages.
  This operation should be executed only once, and before any inference.
  """
  def __init__(self, ema, ema_variables):
    """
    :param ema:         tf.train.ExponentialMovingAverage
    """
    super(RestoreParametersAverageValues, self).__init__()
    self._ema = ema
    self._restore_ops = None
    self.ema_variables = ema_variables

  def begin(self):
    """ Create restoring operations before the graph been finalized. """
    if not self.ema_variables:
      ema_variables = tf.moving_average_variables()
    self._restore_ops = [tf.assign(x, self._ema.average(x)) for x in ema_variables]
    print("==get restore ops==")

  def after_create_session(self, session, coord):
    """ Restore the parameters right after the session been created. """
    print("==restore ema variables==")
    session.run(self._restore_ops)



