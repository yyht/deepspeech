import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from optimizer.adam_weight_decay_utils import AdamWeightDecayOptimizer
from optimizer.optimization import AdamWeightDecayOptimizer as NaiveAdamWeightDecayOptimizer
from tensorflow.python.ops import control_flow_ops

def create_adam_optimizer(
    loss, learning_rate, num_train_steps, weight_decay_rate=0.0, use_tpu=False,
    warmup_steps=0, warmup_proportion=0, lr_decay_power=1.0,
    layerwise_lr_decay_power=-1, n_transformer_layers=None, task_layers=[],
    num_towers=1, tvars=[]):
  """Creates an optimizer and training op."""
  global_step = tf.train.get_or_create_global_step()
  output_learning_rate = tf.identity(learning_rate)
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=weight_decay_rate,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  if not tvars:
    tvars = tf.trainable_variables()
  for var in tvars:
    tf.logging.info("** optimized vars **")
    tf.logging.info(var)
  grads = tf.gradients(loss, tvars)
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=10.0)

  tf.logging.info("*********Num towers is {} *********".format(num_towers))
  for i in range(len(grads)):
    if grads[i] is not None:
      if isinstance(grads[i], ops.IndexedSlices):
        grads[i] = ops.convert_to_tensor(grads[i])
      # grads[i] *= (1. / num_towers)
  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=None)
  new_global_step = global_step + 1
  with tf.control_dependencies([train_op]): 
    train_op = control_flow_ops.group(*[global_step.assign(new_global_step)])
  return train_op, output_learning_rate

def create_optimizer(
    loss, learning_rate, num_train_steps, weight_decay_rate=0.0, use_tpu=False,
    warmup_steps=0, warmup_proportion=0, lr_decay_power=1.0,
    layerwise_lr_decay_power=-1, n_transformer_layers=None, task_layers=[],
    num_towers=1, tvars=[]):
  """Creates an optimizer and training op."""
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=lr_decay_power,
      cycle=False)
  warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
  learning_rate *= tf.minimum(
      1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
  output_learning_rate = tf.identity(learning_rate)

  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=weight_decay_rate,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if not tvars:
    tvars = tf.trainable_variables()
  for var in tvars:
    tf.logging.info("** optimized vars **")
    tf.logging.info(var)
  grads = tf.gradients(loss, tvars)
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=10.0)

  tf.logging.info("*********Num towers is {} *********".format(num_towers))
  for i in range(len(grads)):
    if grads[i] is not None:
      if isinstance(grads[i], ops.IndexedSlices):
        grads[i] = ops.convert_to_tensor(grads[i])
      # grads[i] *= (1. / num_towers)

  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=None)
  new_global_step = global_step + 1
  with tf.control_dependencies([train_op]): 
    train_op = control_flow_ops.group(*[global_step.assign(new_global_step)])
  return train_op, output_learning_rate

def create_optimizer_no_global_step(
    loss, learning_rate, num_train_steps, weight_decay_rate=0.0, use_tpu=False,
    warmup_steps=0, warmup_proportion=0, lr_decay_power=1.0,
    layerwise_lr_decay_power=-1, n_transformer_layers=None, task_layers=[],
    num_towers=1, tvars=[]):
  """Creates an optimizer and training op."""
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=lr_decay_power,
      cycle=False)
  warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
  learning_rate *= tf.minimum(
      1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
  output_learning_rate = tf.identity(learning_rate)

  optimizer = tf.train.AdamOptimizer(
      learning_rate=learning_rate,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6)

  if not tvars:
    tvars = tf.trainable_variables()
  for var in tvars:
    tf.logging.info("** optimized vars **")
    tf.logging.info(var)
  grads = tf.gradients(loss, tvars)
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=10.0)

  tf.logging.info("*********Num towers is {} *********".format(num_towers))
  for i in range(len(grads)):
    if grads[i] is not None:
      if isinstance(grads[i], ops.IndexedSlices):
        grads[i] = ops.convert_to_tensor(grads[i])
      # grads[i] *= (1. / num_towers)

  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=None)
  new_global_step = global_step + 1
  with tf.control_dependencies([train_op]): 
    train_op = control_flow_ops.group(*[global_step.assign(new_global_step)])
  return train_op, output_learning_rate

def naive_create_optimizer(
    loss, learning_rate, num_train_steps, weight_decay_rate=0.0, use_tpu=False,
    warmup_steps=0, warmup_proportion=0, lr_decay_power=1.0,
    layerwise_lr_decay_power=-1, 
    n_transformer_layers=None,
    tvars=[]):
  """Creates an optimizer and training op."""
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=lr_decay_power,
      cycle=False)
  warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
  learning_rate *= tf.minimum(
      1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
  output_learning_rate = tf.identity(learning_rate)
  if layerwise_lr_decay_power > 0:
    print("==apply layerwise_lr_decay_power==")
    learning_rate = _get_layer_lrs(learning_rate, layerwise_lr_decay_power,
                                   n_transformer_layers)
  optimizer = NaiveAdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=weight_decay_rate,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
      include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"])
  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  if not tvars:
    tvars = tf.trainable_variables()
  for var in tvars:
    tf.logging.info("** optimized vars **")
    tf.logging.info(var)
  grads = tf.gradients(loss, tvars)
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=10.0)
  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op, output_learning_rate

def naive_create_adam_optimizer(
    loss, learning_rate, num_train_steps, weight_decay_rate=0.0, use_tpu=False,
    warmup_steps=0, warmup_proportion=0, lr_decay_power=1.0,
    layerwise_lr_decay_power=-1, 
    n_transformer_layers=None,
    tvars=[]):
  """Creates an optimizer and training op."""
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=lr_decay_power,
      cycle=False)
  warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
  learning_rate *= tf.minimum(
      1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
  output_learning_rate = tf.identity(learning_rate)
  if layerwise_lr_decay_power > 0:
    print("==apply layerwise_lr_decay_power==")
    learning_rate = _get_layer_lrs(learning_rate, layerwise_lr_decay_power,
                                   n_transformer_layers)

  optimizer = tf.train.AdamOptimizer(learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-6)
  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  if not tvars:
    tvars = tf.trainable_variables()
  for var in tvars:
    tf.logging.info("** optimized vars **")
    tf.logging.info(var)
  grads = tf.gradients(loss, tvars)
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=10.0)
  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op, output_learning_rate

def naive_create_adam_optimizer_no_global(
    loss, learning_rate, num_train_steps, weight_decay_rate=0.0, use_tpu=False,
    warmup_steps=0, warmup_proportion=0, lr_decay_power=1.0,
    layerwise_lr_decay_power=-1, 
    n_transformer_layers=None,
    tvars=[]):
  """Creates an optimizer and training op."""
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=lr_decay_power,
      cycle=False)
  warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
  learning_rate *= tf.minimum(
      1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
  output_learning_rate = tf.identity(learning_rate)
  if layerwise_lr_decay_power > 0:
    print("==apply layerwise_lr_decay_power==")
    learning_rate = _get_layer_lrs(learning_rate, layerwise_lr_decay_power,
                                   n_transformer_layers)

  optimizer = tf.train.AdamOptimizer(learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-6)
  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  if not tvars:
    tvars = tf.trainable_variables()
  for var in tvars:
    tf.logging.info("** optimized vars **")
    tf.logging.info(var)
  grads = tf.gradients(loss, tvars)
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=10.0)
  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)
  return train_op, output_learning_rate

def naive_create_optimizer_no_global(
    loss, learning_rate, num_train_steps, weight_decay_rate=0.0, use_tpu=False,
    warmup_steps=0, warmup_proportion=0, lr_decay_power=1.0,
    layerwise_lr_decay_power=-1, n_transformer_layers=None,
    tvars=[]):
  """Creates an optimizer and training op."""
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=lr_decay_power,
      cycle=False)
  warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
  learning_rate *= tf.minimum(
      1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
  output_learning_rate = tf.identity(learning_rate)
  if layerwise_lr_decay_power > 0:
    print("==apply layerwise_lr_decay_power==")
    learning_rate = _get_layer_lrs(learning_rate, layerwise_lr_decay_power,
                                   n_transformer_layers)
  optimizer = NaiveAdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=weight_decay_rate,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
      include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"])
  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  if not tvars:
    tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)
  # new_global_step = global_step + 1
  # train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op, output_learning_rate