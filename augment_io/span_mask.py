import numpy as np
import tensorflow as tf

def check_tf_version():
  version = tf.__version__
  print("==tf version==", version)
  if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
    return True
  else:
    return False

def ngram_prob(ngram, mask_prob):
  z = np.random.geometric(p=0.2, size=10000)
  prob = []
  for i in range(ngram):
    prob.append((z==(i+1)).sum()/10000.0)
  sum_prob = sum(prob)
  expected_ngram = 0
  for i, value in enumerate(prob):
    prob[i] /= sum_prob
    expected_ngram += prob[i] * (i+1)
  ngram_mask_prob = mask_prob / (expected_ngram+1e-10)
  for i, value in enumerate(prob):
    prob[i] *= (ngram_mask_prob)

  all_prob = [1-ngram_mask_prob] + prob
  prob_size = int((1+len(prob)) / 2 * len(prob) + 1)

  tran_prob = [0.0]*prob_size
  accum = 0
  tran_prob[0] = all_prob[0]
  tran_prob[1] = all_prob[1]
  for j in range(2, len(all_prob)):
    tran_prob[j+accum] = all_prob[j]
    accum += (j-1)

  hmm_tran_prob = np.ones((prob_size, prob_size)) * np.array([tran_prob])
  for i, value in enumerate(tran_prob):
    if value == 0:
      hmm_tran_prob[i-1] = np.zeros((prob_size, ))
      hmm_tran_prob[i-1][i] = 1
  return tran_prob, hmm_tran_prob

def random_uniform_mask(batch_size, seq_len, mask_probability):
  sample_probs = mask_probability * tf.ones((batch_size, seq_len), dtype=tf.float32)
  noise_dist = tf.distributions.Bernoulli(probs=sample_probs, dtype=tf.float32)
  uniform_mask = noise_dist.sample()
  uniform_mask = tf.cast(uniform_mask, tf.int32)
  return uniform_mask

def dynamic_span_mask_v2(batch_size, seq_len, hmm_tran_prob):
  state = tf.zeros((batch_size, seq_len), dtype=tf.int32)
  tran_size = bert_utils.get_shape_list(hmm_tran_prob, expected_rank=[2])
  init_state_prob = tf.random_uniform([batch_size, tran_size[0]],
              minval=0.0,
              maxval=10.0,
              dtype=tf.float32)
  valid_init_state_mask = tf.expand_dims(tf.cast(tf.not_equal(hmm_tran_prob, 0)[:,0], tf.float32), axis=0)
  init_state_prob *= valid_init_state_mask
  init_state = tf.multinomial(tf.log(init_state_prob)+1e-10,
              num_samples=1,
              output_dtype=tf.int32) # batch x 1

  def hmm_recurrence(i, cur_state, state):
    current_prob = tf.gather_nd(hmm_tran_prob, cur_state)
    next_state = tf.multinomial(tf.log(current_prob+1e-10), 
                  num_samples=1, 
                  output_dtype=tf.int32)
    mask = tf.expand_dims(tf.one_hot(i, seq_len), axis=0)
    state = state + tf.cast(mask, tf.int32) * next_state
    return i+1, next_state, state

  _, _, state = tf.while_loop(
      cond=lambda i, _1, _2: i < seq_len,
      body=hmm_recurrence,
      loop_vars=(1, init_state, state),
      back_prop=False,
      )
  span_mask = tf.cast(tf.not_equal(state, 0), tf.int32)
  return state, span_mask

def mask_method(batch_size, seq_len, hmm_tran_prob_list, **kargs):
  mask_probability = kargs.get("mask_probability", 0.2)
  mask_prior = kargs.get("mask_prior", None)

  span_mask_matrix = []
  for i, hmm_tran_prob in enumerate(hmm_tran_prob_list):
    state, span_mask = dynamic_span_mask_v2(batch_size, seq_len, hmm_tran_prob)
    span_mask_matrix.append(span_mask)
  uniform_mask = random_uniform_mask(batch_size, seq_len, mask_probability)
  span_mask_matrix.append(uniform_mask)

  span_mask_matrix = tf.stack(span_mask_matrix, axis=1) # [batch, len(hmm_tran_prob_list), seq]
  if mask_prior is not None:
    mask_prob = tf.tile(tf.expand_dims(mask_prior, 0), [batch_size, 1])
    tf.logging.info("**** apply predefined mask sample prob **** ")
  else:
    mask_prob = tf.random_uniform([batch_size, len(hmm_tran_prob_list)+1],
              minval=0.0,
              maxval=10.0,
              dtype=tf.float32)
    tf.logging.info("**** apply uniform mask sample prob **** ")
  span_mask_idx = tf.multinomial(tf.log(mask_prob)+1e-10,
              num_samples=1,
              output_dtype=tf.int32) # batch x 1
  
  batch_idx = tf.expand_dims(tf.cast(tf.range(batch_size), tf.int32), axis=-1)
  gather_index = tf.concat([batch_idx, span_mask_idx], axis=-1)
  mixed_random_mask = tf.gather_nd(span_mask_matrix, gather_index)
  tf.logging.info("==applying hmm, unigram, ngram mixture mask sampling==")
  return mixed_random_mask

def _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len, num_predict):
  """Turn beg and end indices into actual mask."""
  non_func_mask = tf.not_equal(inputs, 0)
  all_indices = tf.where(
      non_func_mask,
      tf.range(tgt_len, dtype=tf.int32),
      tf.constant(-1, shape=[tgt_len], dtype=tf.int32))
  candidate_matrix = tf.cast(
      tf.logical_and(
          all_indices[None, :] >= beg_indices[:, None],
          all_indices[None, :] < end_indices[:, None]),
      tf.float32)
  cumsum_matrix = tf.reshape(
      tf.cumsum(tf.reshape(candidate_matrix, [-1])),
      [-1, tgt_len])
  masked_matrix = tf.cast(cumsum_matrix <= tf.cast(num_predict, dtype=cumsum_matrix.dtype), tf.float32)
  target_mask = tf.reduce_sum(candidate_matrix * masked_matrix, axis=0)
  is_target = tf.cast(target_mask, tf.bool)

  return is_target, target_mask

def _token_span_mask(inputs, tgt_len, num_predict, 
                    stride=1, min_tok=1, max_tok=10):
  """Sample token spans as prediction targets."""
  # non_pad_len = tgt_len + 1 - stride

  input_mask = tf.cast(tf.not_equal(inputs, 0), dtype=tf.int32)
  num_tokens = tf.cast(tf.reduce_sum(input_mask, -1), tf.int32)
  num_predict = tf.cast(num_predict, tf.int32)

  non_pad_len = num_tokens + 1 - stride

  chunk_len_fp = tf.cast(non_pad_len / num_predict, dtype=tf.float32)
  round_to_int = lambda x: tf.cast(tf.round(x), tf.int32)

  # Sample span lengths from a zipf distribution
  span_len_seq = np.arange(min_tok, max_tok + 1)
  probs = np.array([1.0 /  (i + 1) for i in span_len_seq])

  probs /= np.sum(probs)
  logits = tf.constant(np.log(probs), dtype=tf.float32)
  if check_tf_version():
    span_lens = tf.random.categorical(
        logits=logits[None],
        num_samples=num_predict,
        dtype=tf.int64,
    )[0] + min_tok
  else:
    span_lens = tf.multinomial(
        logits=logits[None],
        num_samples=num_predict,
        output_dtype=tf.int64,
    )[0] + min_tok

  # Sample the ratio [0.0, 1.0) of left context lengths
  span_lens_fp = tf.cast(span_lens, tf.float32)
  left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
  left_ctx_len = left_ratio * span_lens_fp * (chunk_len_fp - 1)
  left_ctx_len = round_to_int(left_ctx_len)

  # Compute the offset from left start to the right end
  right_offset = round_to_int(span_lens_fp * chunk_len_fp) - left_ctx_len

  # Get the actual begin and end indices
  beg_indices = (tf.cumsum(left_ctx_len) +
                 tf.cumsum(right_offset, exclusive=True))
  end_indices = beg_indices + tf.cast(span_lens, dtype=tf.int32)

  # Remove out of range indices
  valid_idx_mask = end_indices < non_pad_len
  beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
  end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

  # Shuffle valid indices
  num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
  order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int64))
  beg_indices = tf.gather(beg_indices, order)
  end_indices = tf.gather(end_indices, order)

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict)

def _single_token_mask(inputs, tgt_len, num_predict, exclude_mask=None):
  """Sample individual tokens as prediction targets."""
  func_mask = tf.equal(inputs, 0)
  if exclude_mask is None:
    exclude_mask = func_mask
  else:
    exclude_mask = tf.logical_or(func_mask, exclude_mask)
  candidate_mask = tf.logical_not(exclude_mask)

  input_mask = tf.cast(tf.not_equal(inputs, 0), dtype=tf.int64)
  num_tokens = tf.cast(tf.reduce_sum(input_mask, -1), tf.int64)

  all_indices = tf.range(tgt_len, dtype=tf.int64)
  candidate_indices = tf.boolean_mask(all_indices, candidate_mask)
  masked_pos = tf.random.shuffle(candidate_indices)
  if check_tf_version():
    masked_pos = tf.sort(masked_pos[:num_predict])
  else:
    masked_pos = tf.contrib.framework.sort(masked_pos[:num_predict])
  target_mask = tf.sparse_to_dense(
      sparse_indices=masked_pos,
      output_shape=[tgt_len],
      sparse_values=1.0,
      default_value=0.0)
  is_target = tf.cast(target_mask, tf.bool)

  return is_target, target_mask

def _online_sample_masks(
    inputs, tgt_len, num_predict, mask_prob=0.15,
    stride=1, min_tok=1, max_tok=10):
  """Sample target positions to predict."""

  # Set the number of tokens to mask out per example
  input_mask = tf.cast(tf.not_equal(inputs, 0), dtype=tf.int64)
  num_tokens = tf.cast(tf.reduce_sum(input_mask, -1), tf.float32)

  tf.logging.info("mask_prob: `%s`.", mask_prob)

  num_predict = tf.maximum(1, tf.minimum(
      num_predict, tf.cast(tf.round(num_tokens * mask_prob), tf.int32)))
  num_predict = tf.cast(num_predict, tf.int32)
  
  is_target, target_mask = _token_span_mask(
                              inputs, tgt_len, num_predict,
                              stride=1,min_tok=1, max_tok=10)

  valid_mask = tf.not_equal(inputs, 0)
  is_target = tf.logical_and(valid_mask, is_target)
  target_mask = target_mask * tf.cast(valid_mask, tf.float32)

  # Fill in single tokens if not full
  cur_num_masked = tf.reduce_sum(tf.cast(is_target, tf.int32))
  extra_mask, extra_tgt_mask = _single_token_mask(
      inputs, tgt_len, num_predict - cur_num_masked, is_target)
  return tf.logical_or(is_target, extra_mask), target_mask + extra_tgt_mask

def create_target_mapping(
    is_target, seq_len, num_predict, **kwargs):
  """Create target mapping and retrieve the corresponding kwargs."""
  # Get masked indices
  indices = tf.range(seq_len, dtype=tf.int64)
  indices = tf.boolean_mask(indices, is_target)

  # Handle the case that actual_num_predict < num_predict
  actual_num_predict = tf.shape(indices)[0]
  pad_len = num_predict - actual_num_predict

  # Create target mapping
  target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
  paddings = tf.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
  target_mapping = tf.concat([target_mapping, paddings], axis=0)
  target_mapping = tf.reshape(target_mapping,
                                         [num_predict, seq_len])

  example = {}
  example['target_mapping'] = target_mapping
  for k, v in kwargs.items():
    pad_shape = [pad_len] + v.shape.as_list()[1:]
    tgt_shape = [num_predict] + v.shape.as_list()[1:]
    example[k] = tf.concat([
        tf.boolean_mask(v, is_target),
        tf.zeros(shape=pad_shape, dtype=v.dtype)], 0)
    example[k].set_shape(tgt_shape)

  return example

def mask_generator(inputs, tgt_len, 
                num_predict,
                mask_prob=0.15,
                stride=1, min_tok=1, max_tok=10):
  # input is mask
  [is_target, target_mask] = _online_sample_masks(
                inputs, tgt_len, num_predict, 
                mask_prob=mask_prob,
                stride=stride, 
                min_tok=min_tok, 
                max_tok=max_tok)

  # create target mapping
  example = create_target_mapping(
      is_target, tgt_len, num_predict,
      masked_weights=target_mask, 
      masked_target=inputs)

  masked_positions = tf.argmax(example['target_mapping'], axis=-1)
  example['masked_positions'] = tf.cast(masked_positions, dtype=tf.int32)
  example['masked_mask'] = tf.cast(is_target, dtype=tf.float32)

  return example