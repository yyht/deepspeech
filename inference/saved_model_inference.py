from tensorflow.contrib import predictor
import tensorflow as tf
import numpy as np

from collections import OrderedDict

class SavedModelInfer(object):
  def __init__(self, config):
    self.config = config
    self.predict_fn = None

    self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
                  allow_growth=False)
    self.session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=0,
            allow_soft_placement=True,
            gpu_options=self.gpu_options)
    self.graph = tf.Graph()

  def load_model(self):
    with self.graph.as_default():
      self.sess = tf.Session()
      self.predict_fn = predictor.from_saved_model(self.config['model'],
                            graph=self.graph,
                            config=self.session_conf)

      self.logits = tf.placeholder(tf.float32, [None, 501, 1197], name='logits')
      self.reduced_length = tf.placeholder(tf.int32, [None], name='reduced_length')
      [self.decoded_path, 
      self.log_probability] = tf.nn.ctc_beam_search_decoder(
          tf.transpose(self.logits, [1,0,2]), 
          self.reduced_length, 
          beam_width=self.config.get('beam_width', 10), 
          top_paths=self.config.get('top_paths', 10), 
          merge_repeated=False)

      self.decoded_path_lst = []
      for i in range(self.config.get('top_paths', 10)):
        decoded = tf.to_int32(self.decoded_path[i])
        decoded_path_ = tf.sparse_tensor_to_dense(decoded)
        self.decoded_path_lst.append(decoded_path_)

  def infer(self, input_dict):
    with self.graph.as_default():
      if not self.predict_fn:
        tf.logging.info("====reload model====")
        self.load_model()
        tf.logging.info("====succeeded in loading model====")

      outPred = self.predict_fn(input_dict)
    return outPred

  def ctc_beam_decode(self, 
                    outPred,
                    id2output):
    with self.graph.as_default():
      if not self.predict_fn:
        tf.logging.info("====reload model====")
        self.load_model()
        tf.logging.info("====succeeded in loading model====")

      resp = self.sess.run({
          "decoded_path":self.decoded_path_lst,
          "log_probability":self.log_probability
          }, feed_dict={
        self.logits:outPred['logits'],
        self.reduced_length:outPred['reduced_length']
        })

    output_lst = [[]]*outPred['logits'].shape[0]
    for i in range(len(resp['decoded_path'])):
      for batch_id in range(outPred['logits'].shape[0]):
        hypothesis = ' '.join([id2output[no] for no in resp['decoded_path'][i][batch_id]])
        output_lst[batch_id].append({
          "hyp_{}".format(i): {
          "score": resp['log_probability'][batch_id][i],
          "text": hypothesis
          },
        }
      )

    return output_lst