import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq
import attn

import numpy as np

class Model():
    def __init__(self, dtype=tf.float32, **kwargs):
        """
        Args:
          The following kwargs are recognized:
            input_size: dimension of a single input in an input sequence
            output_size: dimension of a single output in an output sequence
            enc_rnn_size: number of units in the LSTM cell (default: 32)
            dec_rnn_size: number of units in the LSTM cell (default: 48)
        """
        self._input_size = kwargs['input_size']
        self._output_size = kwargs['output_size']
        self._enc_rnn_size = kwargs.get('enc_rnn_size', 32)
        self._dec_rnn_size = kwargs.get('dec_rnn_size', 48)
        self._dtype = dtype

    def train(self, batch_size, input_length, output_length):
        """
        Args:
            batch_size: size of training batch
            input_length: maximum length of input sequence
            output_length: maximum length of output sequence
        """
        self.input_data = tf.placeholder(tf.int32, [batch_size, input_length])
        self.sequence_length = tf.placeholder(tf.int32, [batch_size]) # actual sequence lengths

        # embed input_data into a one-hot representation
        inputs = tf.one_hot(self.input_data, self._input_size, dtype=self._dtype)

        with tf.name_scope('bidir-encoder'):
            fw_cell = rnn.BasicLSTMCell(self._enc_rnn_size, state_is_tuple=True)
            bw_cell = rnn.BasicLSTMCell(self._enc_rnn_size, state_is_tuple=True)
            fw_cell_zero = fw_cell.zero_state(batch_size, self._dtype)
            bw_cell_zero = bw_cell.zero_state(batch_size, self._dtype)

            enc_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                         sequence_length=self.sequence_length,
                                                         initial_state_fw=fw_cell_zero,
                                                         initial_state_bw=bw_cell_zero)

        with tf.name_scope('attn-decoder'):
            attn_values = tf.concat(enc_out, 2)
            attn_ifx = attn.BasicAttentionalInterface(attn_values, self.sequence_length)
            dec_cell = rnn.BasicLSTMCell(self._dec_rnn_size, state_is_tuple=True)
            dec_cell = attn.AttentionCellWrapper(dec_cell, attn_ifx)

            def embedding_fn(ids):
                return tf.one_hot(ids, self._output_size, dtype=self._dtype)

            dec_helper = seq2seq.GreedyEmbeddingHelper(embedding_fn,
                    start_tokens=tf.zeros([batch_size], dtype=tf.int32), end_token=1)

            dec = seq2seq.BasicDecoder(dec_cell, dec_helper, dec_cell.zero_state(batch_size, self._dtype))

            self.dec_out, _ = seq2seq.dynamic_decode(dec, output_time_major=True,
                    maximum_iterations=output_length)

        #with tf.name_scope('')

        #with tf.name_scope('loss'):
        #    a

# Also See
#   https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/dw3Y2lnMAJc
#   https://github.com/tensorflow/tensorflow/blob/4ee4d1d/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L109-L111
