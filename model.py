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
            output_sos_id: index of output start-of-sequence id (fed into the
                decoder at start; a reserved index that is never actually
                output; default: 0)
            output_eos_id: index of output end-of-sequence id (default: 1)
            enc_rnn_size: number of units in the LSTM cell (default: 32)
            dec_rnn_size: number of units in the LSTM cell (default: 48)
        """
        self._input_size = kwargs['input_size']
        self._output_size = kwargs['output_size']
        self._output_sos_id = kwargs.get('output_sos_id', 0)
        self._output_eos_id = kwargs.get('output_eos_id', 1)
        self._enc_rnn_size = kwargs.get('enc_rnn_size', 32)
        self._dec_rnn_size = kwargs.get('dec_rnn_size', 48)
        self._dtype = dtype

    def _build_model(self, batch_size, helper_build_fn, decoder_maxiters=None):
        # embed input_data into a one-hot representation
        inputs = tf.one_hot(self.input_data, self._input_size, dtype=self._dtype)

        with tf.name_scope('bidir-encoder'):
            fw_cell = rnn.BasicLSTMCell(self._enc_rnn_size, state_is_tuple=True)
            bw_cell = rnn.BasicLSTMCell(self._enc_rnn_size, state_is_tuple=True)
            fw_cell_zero = fw_cell.zero_state(batch_size, self._dtype)
            bw_cell_zero = bw_cell.zero_state(batch_size, self._dtype)

            enc_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                         sequence_length=self.input_lengths,
                                                         initial_state_fw=fw_cell_zero,
                                                         initial_state_bw=bw_cell_zero)

        with tf.name_scope('attn-decoder'):
            attn_values = tf.concat(enc_out, 2)
            attn_ifx = attn.BasicAttentionalInterface(attn_values, self.input_lengths)
            dec_cell_attn = rnn.BasicLSTMCell(self._dec_rnn_size, state_is_tuple=True)
            dec_cell_attn = attn.AttentionCellWrapper(dec_cell_attn, attn_ifx)
            dec_cell_mid = rnn.BasicLSTMCell(self._dec_rnn_size, state_is_tuple=True)
            dec_cell_out = rnn.BasicLSTMCell(self._output_size, state_is_tuple=True)
            dec_cell = rnn.MultiRNNCell([dec_cell_attn, dec_cell_mid, dec_cell_out],
                                        state_is_tuple=True)

            dec = seq2seq.BasicDecoder(dec_cell, helper_build_fn(),
                                       dec_cell.zero_state(batch_size, self._dtype))

            dec_out, _ = seq2seq.dynamic_decode(dec, output_time_major=False,
                    maximum_iterations=decoder_maxiters, impute_finished=True)

        self.outputs = dec_out.rnn_output
        self.output_ids = dec_out.sample_id

    def _output_onehot(self, ids):
        return tf.one_hot(ids, self._output_size, dtype=self._dtype)

    def train(self, batch_size):
        """Build model for training.
        Args:
            batch_size: size of training batch
        """
        self.input_data = tf.placeholder(tf.int32, [batch_size, None])
        self.input_lengths = tf.placeholder(tf.int32, [batch_size]) # actual input lengths
        self.output_data = tf.placeholder(tf.int32, [batch_size, None])
        self.output_lengths = tf.placeholder(tf.int32, [batch_size]) # actual output lengths

        def train_helper():
            start_ids = tf.fill([batch_size, 1], self._output_sos_id)
            decoder_input_ids = tf.concat([start_ids, self.output_data], 1)
            decoder_inputs = self._output_onehot(decoder_input_ids)
            return seq2seq.TrainingHelper(decoder_inputs, self.output_lengths)

        self._build_model(batch_size, train_helper)

        with tf.name_scope("losses"):
            output_maxlen = tf.shape(self.outputs)[1]
            output_data_slice = tf.slice(self.output_data, [0, 0], [-1, output_maxlen])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.outputs, labels=output_data_slice)
            losses_mask = tf.sequence_mask(
                    self.output_lengths, maxlen=output_maxlen, dtype=self._dtype)
            self.losses = tf.reduce_sum(losses * losses_mask, 1)

    def infer(self, output_maxlen=128):
        """Build model for inference.
        """
        self.input_data = tf.placeholder(tf.int32, [1, None])
        self.input_lengths = None

        def infer_helper():
            return seq2seq.GreedyEmbeddingHelper(
                    self._output_onehot,
                    start_tokens=tf.fill([1], self._output_sos_id),
                    end_token=self._output_eos_id)

        self._build_model(1, infer_helper, decoder_maxiters=output_maxlen)

# Also See
#   https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/dw3Y2lnMAJc
