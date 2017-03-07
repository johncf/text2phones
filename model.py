import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq

import numpy as np

class Model():
    def __init__(self, dtype=tf.float32, **kwargs):
        """
        @param kwargs: The following keys are recognized:
            'input_size': dimension of a single input in an input sequence,
            'output_size': dimension of a single output in an output sequence,
            'input_length': maximum length of input sequence (optional?),
            'output_length': maximum length of output sequence (only for training?)
            'batch_size': batch size (default: 1),
            'enc_rnn_size': number of units in the LSTM cell (default: 32),
            'dec_rnn_size': number of units in the LSTM cell (default: 48),
        """

        self.input_size = kwargs['input_size']
        self.output_size = kwargs['output_size']
        input_length = kwargs['input_length']
        batch_size = kwargs.get('batch_size', 1)
        enc_rnn_size = kwargs.get('enc_rnn_size', 32)
        dec_rnn_size = kwargs.get('dec_rnn_size', 48)

        self.input_data = tf.placeholder(tf.int32, [batch_size, input_length])
        # embed input_data into a one-hot representation
        self.inputs = tf.one_hot(self.input_data, self.input_size, dtype=dtype)
        self.sequence_length = tf.placeholder(tf.int32, [batch_size]) # actual sequence lengths

        fw_cell = rnn.BasicLSTMCell(enc_rnn_size, state_is_tuple=True)
        bw_cell = rnn.BasicLSTMCell(enc_rnn_size, state_is_tuple=True)
        fw_cell_zero = fw_cell.zero_state(batch_size, dtype)
        bw_cell_zero = bw_cell.zero_state(batch_size, dtype)

        enc_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.inputs,
                                                     sequence_length=self.sequence_length,
                                                     initial_state_fw=fw_cell_zero,
                                                     initial_state_bw=bw_cell_zero)

        dec_cell = rnn.BasicLSTMCell(dec_rnn_size, state_is_tuple=True)
        dec_cell = rnn.AttentionCellWrapper(dec_cell, attn_length=input_length,
                                            attn_size=2*enc_rnn_size, state_is_tuple=True)

        def embedding_fn(ids):
            return tf.one_hot(ids, self.output_size, dtype=dtype)
        dec_helper = seq2seq.GreedyEmbeddingHelper(embedding_fn,
                start_tokens=tf.zeros([batch_size], dtype=tf.int32), end_token=1)

        cell_init, attn_init, _ = dec_cell.zero_state(batch_size, dtype)
        attn_state_init = tf.concat(enc_out, 2) # batch_size x input_length x 2*enc_rnn_size
        attn_state_init = tf.reshape(attn_state_init, [batch_size, input_length * 2*enc_rnn_size])
        dec = seq2seq.BasicDecoder(dec_cell, dec_helper, (cell_init, attn_init, attn_state_init))

        self.decoder_out, self.final_state = seq2seq.dynamic_decode(dec)


# Also See
#   https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/dw3Y2lnMAJc
#   https://github.com/tensorflow/tensorflow/blob/4ee4d1d/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L109-L111
