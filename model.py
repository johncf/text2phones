import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq

import numpy as np

class Model():
    def __init__(self, dtype=tf.float32, **kwargs):
        """
        @param kwargs: The following keys are recognized:
            'batch_size': batch size (default: 1),
            'input_max_time': maximum input sequence length (optional?),
            'input_size': dimension of a single input in an input sequence,
            'input_rnn_size': number of units in the LSTM cell (default: 32),
            'output_rnn_size': number of units in the LSTM cell (default: 32),
            'output_size': dimension of a single output in an output sequence,
            'output_max_time': maximum output sequence length (only for training?)
        """

        batch_size = kwargs.get('batch_size', 1)
        input_max_time = kwargs.get('input_max_time', 1)
        input_rnn_size = kwargs.get('input_rnn_size', 32)
        output_rnn_size = kwargs.get('output_rnn_size', 32)

        self.input_data = tf.placeholder(tf.int32, [batch_size,
                                                    input_max_time,
                                                    kwargs['input_size']])
        self.seq_length = tf.placeholder(tf.int32, [batch_size])

        fw_cell = rnn.BasicLSTMCell(input_rnn_size, state_is_tuple=True)
        bw_cell = rnn.BasicLSTMCell(input_rnn_size, state_is_tuple=True)

        enc_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.input_data,
                                                     sequence_length=self.seq_length)

        # TODO attention to encoder from decoder
        dec_cell = rnn.BasicLSTMCell(output_rnn_size, state_is_tuple=True)
        dec_cell = rnn.AttentionCellWrapper(dec_cell, attn_length=input_max_time,
                                            attn_size=2*input_rnn_size, state_is_tuple=True)

        def initialize_fn():
            # TODO (Note: finished: a tensor of bools)
            return (finished, next_inputs)
        def sample_fn(time, outputs, state):
            # TODO
            return sample_ids
        def next_inputs_fn(time, outputs, state, sample_ids):
            # TODO
            return (finished, next_inputs, next_state)
        dec_helper = seq2seq.CustomHelper(initialize_fn, sample_fn, next_inputs_fn)

        cell_init, attn_init, _ = dec_cell.zero_state(batch_size, dtype)
        attn_state_init = tf.concat(enc_out, 2) # batch_size x input_max_time x 2*input_rnn_size
        dec = seq2seq.BasicDecoder(dec_cell, dec_helper, (cell_init, attn_init, attn_state_init))

        dec_out, _ = seq2seq.dynamic_decode(dec_cell)
        self.decoder_out = dec_out


# Also See
#   https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/dw3Y2lnMAJc
#   https://github.com/tensorflow/tensorflow/blob/4ee4d1d/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L109-L111
