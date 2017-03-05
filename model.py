import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

class Model():
    def __init__(self, **kwargs):
        """
        @param kwargs: The following keys are recognized:
            'batch_size': batch size (default: 1),
            'in_max_time': maximum input sequence length (optional?),
            'out_max_time': maximum output sequence length (optional?),
            'input_size': dimension of a single input in an input sequence,
            'input_rnn_size': number of units in the LSTM cell (default: 32),
            'output_rnn_size': number of units in the LSTM cell (default: 32),
            'output_size': dimension of a single output in an output sequence (see rnn.OutputProjectionWrapper)
        """

        self.input_data = tf.placeholder(tf.int32, [kwargs.get('batch_size', 1),
                                                    kwargs['in_max_time'],
                                                    kwargs['input_size']])
        self.seq_length = tf.placeholder(tf.int32, [kwargs.get('batch_size', 1)])

        fw_cell = rnn.BasicLSTMCell(kwargs.get('rnn_size', 32), state_is_tuple=True)
        bw_cell = rnn.BasicLSTMCell(kwargs.get('rnn_size', 32), state_is_tuple=True)

        interms, interm_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.input_data,
                sequence_length=self.seq_length)

        # TODO attention of interms from an output rnn

# Also See
#   https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/dw3Y2lnMAJc
#   https://github.com/tensorflow/tensorflow/blob/4ee4d1d/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L109-L111
