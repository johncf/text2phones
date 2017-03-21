import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq

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
            enc_rnn_size: number of units in the LSTM cell (default: 48)
            dec_rnn_size: number of units in the LSTM cell (default: 56)
        """
        self._input_size = kwargs['input_size']
        self._output_size = kwargs['output_size']
        self._output_sos_id = kwargs.get('output_sos_id', 0)
        self._output_eos_id = kwargs.get('output_eos_id', 1)
        self._enc_rnn_size = kwargs.get('enc_rnn_size', 48)
        self._dec_rnn_size = kwargs.get('dec_rnn_size', 56)
        self._dtype = dtype

    def _build_model(self, batch_size, helper_build_fn, decoder_maxiters=None):
        # embed input_data into a one-hot representation
        inputs = tf.one_hot(self.input_data, self._input_size, dtype=self._dtype)
        inputs_len = self.input_lengths
        ## truncate inputs (debug dummy)
        #trunc_len = 20
        #inputs = tf.slice(inputs, [0,0,0], [-1, trunc_len, -1])
        #inputs_len_x = tf.expand_dims(inputs_len, 1)
        #inputs_len = tf.reduce_min(tf.concat([inputs_len_x, tf.zeros_like(inputs_len_x) + trunc_len], 1), 1)
        ## end of truncate inputs

        with tf.name_scope('bidir-encoder'):
            fw_cell = rnn.BasicLSTMCell(self._enc_rnn_size, state_is_tuple=True)
            bw_cell = rnn.BasicLSTMCell(self._enc_rnn_size, state_is_tuple=True)
            fw_cell_zero = fw_cell.zero_state(batch_size, self._dtype)
            bw_cell_zero = bw_cell.zero_state(batch_size, self._dtype)

            enc_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                         sequence_length=inputs_len,
                                                         initial_state_fw=fw_cell_zero,
                                                         initial_state_bw=bw_cell_zero)

        with tf.name_scope('attn-decoder'):
            attn_values = tf.concat(enc_out, 2)
            attn_mech = seq2seq.LuongAttention(self._enc_rnn_size * 2, attn_values, inputs_len)
            dec_cell_attn = rnn.BasicLSTMCell(self._enc_rnn_size * 2, state_is_tuple=True)
            dec_cell_attn = seq2seq.DynamicAttentionWrapper(dec_cell_attn, attn_mech, self._enc_rnn_size * 2)
            #dec_cell_mid = rnn.BasicLSTMCell(self._dec_rnn_size, state_is_tuple=True)
            dec_cell_out = rnn.BasicLSTMCell(self._output_size, state_is_tuple=True)
            dec_cell = rnn.MultiRNNCell([dec_cell_attn, dec_cell_out],
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
        self.input_data = tf.placeholder(tf.int32, [batch_size, None], name='input_data')
        self.input_lengths = tf.placeholder(tf.int32, [batch_size], name='input_lengths')
        self.output_data = tf.placeholder(tf.int32, [batch_size, None], name='output_data')
        self.output_lengths = tf.placeholder(tf.int32, [batch_size], name='output_lengths')

        output_data_maxlen = tf.shape(self.output_data)[1]
        #output_data_maxlen = 15 # dummy

        def infer_helper():
            return seq2seq.GreedyEmbeddingHelper(
                    self._output_onehot,
                    start_tokens=tf.fill([batch_size], self._output_sos_id),
                    end_token=self._output_eos_id)

        def train_helper():
            start_ids = tf.fill([batch_size, 1], self._output_sos_id)
            decoder_input_ids = tf.concat([start_ids, self.output_data], 1)
            decoder_inputs = self._output_onehot(decoder_input_ids)
            return seq2seq.TrainingHelper(decoder_inputs, self.output_lengths)

        self._build_model(batch_size, infer_helper, decoder_maxiters=output_data_maxlen)

        output_maxlen = tf.minimum(tf.shape(self.outputs)[1], output_data_maxlen)
        out_data_slice = tf.slice(self.output_data, [0, 0], [-1, output_maxlen])
        out_logits_slice = tf.slice(self.outputs, [0, 0, 0], [-1, output_maxlen, -1])
        out_pred_slice = tf.slice(self.output_ids, [0, 0], [-1, output_maxlen])

        with tf.name_scope("costs"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=out_logits_slice, labels=out_data_slice)

            length_mask = tf.sequence_mask(
                    self.output_lengths, maxlen=output_maxlen, dtype=self._dtype)
            losses = losses * length_mask

            # out_id = 41 => space : reduce the cost of mis-output of space by 40%
            space_mask = tf.cast(tf.equal(out_data_slice, 41), dtype=tf.float32)
            losses = losses * (1.0 - 0.4*space_mask)

            # out_id = 2,3,4,5,6 : AA,AE,AH,AO,AW : reduce the cost by 20% for a-confusion
            data_is_a = tf.logical_and(tf.greater_equal(out_data_slice, 2),
                                       tf.less_equal(out_data_slice, 6))
            pred_is_a = tf.logical_and(tf.greater_equal(out_pred_slice, 2),
                                       tf.less_equal(out_pred_slice, 6))
            a_mask = tf.cast(tf.logical_and(data_is_a, pred_is_a), dtype=tf.float32)
            losses = losses * (1.0 - 0.2*a_mask)

            self.losses = tf.reduce_sum(losses, 1)
            tf.summary.scalar('losses', tf.reduce_sum(self.losses))

            inequality = tf.cast(tf.not_equal(self.output_ids, out_data_slice), dtype=tf.float32)
            # reduce inequality inaccuracy by 20% for a-confusion
            inequality = inequality * (1.0 - 0.1*a_mask)
            self.accuracy = tf.reduce_mean(1.0 - inequality)
            tf.summary.scalar('accuracy', tf.reduce_sum(self.accuracy))

        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-5, epsilon=1e-7).minimize(self.losses)

    def infer(self, output_maxlen=128):
        """Build model for inference.
        """
        self.input_data = tf.placeholder(tf.int32, [1, None], name='input_data')
        self.input_lengths = None

        def infer_helper():
            return seq2seq.GreedyEmbeddingHelper(
                    self._output_onehot,
                    start_tokens=tf.fill([1], self._output_sos_id),
                    end_token=self._output_eos_id)

        self._build_model(1, infer_helper, decoder_maxiters=output_maxlen)

# Also See
#   https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/dw3Y2lnMAJc
#            fw_cell1 = rnn.BasicLSTMCell(self._enc_rnn_size, state_is_tuple=True)
#            fw_cell2 = rnn.BasicLSTMCell(self._enc_rnn_size, state_is_tuple=True)
#            fw_cell = rnn.MultiRNNCell([fw_cell1, fw_cell2], state_is_tuple=True)
#            bw_cell1 = rnn.BasicLSTMCell(self._enc_rnn_size, state_is_tuple=True)
#            bw_cell2 = rnn.BasicLSTMCell(self._enc_rnn_size, state_is_tuple=True)
#            bw_cell = rnn.MultiRNNCell([bw_cell1, bw_cell2], state_is_tuple=True)
