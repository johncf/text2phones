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
            enc_size: number of units in the LSTM cell (default: 42)
            dec_size: number of units in the LSTM cell (default: 96)
        """
        self._input_size = kwargs['input_size']
        self._output_size = kwargs['output_size']
        self._output_sos_id = kwargs.get('output_sos_id', 0)
        self._output_eos_id = kwargs.get('output_eos_id', 1)
        self._enc_size = kwargs.get('enc_size', 42)
        self._dec_size = kwargs.get('dec_size', 96)
        self._dtype = dtype

    def _build_model(self, batch_size, helper_build_fn, decoder_maxiters=None, alignment_history=False):
        # embed input_data into a one-hot representation
        inputs = tf.one_hot(self.input_data, self._input_size, dtype=self._dtype)
        inputs_len = self.input_lengths

        with tf.name_scope('conv-encoder'):
            W = tf.Variable(tf.truncated_normal([3, self._input_size, self._enc_size], stddev=0.1), name="conv-weights")
            b = tf.Variable(tf.truncated_normal([self._enc_size], stddev=0.1), name="conv-bias")
            enc_out = tf.nn.elu(tf.nn.conv1d(inputs, W, stride=1, padding='SAME') + b)

        with tf.name_scope('attn-decoder'):
            dec_cell_in1 = rnn.GRUCell(self._dec_size)
            dec_cell_in2 = rnn.GRUCell(self._dec_size)
            memory = enc_out
            attn_mech = seq2seq.LuongMonotonicAttention(self._enc_size, memory,
                                                        memory_sequence_length=inputs_len,
                                                        sigmoid_noise=0.5, score_bias_init=-4.,
                                                        mode='recursive', scale=True)
            dec_cell_attn = rnn.MultiRNNCell([rnn.GRUCell(self._dec_size),
                                              rnn.GRUCell(self._enc_size)], state_is_tuple=True)
            dec_cell_attn = seq2seq.AttentionWrapper(dec_cell_attn,
                                                     attn_mech,
                                                     attention_layer_size=self._enc_size,
                                                     alignment_history=alignment_history)
            dec_cell_out = rnn.GRUCell(self._output_size)
            dec_cell = rnn.MultiRNNCell([dec_cell_in1, dec_cell_in2, dec_cell_attn, dec_cell_out],
                                        state_is_tuple=True)

            dec = seq2seq.BasicDecoder(dec_cell, helper_build_fn(),
                                       dec_cell.zero_state(batch_size, self._dtype))

            dec_out, dec_state, _ = seq2seq.dynamic_decode(dec, output_time_major=False,
                    maximum_iterations=decoder_maxiters, impute_finished=True)

        self.outputs = dec_out.rnn_output
        self.output_ids = dec_out.sample_id
        self.final_state = dec_state

    def _output_onehot(self, ids):
        return tf.one_hot(ids, self._output_size, dtype=self._dtype)

    def train(self, batch_size, learning_rate=1e-4, out_help=False, time_discount=0.4, sampling_probability=0.2):
        """Build model for training.
        Args:
            batch_size: size of training batch
        """
        self.input_data = tf.placeholder(tf.int32, [batch_size, None], name='input_data')
        self.input_lengths = tf.placeholder(tf.int32, [batch_size], name='input_lengths')
        self.output_data = tf.placeholder(tf.int32, [batch_size, None], name='output_data')
        self.output_lengths = tf.placeholder(tf.int32, [batch_size], name='output_lengths')

        output_data_maxlen = tf.shape(self.output_data)[1]

        def infer_helper():
            return seq2seq.GreedyEmbeddingHelper(
                    self._output_onehot,
                    start_tokens=tf.fill([batch_size], self._output_sos_id),
                    end_token=self._output_eos_id)

        def train_helper():
            start_ids = tf.fill([batch_size, 1], self._output_sos_id)
            decoder_input_ids = tf.concat([start_ids, self.output_data], 1)
            decoder_inputs = self._output_onehot(decoder_input_ids)
            return seq2seq.ScheduledEmbeddingTrainingHelper(decoder_inputs, self.output_lengths,
                    self._output_onehot, sampling_probability)

        helper = train_helper if out_help else infer_helper

        self._build_model(batch_size, helper, decoder_maxiters=output_data_maxlen)

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

            # out_id = 2,3,4,5,6 : AA,AE,AH,AO,AW : reduce the cost by 20% for a-confusion
            data_is_a = tf.logical_and(tf.greater_equal(out_data_slice, 2),
                                       tf.less_equal(out_data_slice, 6))
            pred_is_a = tf.logical_and(tf.greater_equal(out_pred_slice, 2),
                                       tf.less_equal(out_pred_slice, 6))
            a_mask = tf.cast(tf.logical_and(data_is_a, pred_is_a), dtype=tf.float32)
            losses = losses * (1.0 - 0.2*a_mask)

            if time_discount > 0:
                # time discounts (only when using infer helper?)
                factors = tf.pow(tf.range(1, tf.to_float(output_maxlen + 1), dtype=tf.float32),
                                 -time_discount)
                losses = losses * tf.expand_dims(factors, 0)

            losses = tf.reduce_sum(losses, 1)
            self.losses = tf.reduce_sum(losses)
            tf.summary.scalar('losses', self.losses)

            inequality = tf.cast(tf.not_equal(self.output_ids, out_data_slice), dtype=tf.float32)
            # reduce inequality inaccuracy by 20% for a-confusion
            inequality = inequality * (1.0 - 0.1*a_mask)
            self.accuracy = tf.reduce_mean(1.0 - inequality)
            tf.summary.scalar('accuracy', tf.reduce_sum(self.accuracy))

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        decay_rate = tf.constant(0.8, dtype=tf.float64)
        self.learning_rate = learning_rate * tf.pow(decay_rate, tf.floor(self.global_step/4000))
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = opt.minimize(losses, global_step=self.global_step)

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

        self._build_model(1, infer_helper, decoder_maxiters=output_maxlen, alignment_history=True)

# Also See
#   https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/dw3Y2lnMAJc
