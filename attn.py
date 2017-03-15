import tensorflow as tf
from tensorflow.contrib import rnn

class AttentionalInterface:
    def __init__(self, inputs, state_size, layers=[]):
        """Create an attentional interface which can be used to compute
        attentional context tensor given the hidden state.
        Args:
          inputs: input tensor of size [batch_size x sequence_length x input_size]
                  on which attention is implemented.
          state_size:
        """

# Copy-pasted from tensorflow repository: /contrib/rnn/python/ops/rnn_cell.py
class AttentionCellWrapper(rnn.RNNCell):
    """Basic attention cell wrapper.
    Implementation based on https://arxiv.org/abs/1409.0473.
    """

    def __init__(self, cell, attn_length, attn_size=None, attn_vec_size=None,
                 input_size=None, reuse=None):
        """Create a cell with attention.
        Args:
          cell: an RNNCell, an attention is added to it.
          attn_length: integer, the size of an attention window.
          attn_size: integer, the size of an attention vector. Equal to
              cell.output_size by default.
          input_size: integer, the size of a hidden linear layer,
              built from inputs and attention. Derived from the input tensor
              by default.
          reuse: (optional) Python boolean describing whether to reuse variables
              in an existing scope.  If not `True`, and the existing scope already has
              the given variables, an error is raised.
        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if attn_length is zero or less.
        """
        if not isinstance(cell, rnn.RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        #if nest.is_sequence(cell.state_size) and not state_is_tuple:
        #    raise ValueError("Cell returns tuple of states, but the flag "
        #                     "state_is_tuple is not set. State size is: %s"
        #                     % str(cell.state_size))
        if attn_length <= 0:
            raise ValueError("attn_length should be greater than zero, got %s"
                             % str(attn_length))
        if attn_size is None:
            attn_size = cell.output_size
        if attn_vec_size is None:
            attn_vec_size = attn_size
        self._cell = cell
        self._attn_vec_size = attn_vec_size
        self._input_size = input_size
        self._attn_size = attn_size
        self._attn_length = attn_length
        self._reuse = reuse

    @property
    def state_size(self):
        size = (self._cell.state_size, self._attn_size,
                self._attn_size * self._attn_length)
        return size

    @property
    def output_size(self):
        return self._attn_size

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell with attention (LSTMA)."""
        with tf.variable_scope(scope or "attention_cell_wrapper", reuse=self._reuse):
            state, attns = state
            attn_states = tf.reshape(attn_states,
                                     [-1, self._attn_length, self._attn_size])
            input_size = self._input_size
            if input_size is None:
                input_size = inputs.get_shape().as_list()[1]
            inputs = _linear([inputs, attns], input_size, True)
            lstm_output, new_state = self._cell(inputs, state)
            #if self._state_is_tuple:
            #    new_state_cat = tf.concat(nest.flatten(new_state), 1)
            #else:
            #    new_state_cat = new_state
            #new_attns, new_attn_states = self._attention(
            #    new_state_cat, attn_states)
            with tf.variable_scope("attn_output_projection"):
                output = _linear([lstm_output, new_attns],
                                 self._attn_size, True)
            new_attn_states = tf.concat(
                [new_attn_states, tf.expand_dims(output, 1)], 1)
            new_attn_states = tf.reshape(
                new_attn_states, [-1, self._attn_length * self._attn_size])
            new_state = (new_state, new_attns, new_attn_states)
            return output, new_state

    def _attention(self, query, attn_states):
        conv2d = tf.nn.conv2d
        reduce_sum = tf.reduce_sum
        softmax = tf.nn.softmax
        tanh = tf.tanh

        with tf.variable_scope("attention"):
            k = tf.get_variable(
                "attn_w", [1, 1, self._attn_size, self._attn_vec_size])
            v = tf.get_variable("attn_v", [self._attn_vec_size])
            hidden = tf.reshape(attn_states,
                                [-1, self._attn_length, 1, self._attn_size])
            hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            y = _linear([query], self._attn_vec_size, True)
            y = tf.reshape(y, [-1, 1, 1, self._attn_vec_size])
            s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
            a = softmax(s)
            d = reduce_sum(
                tf.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
            new_attns = tf.reshape(d, [-1, self._attn_size])
            new_attn_states = tf.slice(attn_states, [0, 1, 0], [-1, -1, -1])
            return new_attns, new_attn_states


def _linear(args, output_size, bias, bias_start=0.0):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    #if args is None or (nest.is_sequence(args) and not args):
    #    raise ValueError("`args` must be specified")
    #if not nest.is_sequence(args):
    #    args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable(
            "weights", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = tf.get_variable(
                "biases", [output_size], dtype=dtype,
                initializer=tf.constant_initializer(bias_start, dtype=dtype))
        return tf.nn.bias_add(res, biases)
