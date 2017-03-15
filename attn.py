import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.util import nest

class AttentionalInterface:
    """Abstract object representing an attentional interface, which can be used
    to compute context tensor from a query.
    """
    def __call__(self, query, scope=None):
        """
        Args:
          query: tensor of shape [B, ...] using which context is computed.
              (possibly the current state of the decoder)
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        raise NotImplementedError("Abstract property")

# see: https://theneuralperspective.com/2016/11/20/recurrent-neural-network-rnn-part-4
class BasicAttentionalInterface(AttentionalInterface):
    """
    Args:
      values: tensor of shape [B, T, ...] on which attention is implemented.
      values_length: int32 tensor of shape [B] indicating the true length of
          the sequences.
      layers: number of layers of the feed forward network that is used to
          compute the context vector. By default the inputs and query are
          directly transformed into the context vector terms using a linear
          transformation step.
    """
    def __init__(self, values, values_length, layers=[]):
        self._values = values
        self._values_length = values_length
        self._layers = layers

    @property
    def output_size(self):
        return tf.shape(self._values)[2:]

class AttentionCellWrapper(rnn.RNNCell):
    """Basic attention cell wrapper.
    Implementation based on https://arxiv.org/abs/1409.0473.
    """

    def __init__(self, cell, attn_ifx, reuse=None):
        """Create a cell with attention.
        Args:
          cell: an RNNCell, an attention is added to it.
          attn_ifx: an AttentionalInterface object
          reuse: (optional) Python boolean describing whether to reuse variables
              in an existing scope.  If not `True`, and the existing scope already has
              the given variables, an error is raised.
        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if attn_length is zero or less.
        """
        if not isinstance(cell, rnn.RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        if not isinstance(attn_ifx, AttentionalInterface):
            raise TypeError("The parameter attn_ifx is not AttentionalInterface.")
        self._cell = cell
        self._attn_ifx = attn_ifx
        self._reuse = reuse

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell with attention (LSTMA)."""
        with tf.variable_scope(scope or "attention_cell_wrapper", reuse=self._reuse):
            attn_ctx = self._attn_ifx(state)
            inputs = tf.concat([inputs, attn_ctx], 1)
            output, new_state = self._cell(inputs, state)
            return output, new_state

    def _attention(self, query, attn_states):
        conv2d = tf.nn.conv2d

        with tf.variable_scope("attention"):
            k = tf.get_variable(
                "attn_w", [1, 1, self._attn_size, self._attn_vec_size])
            v = tf.get_variable("attn_v", [self._attn_vec_size])
            hidden = tf.reshape(attn_states,
                                [-1, self._attn_length, 1, self._attn_size])
            hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            #TODO figure out the shape of hidden_features


# Copy-pasted from tensorflow repository: /contrib/rnn/python/ops/rnn_cell.py
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
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

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
