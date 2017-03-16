import abc, six
import tensorflow as tf
from tensorflow.contrib import rnn, layers
from tensorflow.python.util import nest

@six.add_metaclass(abc.ABCMeta)
class AttentionalInterface1D:
    """Abstract object representing an attentional interface on a sequence of
    values. The attention should be implemented along the second dimension of
    values parameter (ie. T in [B, T, ...]).
    """
    def __init__(self, values, values_length=None, reuse=None):
        """
        Args:
          values: A tensor of shape [B, T, ...].
          values_length: int32 tensor of shape [B] indicating the true lengths
              of the sequences. If None, the entire T is used.
        """
        self._values = values
        self._values_length = values_length
        self._dtype = values.dtype
        self._reuse = reuse

    @abc.abstractmethod
    def __call__(self, query, scope=None):
        """
        Args:
          query: a tensor, of shape [B, Q] and same type as values, using which
              attentional context is computed.
        Returns:
          context_vector: the final context vector
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Returns [B, V]."""
        vshape = tf.shape(self._values)
        return vshape[:1] + vshape[2:]


class BasicAttentionalInterface(AttentionalInterface1D):
    """Attentional interface based on https://arxiv.org/abs/1409.0473.
    """
    def __init__(self, values, values_length, activation_fn=tf.tanh, layers=[], reuse=None):
        """
        Args:
          values: tensor of shape [B, T, V] on which attention is implemented.
          values_length: int32 tensor of shape [B] indicating the true length of
              the sequences. If None, the entire T is used.
          layers: number of layers of the feed forward network that is used to
              compute the context vector. By default the inputs and query are
              directly transformed into the context vector terms using a linear
              transformation step.
          activation_fn: activation function to use for each layer in layers.
          reuse: (optional) Python boolean describing whether to reuse variables
              in an existing scope. If not `True`, and the existing scope
              already has the given variables, an error is raised.
        """
        super(BasicAttentionalInterface, self).__init__(values, values_length, reuse=reuse)
        self._layers = layers

    def __call__(self, query, scope=None):
        with tf.variable_scope(scope, "basic_attentional_interface", reuse=self._reuse):
            values_maxlen = tf.shape(self._values)[1]
            query = tf.expand_dims(query, 1)
            query = tf.tile(query, [1, values_maxlen, 1])
            last_outputs = tf.concat([self._values, query], 2)
            for (i, num_outputs) in enumerate(self._layers):
                last_outputs = activation_fn(_linear_seq(last_outputs, num_outputs,
                                             name="attn-ifx-layer{0}".format(i)))
            scores = tf.squeeze(_linear_seq(last_outputs, 1, name="attn-ifx-score"), [2])
            if self._values_length is not None:
                scores_mask = tf.sequence_mask(self._values_length,
                                               maxlen=values_maxlen,
                                               dtype=self._dtype)
                scores = scores * scores_mask
            scores_norm = tf.nn.softmax(scores, name="scores-softmax")
            scores_norm = tf.expand_dims(scores_norm, 2)

            context = scores_norm * self._values # broadcast and multiply element-wise
            context = tf.reduce_sum(context, 1, name="context")
            return context


class AttentionCellWrapper(rnn.RNNCell):
    """Attention cell wrapper.
    """

    def __init__(self, cell, attn_ifx, reuse=None):
        """Create a cell with attention.
        Args:
          cell: an RNNCell, an attention is added to it.
          attn_ifx: an AttentionalInterface1D object
          reuse: (optional) Python boolean describing whether to reuse variables
              in an existing scope.  If not `True`, and the existing scope
              already has the given variables, an error is raised.
        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if attn_length is zero or less.
        """
        if not isinstance(cell, rnn.RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        if not isinstance(attn_ifx, AttentionalInterface1D):
            raise TypeError("The parameter attn_ifx is not AttentionalInterface1D.")
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
        with tf.variable_scope(scope or "attention_cell_wrapper", reuse=self._reuse):
            if nest.is_sequence(self._cell.state_size):
                query = tf.concat(nest.flatten(state), 1)
            else:
                query = state
            attn_ctx = self._attn_ifx(query)
            inputs = tf.concat([inputs, attn_ctx], 1)
            output, new_state = self._cell(inputs, state)
            return output, new_state


def _linear_seq(inputs, out_channels, name=None):
    """Apply a linear transformation on a sequence, using 1D convolution with
    unit filter-width and strides.
    Args:
      inputs: a 3D Tensor of size [B, T, C], Tensors.
      out_channels: int, third dimension of outputs.
    Returns:
      A 3D Tensor with shape [B, T, out_channels]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    in_channels = inputs.shape[2]
    dtype = inputs.dtype

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        filters = tf.get_variable(
            "filters", [1, in_channels, out_channels], dtype=dtype)
        res = tf.nn.conv1d(inputs, filters, 1, 'SAME', name=name)
        biases = tf.get_variable(
            "biases", [out_channels], dtype=dtype,
            initializer=tf.constant_initializer(0.0, dtype=dtype))
        # bias_add broadcasts biases automatically across higher dimensions
        return tf.nn.bias_add(res, biases)
