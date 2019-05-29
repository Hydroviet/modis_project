import math
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn import _should_cache, _is_keras_rnn_cell

class MTConvLSTMCell(rnn_cell_impl.RNNCell):
  """MTConvolutional LSTM recurrent network cell.
  """

  def __init__(self,
               conv_ndims,
               input_shape,
               output_channels,
               kernel_shape,
               use_bias=True,
               forget_bias=1.0,
               initializers=None,
               name="mtconv_lstm_cell"):
    """Construct MTConvLSTMCell.
    Args:
      conv_ndims: MTConvolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as int tuple, excluding the batch size.
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
      use_bias: (bool) Use bias in convolutions.
      skip_connection: If set to `True`, concatenate the input to the
        output of the conv LSTM. Default: `False`.
      forget_bias: Forget bias.
      initializers: Unused.
      name: Name of the module.
    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(MTConvLSTMCell, self).__init__(name=name)

    if conv_ndims != len(input_shape) - 1:
      raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
          input_shape, conv_ndims))

    self._g = 2
    self._T = [2**k for k in range(self._g)]
    self._conv_ndims = conv_ndims
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._use_bias = use_bias
    self._forget_bias = forget_bias

    self._total_output_channels = output_channels

    state_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._output_channels, self._g])
    self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
    self._output_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._total_output_channels, self._g])

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state, time, scope=None):
    cell, hidden = state
    cell = tf.unstack(cell, self._g, -1)
    hidden = tf.unstack(hidden, self._g, -1)
    new_cell = []
    output = []
    for k in range(self._g):
        cell_k = cell[k]
        hidden_k = hidden[k]
        if (time + 1) % self._T[k] == 0:
            new_hidden = _conv([inputs, hidden_k], self._kernel_shape,
                                2 * self._output_channels, self._use_bias, self._g, k)
            gates_1 = array_ops.split(
                value=new_hidden[0], num_or_size_splits=2, axis=self._conv_ndims + 1)
            gates_2 = array_ops.split(
                value=new_hidden[1], num_or_size_splits=2, axis=self._conv_ndims + 1)

            input_gate, new_input = gates_1
            forget_gate, output_gate = gates_2
            new_cell_k = math_ops.sigmoid(forget_gate + self._forget_bias) * cell_k
            new_cell_k += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
            output_k = math_ops.tanh(new_cell_k) * math_ops.sigmoid(output_gate)
        else:
            new_cell_k = cell_k
            output_k = hidden_k
        new_cell.append(tf.expand_dims(new_cell_k, axis=-1))
        output.append(tf.expand_dims(output_k, axis=-1))

    new_cell = array_ops.concat(axis=-1, values=new_cell)
    output = array_ops.concat(axis=-1, values=output)
    new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
    return output, new_state


class MTConv1DLSTMCell(MTConvLSTMCell):
  """1D MTConvolutional LSTM recurrent network cell.
  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self, name="conv_1d_lstm_cell", **kwargs):
    """Construct MTConv1DLSTM. See `MTConvLSTMCell` for more details."""
    super(MTConv1DLSTMCell, self).__init__(conv_ndims=1, name=name, **kwargs)


class MTConv2DLSTMCell(MTConvLSTMCell):
  """2D MTConvolutional LSTM recurrent network cell.
  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self, name="conv_2d_lstm_cell", **kwargs):
    """Construct MTConv2DLSTM. See `MTConvLSTMCell` for more details."""
    super(MTConv2DLSTMCell, self).__init__(conv_ndims=2, name=name, **kwargs)


class MTConv3DLSTMCell(MTConvLSTMCell):
  """3D MTConvolutional LSTM recurrent network cell.
  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self, name="conv_3d_lstm_cell", **kwargs):
    """Construct MTConv3DLSTM. See `MTConvLSTMCell` for more details."""
    super(MTConv3DLSTMCell, self).__init__(conv_ndims=3, name=name, **kwargs)


def _conv(args, filter_size, num_features, bias, g, k, bias_start=0.0):
  """MTConvolution.
  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias: Whether to use biases in the convolution layer.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  shape_length = len(shapes[0])
  for shape in shapes:
    if len(shape) not in [3, 4, 5]:
      raise ValueError("MTConv Linear expects 3D, 4D "
                       "or 5D arguments: %s" % str(shapes))
    if len(shape) != len(shapes[0]):
      raise ValueError("MTConv Linear expects all args "
                       "to be of same Dimension: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[-1]
  dtype = [a.dtype for a in args][0]

  # determine correct conv operation
  if shape_length == 3:
    conv_op = nn_ops.conv1d
    strides = 1
  elif shape_length == 4:
    conv_op = nn_ops.conv2d
    strides = shape_length * [1]
  elif shape_length == 5:
    conv_op = nn_ops.conv3d
    strides = shape_length * [1]

  # Now the computation."
  with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
    kernels = tf.get_variable(
        "kernels", [g] + filter_size + [shapes[0][-1], num_features], dtype=dtype)
    recurrent_kernels = tf.get_variable(
        "recurrent_kernels", [g, g] + filter_size + [shapes[1][-1], num_features], dtype=dtype)
    kernel = kernels[k]
    recurrent_kernel = tf.zeros_like(recurrent_kernels[0][0])
    for j in range(k + 1):
      recurrent_kernel += recurrent_kernels[j][k]

    res_1 = conv_op(
      args[0],
      kernel,
      strides,
      padding="SAME")
    res_2 = conv_op(
      args[1],
      recurrent_kernel,
      strides,
      padding="SAME")

    if not bias:
      return (res_1, res_2)
    bias_term = vs.get_variable(
      "biases", [num_features],
      dtype=dtype,
      initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return (res_1 + bias_term, res_2 + bias_term)


def static_rnn(cell,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
  """Creates a recurrent neural network specified by RNNCell `cell`.
  The simplest form of RNN network generated is:
  ```python
    state = cell.zero_state(...)
    outputs = []
    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)
    return (outputs, state)
  ```
  However, a few other options are available:
  An initial state can be provided.
  If the sequence_length vector is provided, dynamic calculation is performed.
  This method of calculation does not compute the RNN steps past the maximum
  sequence length of the minibatch (thus saving computational time),
  and properly propagates the state at an example's sequence length
  to the final state output.
  The dynamic calculation performed is, at time `t` for batch row `b`,
  ```python
    (output, state)(b, t) =
      (t >= sequence_length(b))
        ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
        : cell(input(b, t), state(b, t - 1))
  ```
  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a `Tensor` of shape
      `[batch_size, input_size]`, or a nested tuple of such elements.
    initial_state: (optional) An initial state for the RNN.
      If `cell.state_size` is an integer, this must be
      a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
      If `cell.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    sequence_length: Specifies the length of each sequence in inputs.
      An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
    scope: VariableScope for the created subgraph; defaults to "rnn".
  Returns:
    A pair (outputs, state) where:
    - outputs is a length T list of outputs (one for each input), or a nested
      tuple of such elements.
    - state is the final state
  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If `inputs` is `None` or an empty list, or if the input depth
      (column size) cannot be inferred from inputs via shape inference.
  """
  rnn_cell_impl.assert_like_rnncell("cell", cell)
  if not nest.is_sequence(inputs):
    raise TypeError("inputs must be a sequence")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with vs.variable_scope(scope or "rnn") as varscope:
    if _should_cache():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    # Obtain the first sequence of the input
    first_input = inputs
    while nest.is_sequence(first_input):
      first_input = first_input[0]

    # Temporarily avoid EmbeddingWrapper and seq2seq badness
    # TODO(lukaszkaiser): remove EmbeddingWrapper
    if first_input.get_shape().ndims != 1:

      input_shape = first_input.get_shape().with_rank_at_least(2)
      fixed_batch_size = input_shape[0]

      flat_inputs = nest.flatten(inputs)
      for flat_input in flat_inputs:
        input_shape = flat_input.get_shape().with_rank_at_least(2)
        batch_size, input_size = input_shape[0], input_shape[1:]
        fixed_batch_size.merge_with(batch_size)
        for i, size in enumerate(input_size):
          if size.value is None:
            raise ValueError(
                "Input size (dimension %d of inputs) must be accessible via "
                "shape inference, but saw value None." % i)
    else:
      fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

    if fixed_batch_size.value:
      batch_size = fixed_batch_size.value
    else:
      batch_size = array_ops.shape(first_input)[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, "
                         "dtype must be specified")
      if getattr(cell, "get_initial_state", None) is not None:
        state = cell.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype)
      else:
        state = cell.zero_state(batch_size, dtype)

    if sequence_length is not None:  # Prepare variables
      sequence_length = ops.convert_to_tensor(
          sequence_length, name="sequence_length")
      if sequence_length.get_shape().ndims not in (None, 1):
        raise ValueError(
            "sequence_length must be a vector of length batch_size")

      def _create_zero_output(output_size):
        # convert int to TensorShape if necessary
        size = _concat(batch_size, output_size)
        output = array_ops.zeros(
            array_ops.stack(size), _infer_state_dtype(dtype, state))
        shape = _concat(fixed_batch_size.value, output_size, static=True)
        output.set_shape(tensor_shape.TensorShape(shape))
        return output

      output_size = cell.output_size
      flat_output_size = nest.flatten(output_size)
      flat_zero_output = tuple(
          _create_zero_output(size) for size in flat_output_size)
      zero_output = nest.pack_sequence_as(
          structure=output_size, flat_sequence=flat_zero_output)

      sequence_length = math_ops.to_int32(sequence_length)
      min_sequence_length = math_ops.reduce_min(sequence_length)
      max_sequence_length = math_ops.reduce_max(sequence_length)

    # Keras RNN cells only accept state as list, even if it's a single tensor.
    is_keras_rnn_cell = _is_keras_rnn_cell(cell)
    if is_keras_rnn_cell and not nest.is_sequence(state):
      state = [state]
    for time, input_ in enumerate(inputs):
      if time > 0:
        varscope.reuse_variables()
      # pylint: disable=cell-var-from-loop
      call_cell = lambda: cell.call(input_, state, time)
      # pylint: enable=cell-var-from-loop
      if sequence_length is not None:
        (output, state) = _rnn_step(
            time=time,
            sequence_length=sequence_length,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            zero_output=zero_output,
            state=state,
            call_cell=call_cell,
            state_size=cell.state_size)
      else:
        (output, state) = call_cell()
      outputs.append(output)
    # Keras RNN cells only return state as list, even if it's a single tensor.
    if is_keras_rnn_cell and len(state) == 1:
      state = state[0]

    return (outputs, state)
