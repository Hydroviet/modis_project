# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.recurrent import Recurrent

import numpy as np
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces
import tensorflow as tf


class GraphRecurrent(Recurrent):
    """Abstract base class for recurrent layers.

    Do not use in a model -- it's not a functional layer!

    # Arguments
        units: Integer, the dimensionality of the output space
            (i.e. the number output filters in the convolution).
        graph_tensor: A tensor of shape [K_adjacency_power, num_graph_nodes, num_graph_nodes],
            containing graph convolution/filter matrices.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, rocess the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.

    # Input shape
        4D tensor with shape `(num_samples, timesteps, num_nodes, input_dim)`.

    # Output shape
        - if `return_sequences`: 4D tensor with shape
            `(num_samples, timesteps, num_nodes, output_dim/units)`.
        - else, 3D tensor with shape `(num_samples, num_nodes, output_dim/units)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.
        **Note:** for the time being, masking is only supported with Theano.

    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.

        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                a `batch_input_size=(...)` to the first layer in your model.
                This is the expected shape of your inputs *including the batch
                size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.

        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    """

    def __init__(self, units,
                 graph_tensor,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 **kwargs):

        super(GraphRecurrent, self).__init__(**kwargs)

        self.units = units  # hidden_units or dimensionality of the output space.

        self.poly_degree = graph_tensor.shape[0] - 1  # adjacecny power degree
        self.num_nodes = graph_tensor.shape[2]  # num nodes in a graph
        graph_tensor = K.constant(graph_tensor, dtype=K.floatx())
        self.graph_tensor = graph_tensor  # output_shape = [K, input_dim, input_dim]

        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.input_spec = [InputSpec(ndim=3)]
        self.state_spec = None

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.units)  # output_shape = [num_samples, timesteps, output_dim]
        else:
            output_shape = (input_shape[0], self.units)  # output_shape = [num_samples, output_dim]

        if self.return_state:
            state_shape = [(input_shape[0], input_shape[1], self.units) for _ in self.states]
            return [output_shape] + state_shape  # output_shape, state_shape_hidden, state_shape_cell
        else:
            return output_shape

    def get_config(self):
        config = {'units': self.units,
                  'graph_tensor': self.graph_tensor,
                  'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful}
        base_config = super(GraphRecurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GraphLSTM(GraphRecurrent):
    def __init__(self, units,
                 graph_tensor,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(GraphLSTM, self).__init__(units, 
                                        graph_tensor,
                                        return_sequences=return_sequences,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        **kwargs)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self.state_spec = [InputSpec(shape=(None, self.units)),
                           InputSpec(shape=(None, self.units))]

    def build(self, input_shape):
        # input_shape = [num_samples, timesteps, num_nodes, input_dim]
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[-1]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

        self.states = [None, None]  # initial states: two zero tensors of shape (num_samples, units)
        if self.stateful:
            self.reset_states()

        kernel_shape = ((self.poly_degree + 1) * self.input_dim, self.units * 4)
        self.kernel_shape = kernel_shape  # output_shape = [input_dim * K, output_dim * 4]
        recurrent_kernel_shape = ((self.poly_degree + 1) * self.units, self.units * 4)
        self.recurrent_kernel_shape = recurrent_kernel_shape  # output_shape = [output_dim * K, output_dim * 4]

        self.kernel = self.add_weight(shape=kernel_shape,
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def get_initial_state(self, inputs):
        # get states for all-zero tensor input of shape (samples, output_dim)

        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        shape = list(self.kernel_shape)
        shape[-1] = self.units

        initial_state = self.input_op(initial_state, K.zeros(tuple(shape)))   # (samples, output_dim)
        initial_states = [initial_state for _ in range(2)]

        return initial_states

    def reset_states(self):

        if not self.stateful:
            raise RuntimeError('Layer must be stateful.')

        input_shape = self.input_spec[0].shape
        output_shape = self.compute_output_shape(input_shape)

        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size). '
                             'Got input shape: ' + str(input_shape))

        units = output_shape[-1]

        if hasattr(self, 'states'):
            K.set_value(self.states[0], np.zeros((input_shape[0], units)))
            K.set_value(self.states[1], np.zeros((input_shape[0], units)))
        else:
            self.states = [K.zeros((input_shape[0], units)),
                           K.zeros((input_shape[0], units))]

    def get_constants(self, inputs, training=None):
        constants = []
        if 0 < self.dropout < 1:
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones += 1

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.recurrent_dropout < 1:
            shape = list(self.kernel_shape)
            shape[-1] = self.units
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones = self.input_conv(ones, K.zeros(shape))
            ones += 1.

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants


    def input_op(self, x, w, b=None):
        # x = [num_samples, input_dim]
        # w = [input_dim * K, output_dim]
        # graph_tensor = [K, input_dim, input_dim]

        op_out = K.dot(self.graph_tensor, x)  # output_shape = [K, input_dim, num_samples]
        op_out = tf.transpose(op_out, perm=[2, 0, 1])  # output_shape = [num_samples, K, input_dim]
        op_out_shape = op_out.get_shape().as_list()
        op_out = K.reshape(op_out, shape=(-1, op_out_shape[1] * op_out_shape[2]))  # output_shape = [num_samples, num_nodes, input_dim * K]

        op_out = K.dot(op_out, w)  # output_shape = [num_samples, num_nodes, output_dim]

        if b is not None:
            op_out = K.bias_add(op_out, b)
        return op_out  # output_shape = [num_samples, num_nodes, output_dim]

    def recurrent_op(self, x, w):
        # x = [num_samples, num_nodes, output_dim]
        # w = [output_dim * K, output_dim]
        # graph_tensor = [K, num_nodes, num_nodes]

        op_out = K.dot(self.graph_tensor, x)
        op_out = tf.transpose(op_out, perm=[2, 0, 1])
        op_out_shape = op_out.get_shape().as_list()
        op_out = K.reshape(op_out, shape=(-1, op_out_shape[1] * op_out_shape[2]))

        op_out = K.dot(op_out, w)

        return op_out  # output_shape = [num_samples, num_nodes, output_dim]

    def step(self, inputs, states):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        x_i = self.input_op(inputs_i, self.kernel_i, self.bias_i)
        x_f = self.input_op(inputs_f, self.kernel_f, self.bias_f)
        x_c = self.input_op(inputs_c, self.kernel_c, self.bias_c)
        x_o = self.input_op(inputs_o, self.kernel_o, self.bias_o)

        h_i = self.recurrent_op(h_tm1_i, self.recurrent_kernel_i)
        h_f = self.recurrent_op(h_tm1_f, self.recurrent_kernel_f)
        h_c = self.recurrent_op(h_tm1_c, self.recurrent_kernel_c)
        h_o = self.recurrent_op(h_tm1_o, self.recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)

        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(GraphLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
