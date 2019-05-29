import sys
import keras
import tensorflow as tf
from keras import backend as K
from keras.optimizers import adam
from keras.models import Model, Input
from keras.layers import Layer, RNN, GRUCell, GRU, LSTM
from keras.layers import deserialize as deserialize_layer
from keras import activations, initializers, regularizers, constraints

def tf_print(tensor, message=None):
    def print_message(x):
        sys.stdout.write(message + " %s\n" % x)
        return x

    prints = [tf.py_func(print_message, [tensor], tensor.dtype)]
    with tf.control_dependencies(prints):
        op = tensor + 1
        op = tf.identity(op)
    return op

# GRUCellKeepDim
class GRUCellKeepDim(GRUCell):
    def __init__(self, n_dims, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 reset_after=False,
                 **kwargs):
        super(GRUCellKeepDim, self).__init__(units, activation, recurrent_activation,
                                             use_bias, kernel_initializer, recurrent_initializer, bias_initializer,
                                             kernel_regularizer, recurrent_regularizer, bias_regularizer,
                                             kernel_constraint, recurrent_constraint, bias_constraint,
                                             dropout, recurrent_dropout, implementation, reset_after, **kwargs)
        self.n_dims = n_dims

    def build(self, input_shape):
        self.last_kernel = self.add_weight(shape=(self.units, self.n_dims),
                                           name='last_kernel',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        super(GRUCellKeepDim, self).build(input_shape)

    def call(self, inputs, states, training=None):
        output, states = super(GRUCellKeepDim, self).call(inputs, states, training)
        output = K.dot(output, self.last_kernel)
        return output, states

    @property
    def state_size(self):
        return self.units

    def state_size(self, state_size):
        self.state_size = state_size

# GRUKeepDim
class GRUKeepDim(GRU):
    def __init__(self, n_dims, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=False,
                 **kwargs):
        cell = GRUCellKeepDim(n_dims, units,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              recurrent_constraint=recurrent_constraint,
                              bias_constraint=bias_constraint,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout,
                              implementation=implementation,
                              reset_after=reset_after)
        RNN.__init__(self, cell,
                     return_sequences=return_sequences,
                     return_state=return_state,
                     go_backwards=go_backwards,
                     stateful=stateful,
                     unroll=unroll,
                     **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if hasattr(self.cell.state_size, '__len__'):
            state_size = self.cell.state_size
        else:
            state_size = [self.cell.state_size]
        output_dim = self.cell.n_dims

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], output_dim)
        else:
            output_shape = (input_shape[0], output_dim)

        if self.return_state:
            state_shape = [(input_shape[0], dim) for dim in state_size]
            return [output_shape] + state_shape
        else:
            return output_shape

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(GRUKeepDim, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state)

    @property
    def state_size(self):
        return self.cell.state_size

class GRU_1(GRU):
    @property
    def state_size(self):
        return self.units

    def state_size(self, state_size):
        self.state_size = state_size

class LSTM_1(LSTM):
    @property
    def state_size(self):
        return self.units

    def state_size(self, state_size):
        self.state_size = state_size


class GRNNCell(Layer):
    def __init__(cells, **kwargs):
        super(GRNNCell, self).__init__(**kwargs)
        self.cells = cells

    def __init__(self, cells, **kwargs):
        for cell in cells:
            if not hasattr(cell, 'call'):
                raise ValueError('All cells must have a `call` method. '
                                 'received cells:', cells)
            if not hasattr(cell, 'state_size'):
                raise ValueError('All cells must have a '
                                 '`state_size` attribute. '
                                 'received cells:', cells)
        self.cells = cells
        super(GRNNCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        state_size = []
        for cell in self.cells:
            if hasattr(cell.state_size, '__len__'):
                state_size += list(cell.state_size)
            else:
                state_size.append(cell.state_size)
        return tuple(state_size)

    def call(self, inputs, states, constants=None, **kwargs):
        # Recover per-cell states.
        nested_states = []
        for cell in self.cells:
            if hasattr(cell.state_size, '__len__'):
                nested_states.append(states[:len(cell.state_size)])
                states = states[len(cell.state_size):]
            else:
                nested_states.append([states[0]])
                states = states[1:]

        # Call the cells in order and store the returned states.
        new_nested_states = []
        for cell, states in zip(self.cells, nested_states):
            if has_arg(cell.call, 'constants'):
                inputs, states = cell.call(inputs, states,
                                           constants=constants,
                                           **kwargs)
            else:
                inputs, states = cell.call(inputs, states, **kwargs)
            new_nested_states.append(states)

        # Format the new states as a flat list
        # in reverse cell order.
        states = []
        for cell_states in new_nested_states:
            states += cell_states
        return inputs, states

    def build(self, input_shape):
        timesteps = input_shape[1]
        n_dims=input_shape[-1]
        cell_input_shape = (None, timesteps, n_dims)
        for cell in self.cells:
            if isinstance(cell, Layer):
                cell.build(cell_input_shape)
        self.built = True

    def get_config(self):
        cells = []
        for cell in self.cells:
            cells.append({'class_name': cell.__class__.__name__,
                          'config': cell.get_config()})
        config = {'cells': cells}
        base_config = super(GRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        cells = []
        for cell_config in config.pop('cells'):
            cells.append(deserialize_layer(cell_config,
                                           custom_objects=custom_objects))
        return cls(cells, **config)

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                weights += cell.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                weights += cell.non_trainable_weights
        if not self.trainable:
            trainable_weights = []
            for cell in self.cells:
                if isinstance(cell, Layer):
                    trainable_weights += cell.trainable_weights
            return trainable_weights + weights
        return weights

    def get_weights(self):
        """Retrieves the weights of the model.

        # Returns
            A flat list of Numpy arrays.
        """
        weights = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                weights += cell.weights
        return K.batch_get_value(weights)

    def set_weights(self, weights):
        """Sets the weights of the model.

        # Arguments
            weights: A list of Numpy arrays with shapes and types matching
                the output of `model.get_weights()`.
        """
        tuples = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                num_param = len(cell.weights)
                weights = weights[:num_param]
                for sw, w in zip(cell.weights, weights):
                    tuples.append((sw, w))
                weights = weights[num_param:]
        K.batch_set_value(tuples)

    @property
    def losses(self):
        losses = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                cell_losses = cell.losses
                losses += cell_losses
        return losses

    def get_losses_for(self, inputs=None):
        losses = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                cell_losses = cell.get_losses_for(inputs)
                losses += cell_losses
        return losses


# GRNN
class GRNN(Layer):
    def __init__(self, n_nodes, n_dims, n_hiddens,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 return_sequences=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 keep_dims=True,
                 cell_type='gru',
                 **kwargs):
        super(GRNN, self).__init__(**kwargs)
        self.n_nodes = n_nodes
        self.n_dims = n_dims
        self.n_hiddens = n_hiddens

        cells = []
        if cell_type == 'gru':
            if keep_dims:
                for i in range(n_nodes):
                    cell = GRUKeepDim(n_dims=n_dims, units=n_hiddens,
                                      activation=activation,
                                      recurrent_activation=recurrent_activation,
                                      return_state=True,
                                      return_sequences=return_sequences,
                                      dropout=dropout,
                                      recurrent_dropout=recurrent_dropout)
                    cells.append(cell)
            else:
                for i in range(n_nodes):
                    cell = GRU_1(units=n_hiddens,
                                 activation=activation,
                                 recurrent_activation=recurrent_activation,
                                 return_state=True,
                                 return_sequences=return_sequences,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout)
                    cells.append(cell)
        else:
            for i in range(n_nodes):
                cell = LSTM_1(units=n_hiddens,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              return_state=True,
                              return_sequences=return_sequences,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout)
                cells.append(cell)


        self.return_sequences = return_sequences
        self.grnn_cell = GRNNCell(cells)
        self.h_state = None
        self.trainable = True
        self.keep_dims = keep_dims
        self._num_constants = None

    def build(self, input_shape):
        # input_shape[0] = (T, N, D)
        # h_state = (T, D/H, N)
        input_shape = input_shape[0]
        self.timesteps = input_shape[1]
        cell_input_shape = (None, self.timesteps, self.n_dims)
        self.grnn_cell.build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        # output_shape[0] = (N, D/H)
        input_shape = input_shape[0]
        cell_input_shape = (input_shape[0], input_shape[-1])
        cell_output_shape = self.grnn_cell.cells[0].compute_output_shape(cell_input_shape)
        output_shape = cell_output_shape[0]
        if self.return_sequences:
            return (None, output_shape[0], self.n_nodes, output_shape[-1]) # (T, N, H)
        else:
            return (None, self.n_nodes, output_shape[-1]) # (N, H)

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (B, T, H, N)
        # inputs : (B, T, N, D)
        initial_state = K.zeros_like(inputs)  # (B, T, N, D)
        initial_state = K.sum(initial_state, axis=(1, -1))  # (B, N,)
        initial_state = K.expand_dims(initial_state, axis=1)  # (B, 1, N)
        cell = self.grnn_cell.cells[0].cell
        state_size = cell.state_size
        if hasattr(state_size, '__len__'):
            return [K.tile(initial_state, [1, dim, 1]) for dim in state_size]
        else:
            return K.tile(initial_state, [1, state_size, 1])

    def call(self, x, training=None):
        # A = (B, N, N)
        x_main = x[0]
        A = x[1]

        if self.h_state is None:
            self.h_state = self.get_initial_state(x_main) # (B, D/H, N), (B, D/H, N) x (B, N, N) = (B, D/H, N)
        S = bmm(self.h_state, A, axes=(2,1)) # S: (B, D/H, N)

        #TODO: add tf.while_loop
        O_list = []
        H_list = []
        for n in range(self.n_nodes):
            cell = self.grnn_cell.cells[n]
            x_n = x_main[:, :, n, :]
            S_n = get_s(S, n)
            cell_output = cell.call(x_n,
                    initial_state=S_n,
                    training=training)

            O = cell_output[0] # O = (B, D/H), return_sequences: (B, T, D/H)
            H = cell_output[1] # H = (B, T, D/H)

            O_list.append(K.expand_dims(O, axis=-2))
            H_list.append(K.expand_dims(H, axis=-1))

        O = K.concatenate(O_list, axis=-2)
        if training:
            self.h_state = concatenate(H_list, axis=-1)
        return O

    @property
    def states(self):
        if self.h_state is None:
            if isinstance(self.grnn_cell.state_size, int):
                num_states = 1
            else:
                num_states = len(self.grnn_cell.state_size)
            #return [None for _ in range(num_states)]
            return None
        return self.h_state

    @states.setter
    def states(self, h_state):
        self.h_state = h_state

    def get_config(self):
        config = {'return_sequences': self.return_sequences}
        if self._num_constants is not None:
            config['num_constants'] = self._num_constants

        cell_config = self.grnn_cell.get_config()
        config['grnn_cell'] = {'class_name': self.grnn_cell.__class__.__name__,
                               'config': cell_config}
        base_config = super(GRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        cell = deserialize_layer(config.pop('grnn_cell'),
                                 custom_objects=custom_objects)
        num_constants = config.pop('num_constants', None)
        layer = cls(cell, **config)
        layer._num_constants = num_constants
        return layer

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        if isinstance(self.grnn_cell, Layer):
            return self.grnn_cell.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
        if isinstance(self.grnn_cell, Layer):
            if not self.trainable:
                return self.grnn_cell.weights
            return self.grnn_cell.non_trainable_weights
        return []

    @property
    def losses(self):
        layer_losses = super(GRNN, self).losses
        if isinstance(self.grnn_cell, Layer):
            return self.grnn_cell.losses + layer_losses
        return layer_losses

    def get_losses_for(self, inputs=None):
        if isinstance(self.grnn_cell, Layer):
            cell_losses = self.grnn_cell.get_losses_for(inputs)
            return cell_losses + super(GRNN, self).get_losses_for(inputs)
        return super(GRNN, self).get_losses_for(inputs)

def bmm(x, A, axes):
    if not isinstance(x, list):
        return K.batch_dot(x, A, axes=axes)
    else:
        return [K.batch_dot(xi, A, axes=axes) for xi in x]

def get_s(S, n):
    if not isinstance(S, list):
        return [S[:, :, n]]
    else:
        return [Si[:,:,n] for Si in S]

def concatenate(H, axis):
    if not isinstance(H, list):
        return K.concatenate(H, axis=axis)
    else:
        return [K.concatenate(Hi, axis=axis) for Hi in H]
