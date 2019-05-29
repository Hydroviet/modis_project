import keras.backend as K
from keras.layers import Layer, InputSpec

class RNN(Layer):
    def __init__(self, cell,
                 **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.cell = cell
        self.input_spec = [InputSpec(ndim=3)]
        self.state_spec = None
        self._states = None
        self.constants_spec = None

    def compute_output_shape(self, input_shape):
        if hasattr(self.cell.state_size, '__len__'):
            state_size = self.cell.state_size
        else:
            state_size = [self.cell.state_size]
        output_dim = state_size[0]

        output_shape = (input_shape[0], output_dim)

        if self.return_state:
            state_shape = [(input_shape[0], dim) for dim in state_size]
            return [output_shape] + state_shape
        else:
            return output_shape

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = None
        ipnut_dim = input_shape[-1]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, input_dim))
        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            self.cell.build(step_input_shape)

        if hasattr(self.cell.state_size, '__len__'):
            state_size = self.cell.state_size
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if [spec.shape[-1] for spec in self.state_spec] != state_size:
                raise ValueError(
                    'An `initial_state` was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'however `cell.state_size` is '
                    '{}'.format(self.state_spec, self.cell.state_size))
        else:
            self.state_spec = [InputSpec(shape=(None, dim))
                               for dim in state_size]
        self.built = True


   def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]

    def __call__(self, inputs, **kwargs):
        return super(RNN, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        initial_state = self.get_initial_state(inputs)
        input_shape = K.shape(inputs)
        timesteps = input_shape[1]
        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        def step(inputs, states):
            return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = K.rnn(step,
                                             inputs,
                                             initial_state,
                                             input_lengths=timesteps)
        return last_output

    def get_config(self):
        #if self._num_constants is not None:
        #    config['num_constants'] = self._num_constants

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(RNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from . import deserialize as deserialize_layer
        cell = deserialize_layer(config.pop('cell'),
                                 custom_objects=custom_objects)
        num_constants = config.pop('num_constants', None)
        layer = cls(cell, **config)
        layer._num_constants = num_constants
        return layer

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        if isinstance(self.cell, Layer):
            return self.cell.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
        if isinstance(self.cell, Layer):
            if not self.trainable:
                return self.cell.weights
            return self.cell.non_trainable_weights
        return []

    @property
    def losses(self):
        layer_losses = super(RNN, self).losses
        if isinstance(self.cell, Layer):
            return self.cell.losses + layer_losses
        return layer_losses

    def get_losses_for(self, inputs=None):
        if isinstance(self.cell, Layer):
            cell_losses = self.cell.get_losses_for(inputs)
            return cell_losses + super(RNN, self).get_losses_for(inputs)
        return super(RNN, self).get_losses_for(inputs)


class GRUCell(Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 **kwargs):
        super(GRUCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel')

        bias_shape = (3 * self.units,)
        self.bias = self.add_weight(shape=bias_shape,
                                    name='bias')
        self.input_bias, self.recurrent_bias = self.bias, None

        # update gate
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        # reset gate
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        # new gate
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        self.input_bias_z = self.input_bias[:self.units]
        self.input_bias_r = self.input_bias[self.units: self.units * 2]
        self.input_bias_h = self.input_bias[self.units * 2:]

        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

        x_z = K.dot(inputs_z, self.kernel_z)
        x_r = K.dot(inputs_r, self.kernel_r)
        x_h = K.dot(inputs_h, self.kernel_h)

        x_z = K.bias_add(x_z, self.input_bias_z)
        x_r = K.bias_add(x_r, self.input_bias_r)
        x_h = K.bias_add(x_h, self.input_bias_h)

        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

        recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel_z)
        recurrent_r = K.dot(h_tm1_r, self.recurrent_kernel_r)

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        recurrent_h = K.dot(r * h_tm1_h, self.recurrent_kernel_h)
        hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation)}
        base_config = super(GRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRU(RNN):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 **kwargs):
        cell = GRUCell(units,
                       activation=activation,
                       recurrent_activation=recurrent_activation)
        super(GRU, self).__init__(cell, **kwargs)

    def call(self, inputs, training=None):
        super(GRU, self).call(inputs, training=training)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation)}
        base_config = super(GRU, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
