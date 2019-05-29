import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, orthogonal_initializer


class ConvLSTM(object):

    def __init__(self, is_training, data_format, batch_norm_decay, batch_norm_epsilon):
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._is_training = is_training
        assert data_format in ('channels_first', 'channels_last')
        self._data_format = data_format
        self.filters = [16, 8, 1]
        self.in_steps = 14
        self.out_steps = 12
        self.input_shape = [512, 512, 1]
        self.keep_rate = 1.0

    def forward_pass(self, x):
        x = tf.unstack(x, self.in_steps, 1)
        filters = [self.input_shape[-1]] + self.filters
        for i in range(1, len(filters)):
            filter = filters[i]
            input_shape = self.input_shape[:-1] + [filters[i - 1]]
            convLSTM2D_cell = rnn.Conv2DLSTMCell(input_shape=input_shape,
                                                 output_channels=filter, kernel_shape=[3,3],
                                                 forget_bias=1.0,
                                                 initializers=orthogonal_initializer(),
                                                 name="conv_lstm_cell_{}".format(i))
            dropout_cell = DropoutWrapper(convLSTM2D_cell, input_keep_prob=self.keep_rate,
                                          output_keep_prob=self.keep_rate,
                                          state_keep_prob=self.keep_rate)
            outputs, states = tf.nn.static_rnn(dropout_cell, x, dtype=tf.float32)
            outputs = self._batch_norm(outputs)
            x = tf.unstack(outputs, self.in_steps, 0)

        outputs = outputs[-self.out_steps:]
        y_hat = tf.transpose(outputs, perm=[1,0,2,3,4])
        return y_hat

    def _batch_norm(self, x):
        if self._data_format == 'channels_first':
            data_format = 'NCHW'
        else:
            data_format = 'NHWC'
        return tf.contrib.layers.batch_norm(
            x,
            decay=self._batch_norm_decay,
            center=True,
            scale=True,
            epsilon=self._batch_norm_epsilon,
            is_training=self._is_training,
            fused=True,
            data_format=data_format)
