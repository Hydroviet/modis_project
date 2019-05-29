import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, orthogonal_initializer

from mt_clstm import MTConv2DLSTMCell, static_rnn

class MTConvLSTM(object):

    def __init__(self, filters, is_training, data_format, batch_norm_decay, batch_norm_epsilon):
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._is_training = is_training
        assert data_format in ('channels_first', 'channels_last')
        self._data_format = data_format
        self.filters = [16, 16, 8, 1]
        self.in_steps = 14
        self.out_steps = 1
        self.input_shape = [32, 32, 1]
        self.keep_rate = 1.0

    def _dense(self, inputs):
         outputs = []
         for i, x in enumerate(inputs):
             shape_1 = tf.shape(x)
             shape = x.shape
             x = tf.reshape(x, (-1, shape[-1]))
             w = tf.get_variable("w_dense_{}".format(i), [shape[-1], 1], dtype=tf.float32)
             x = tf.matmul(x, w)
             x = tf.reshape(x, shape_1[:-1])
             outputs.append(x)
         return outputs

    def forward_pass(self, x):
        x = tf.unstack(x, self.in_steps, 1)
        filters = [self.input_shape[-1]] + self.filters
        for i in range(1, len(filters)):
            with tf.variable_scope('filter_{}'.format(i)):
                filter = filters[i]
                input_shape = self.input_shape[:-1] + [filters[i - 1]]
                mt_convLSTM2D_cell = MTConv2DLSTMCell(input_shape=input_shape,
                                                      output_channels=filter, kernel_shape=[3,3],
                                                      forget_bias=1.0,
                                                      initializers=orthogonal_initializer(),
                                                      name="mt_conv_lstm_cell_{}".format(i))
                outputs, states = static_rnn(mt_convLSTM2D_cell, x, dtype=tf.float32)
                outputs = self._dense(outputs)

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
