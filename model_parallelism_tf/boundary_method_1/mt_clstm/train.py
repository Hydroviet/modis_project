import os
import sys
import math
import argparse
import numpy as np
from datetime import datetime
sys.path.append('..')
sys.path.append('../..')

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, orthogonal_initializer

from modis_utils.misc import restore_data, cache_data
from utils import get_list_filenames
from mt_clstm import MTConv2DLSTMCell, static_rnn


def restore_data_1(path):
    data = restore_data(path)
    res = [np.expand_dims(x, axis=-1) for x in data]
    return tuple(res)

IP_ADDRESS = 'localhost'
START_PORT = 11605
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ECLM:
    def __init__(self, in_steps, out_steps, img_shape, filters=[32,1], num_blocks=1, mode="train",
                 starter_learning_rate=0.001, decay_step=500, decay_rate=1.0, verbose_step=1):
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.img_height, self.img_width = img_shape
        self.num_blocks = num_blocks
        self.input_shape = [self.img_height, self.img_width, 1]
        self.filters = filters

        self.starter_learning_rate = starter_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.global_step = None
        self.learning_rate = None
        self.keep_rate = None
        self.verbose_step = verbose_step
        self.mode = mode

        self.x = None
        self.y = None
        self.y_hat = None

        self.is_training = None
        self.loss = None


    def _create_learning_rate(self):
        with tf.variable_scope("parameter"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                self.decay_step, self.decay_rate, staircase=True, name="learning_rate")

    def _create_placeholders(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope("input"):
                self.x = tf.placeholder(tf.float32, shape=[None, self.in_steps, self.img_height, self.img_width, 1],
                                        name="input_imgs")
                self.y = tf.placeholder(tf.float32, shape=[None, self.out_steps, self.img_height, self.img_width, 1],
                                        name="target_imgs")
                self.is_training = tf.placeholder(tf.bool, name="mode")
                self.keep_rate = tf.placeholder(tf.float32, name="keep_rate")

    def batch_norm_layer(self, signal, scope, activation_fn=None):
        return tf.cond(self.is_training,
                       lambda: batch_norm(signal, is_training=True, param_initializers={
                                    "beta": tf.constant_initializer(3.), "gamma": tf.constant_initializer(2.5)},
                               center=True, scale=True, activation_fn=activation_fn, decay=1.0, scope=scope),
                       lambda: batch_norm(signal, is_training=False, param_initializers={
                                    "beta": tf.constant_initializer(3.), "gamma": tf.constant_initializer(2.5)},
                               center=True, scale=True, activation_fn=activation_fn, decay=1.0, scope=scope, reuse=True))

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

    def inference(self):
        with tf.variable_scope('filter_0'):
            print('filter:', self.filters[0])
            x = tf.unstack(self.x, self.in_steps, 1)
            mt_convLSTM2D_cell = MTConv2DLSTMCell(input_shape=self.input_shape,
                                                 output_channels=self.filters[0], kernel_shape=[3,3],
                                                 forget_bias=1.0,
                                                 initializers=orthogonal_initializer(),
                                                 name="conv_lstm_cell_{}".format(self.filters[0]))
            outputs, states = static_rnn(mt_convLSTM2D_cell, x, dtype=tf.float32)
            outputs = self._dense(outputs)
            scope = "activation_batch_norm_{}".format(self.filters[0])
            x = self.batch_norm_layer(outputs, scope=scope, activation_fn=tf.nn.tanh)
            x = tf.unstack(x, self.in_steps, 0)

        for i, filter in enumerate(self.filters[1:]):
            v_scope = 'filter_{}'.format(i + 1)
            with tf.variable_scope(v_scope):
                print('filter:', filter)
                mt_convLSTM2D_cell = MTConv2DLSTMCell(input_shape=self.input_shape,
                                                     output_channels=filter, kernel_shape=[3,3],
                                                     forget_bias=1.0,
                                                     initializers=orthogonal_initializer(),
                                                     name="conv_lstm_cell_{}".format(filter))
                outputs, states = static_rnn(mt_convLSTM2D_cell, x, dtype=tf.float32)
                x = self._dense(outputs)
                scope = "activation_batch_norm_{}".format(filter)
                outputs = self.batch_norm_layer(outputs, scope=scope, activation_fn=tf.nn.tanh)
                x = tf.unstack(x, self.in_steps, 0)

        outputs = x[-self.out_steps:]
        self.y_hat = tf.transpose(outputs, perm=[1,0,2,3,4])
        return self.y_hat


    def _create_loss(self):
        self.y_hat = self.inference()
        self.loss = tf.losses.mean_squared_error(self.y, self.y_hat, scope="loss")

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name="optimizer")\
            .minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        self._create_learning_rate()
        self._create_placeholders()
        self._create_loss()
        if self.mode == "train":
            self._create_optimizer()


def train(eclm, train_filenames, val_filenames, sess, init, train_epochs=3, batch_size=1, keep_rate=1.):
    initial_step = 1
    train_step_per_epoch = len(train_filenames)
    val_step_per_epoch = len(val_filenames)
    VERBOSE_STEP = eclm.verbose_step

    saver = tf.train.Saver()
    min_validation_loss = 100000000.

    sess.run(init)
    writer = tf.summary.FileWriter("./graphs_1", sess.graph)
    for i in range(initial_step, initial_step + train_epochs):
        print('Epoch {:04}:'.format(i))
        train_losses = []
        val_losses = []

        for j in range(train_step_per_epoch):
            x, y, _, _ = restore_data_1(train_filenames[j])
            _, loss = sess.run([eclm.optimizer, eclm.loss],
                                feed_dict={eclm.x: x, eclm.y: y,
                                           eclm.is_training: True, eclm.keep_rate: keep_rate})

            if j % VERBOSE_STEP == 0:
                print('     train_step {} - train_loss = {:0.7f}'.format(j, loss))
            train_losses.append(loss)
        train_losses = np.asarray(train_losses)
        avg_train_loss = np.mean(train_losses)
        print('Average Train Loss: {:0.7f}'.format(avg_train_loss))
        summary = tf.Summary()
        summary.value.add(tag="train_loss", simple_value=avg_train_loss)

        for j in range(val_step_per_epoch):
            x, y, _, _ = restore_data_1(val_filenames[j])
            loss = sess.run(eclm.loss,
                             feed_dict={eclm.x: x, eclm.y: y,
                                        eclm.is_training: False, eclm.keep_rate: keep_rate})
            if j % VERBOSE_STEP == 0:
                print('     val_step {} - val_loss = {:0.7f}'.format(j, loss))
            val_losses.append(loss)
        val_losses = np.asarray(val_losses)
        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < min_validation_loss:
            min_validation_loss = avg_val_loss
            saver.save(sess, "./checkpoint/best_model", i)
        print('Average Val Loss: {:0.7f}'.format(avg_val_loss))
        summary.value.add(tag="val_loss", simple_value=avg_val_loss)

        writer.add_summary(summary, global_step=i)


def inference_all(sess, eclm, test_filenames, batch_size=1, keep_rate=1.0, inference_dir=None):
    if inference_dir:
        if not os.path.exists(inference_dir):
            os.makedirs(inference_dir)

    n_tests = len(test_filenames)
    steps = n_tests
    test_losses = []
    VERBOSE_STEP = eclm.verbose_step

    for i in range(steps):
        x, y, _, _ = restore_data_1_batch(test_filenames[i])
        loss, y_hat = sess.run([eclm.loss, eclm.y_hat],
                                feed_dict={eclm.x: x, eclm.y: y,
                                           eclm.is_training: False, eclm.keep_rate: keep_rate})
        if inference_dir:
            cache_data(y_hat, os.path.join(inference_dir, '{}.dat'.format(i)))
        if i % VERBOSE_STEP == 0:
            print('     test_{} - test_loss = {:0.7f}'.format(i, loss))
        test_losses.append(loss)
    test_losses = np.asarray(test_losses)
    avg_test_loss = np.mean(test_losses)
    print('Average Test Loss: {:0.7f}'.format(avg_test_loss))
    if inference_dir:
        cache_data(test_losses, os.path.join(inference_dir, 'loss.dat'))


def inference(sess, eclm, test_inputs, pw_inputs, keep_rate=1.0, inference_path=None):
    y_hat = sess.run([eclm.y_hat],
                      feed_dict={eclm.x: test_inputs,
                                 eclm.is_training: False, eclm.keep_rate: keep_rate})
    if inference_path:
        cache_data(y_hat, inference_path)
    return y_hat


def arg_parse():
    # python train.py --mode train --data_dir sequence_data --gpus 4 --num_blocks 4 --batch_size 1 --train_epochs 20 --lr 0.001 --verbose_step 10
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--data_dir", default='../sequence_data/12', help="data dir")
    parser.add_argument("--num_blocks", default=1, type=int, help="depth of network")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--train_epochs", default=2, type=int, help="train epochs")
    parser.add_argument("--lr", default=0.00002, type=float, help="starter learning rate")
    parser.add_argument("--verbose_step", default=10, type=int, help="verbose step")
    return parser.parse_args()


def main():
    parser = arg_parse()

    in_steps = 14
    out_steps = 12

    mode = parser.mode
    sequence_data_dir = parser.data_dir
    gpus = 1
    num_blocks = parser.num_blocks
    batch_size = parser.batch_size
    train_epochs = parser.train_epochs
    learning_rate = parser.lr
    verbose_step = parser.verbose_step


    train_filenames = get_list_filenames(sequence_data_dir, 'train')[:3]
    val_filenames = get_list_filenames(sequence_data_dir, 'val')[:3]
    test_filenames = get_list_filenames(sequence_data_dir, 'test')[:3]
    assert os.path.isfile(train_filenames[0])
    assert os.path.isfile(val_filenames[0])
    assert os.path.isfile(test_filenames[0])

    sample_data = restore_data_1(train_filenames[0])[0]
    img_shape = sample_data.shape[2:4]
    print('img_shape:', img_shape)

    ###################
    filters = [8, 1]
    ###################

    if mode == 'train':
        eclm = ECLM(in_steps, out_steps, img_shape, filters,
                    num_blocks=num_blocks, starter_learning_rate=learning_rate,
                    verbose_step=verbose_step)
        eclm.build_graph()
        with tf.device('/gpu:0'):
            init = tf.global_variables_initializer()
            sess = tf.Session()
        train(eclm, train_filenames, val_filenames, sess, init, train_epochs, batch_size)
    else:
        eclm = ECLM(in_steps, out_steps, img_shape, filters,
                    mode="inference", num_blocks=num_blocks,
                    starter_learning_rate=learning_rate,
                    verbose_step=verbose_step)
        eclm.build_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoint/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            inference_dir = os.path.join('inference', str(out_steps))
            inference_all(sess, eclm, test_filenames, inference_dir=inference_dir)


if __name__ == '__main__':
    main()

