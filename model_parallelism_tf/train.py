import os
import sys
import math
import argparse
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.layers.python.layers.layers import batch_norm
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, orthogonal_initializer

import gen_data
from modis_utils.misc import restore_data, cache_data
from utils import get_block_idx_per_gpu, get_list_filenames
from utils import restore_data_batch, tf_scale, get_available_gpus


IP_ADDRESS = 'localhost'
START_PORT = 11605
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ECLM:
    def __init__(self, in_steps, out_steps, img_shape, devices, filters=[32,1], num_blocks=1, mode="train",
                 starter_learning_rate=0.001, decay_step=500, decay_rate=1.0, verbose_step=1):
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.img_height, self.img_width = img_shape
        self.num_blocks = num_blocks
        self.input_shape = [self.img_height, self.img_width, 1]
        self.filters = filters

        self.starter_learning_rate = starter_learning_rate
        self.devices = devices
        self.gpus = len(devices)
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.global_step = None
        self.learning_rate = None
        self.keep_rate = None
        self.verbose_step = verbose_step
        self.mode = mode

        self.x = None
        self.pw = None
        self.y = None
        self.y_pw = None
        self.y_hat = None

        self.is_training = None
        self.loss = None

        self.gpus_names = get_available_gpus()[:self.gpus]
        self.block_idx_per_gpu = get_block_idx_per_gpu(self.num_blocks, self.gpus)

    def _create_learning_rate(self):
        with tf.variable_scope("parameter"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                self.decay_step, self.decay_rate, staircase=True, name="learning_rate")

    def _create_placeholders(self):
        with tf.device(self.devices[0]):
            with tf.variable_scope("input"):
                self.x = tf.placeholder(tf.float32, shape=[None, self.in_steps, self.img_height, self.img_width, 1],
                                        name="input_imgs")
                self.pw = tf.placeholder(tf.float32, shape=[None, self.in_steps, self.img_height, self.img_width, 1],
                                        name="input_pws")
                self.y = tf.placeholder(tf.float32, shape=[None, self.out_steps, self.img_height, self.img_width, 1],
                                        name="target_imgs")
                self.y_pw = tf.placeholder(tf.float32, shape=[None, self.out_steps, self.img_height, self.img_width, 1],
                                           name="target_pws")
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

    def inference(self):
        with tf.device(self.devices[0]):
            x = self.x * self.pw
            x = self.batch_norm_layer(x, scope="pw_input")
            x = 2*tf.tanh(x)
            x = tf.unstack(self.x, self.in_steps, 1)
            convLSTM2D_cell = rnn.Conv2DLSTMCell(input_shape=self.input_shape,
                                                 output_channels=self.filters[0], kernel_shape=[3,3],
                                                 forget_bias=1.0,
                                                 initializers=orthogonal_initializer(),
                                                 name="conv_lstm_cell_{}".format(self.filters[0]))
            dropout_cell = DropoutWrapper(convLSTM2D_cell, input_keep_prob=self.keep_rate,
                                          output_keep_prob=self.keep_rate,
                                          state_keep_prob=self.keep_rate)
            outputs, states = tf.nn.static_rnn(dropout_cell, x, dtype=tf.float32)
            scope = "activation_batch_norm_{}".format(self.filters[0])
            outputs = self.batch_norm_layer(outputs, scope=scope, activation_fn=tf.nn.tanh)
            outputs = 2*outputs
            x = tf.unstack(outputs, self.in_steps, 0)

        with tf.device(self.devices[-1]):
            for filter in self.filters[1:]:
                convLSTM2D_cell = rnn.Conv2DLSTMCell(input_shape=self.input_shape,
                                                     output_channels=filter, kernel_shape=[3,3],
                                                     forget_bias=1.0,
                                                     initializers=orthogonal_initializer(),
                                                     name="conv_lstm_cell_{}".format(filter))
                dropout_cell = DropoutWrapper(convLSTM2D_cell, input_keep_prob=self.keep_rate,
                                              output_keep_prob=self.keep_rate,
                                              state_keep_prob=self.keep_rate)
                outputs, states = tf.nn.static_rnn(dropout_cell, x, dtype=tf.float32)
                scope = "activation_batch_norm_{}".format(filter)
                outputs = self.batch_norm_layer(outputs, scope=scope, activation_fn=tf.nn.tanh)
                outputs = 2*outputs
                x = tf.unstack(outputs, self.in_steps, 0)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        outputs = outputs[-self.out_steps:]
        self.y_hat = tf.transpose(outputs, perm=[1,0,2,3,4])
        return self.y_hat


    def _create_loss(self):
        with tf.device(self.devices[-1]):
            self.y_hat = self.inference()
            self.loss = tf.reduce_mean(tf.multiply(tf.pow((self.y - self.y_hat), 2.0), self.y_pw, name="loss"))

    def _create_optimizer(self):
        with tf.device(self.devices[-1]):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name="optimizer")\
                .minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        self._create_learning_rate()
        self._create_placeholders()
        self._create_loss()
        if self.mode == "train":
            self._create_optimizer()


def train(eclms, train_filenames, val_filenames, server, init, train_epochs=3, batch_size=1, keep_rate=1.):
    initial_step = 1
    train_step_per_epoch = math.ceil(len(train_filenames)/batch_size)
    val_step_per_epoch = math.ceil(len(val_filenames)/batch_size)
    VERBOSE_STEP = eclm.verbose_step

    saver = tf.train.Saver()
    min_validation_loss = 100000000.
    with tf.Session(server.target) as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./graphs_1", sess.graph)
        for i in range(initial_step, initial_step + train_epochs):
            print('Epoch {:04}:'.format(i))
            train_losses = []
            val_losses = []

            for j in range(train_step_per_epoch):
                x, y, pw, y_pw = restore_data_batch(train_filenames[j*batch_size : (j + 1)*batch_size])
                for eclm in eclms:
                    _, loss = sess.run([eclm.optimizer, eclm.loss],
                                        feed_dict={eclm.x: x, eclm.y: y, eclm.pw: pw, eclm.y_pw: y_pw,
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
                x, y, pw, y_pw = restore_data_batch(val_filenames[j*batch_size : (j + 1)*batch_size])
                loss = sess.run(eclm.loss,
                                 feed_dict={eclm.x: x, eclm.y: y, eclm.pw: pw, eclm.y_pw: y_pw,
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
    steps = math.ceil(steps/batch_size)
    test_losses = []
    VERBOSE_STEP = eclm.verbose_step

    for i in range(steps):
        x, y, pw, y_pw = restore_data_batch(test_filenames[i*batch_size: (i + 1)*batch_size])
        loss, y_hat = sess.run([eclm.loss, eclm.y_hat],
                                feed_dict={eclm.x: x, eclm.y: y, eclm.pw: pw, eclm.y_pw: y_pw,
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
                      feed_dict={eclm.x: test_inputs, eclm.pw: pw_inputs,
                                 eclm.is_training: False, eclm.keep_rate: keep_rate})
    if inference_path:
        cache_data(y_hat, inference_path)
    return y_hat


def arg_parse():
    # python train.py --mode train --data_dir sequence_data --gpus 4 --num_blocks 4 --batch_size 1 --train_epochs 20 --lr 0.001 --verbose_step 10
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--data_dir", default=gen_data.sequence_data_dir, help="data dir")
    parser.add_argument("--gpus", default=len(get_available_gpus()), type=int, help="number of gpus")
    parser.add_argument("--num_blocks", default=1, type=int, help="depth of network")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--train_epochs", default=50, type=int, help="train epochs")
    parser.add_argument("--lr", default=0.00002, type=float, help="starter learning rate")
    parser.add_argument("--verbose_step", default=10, type=int, help="verbose step")
    return parser.parse_args()


CONST_DEVICE = '/job:worker/task:'
def get_devices(n_devices):
    return [CONST_DEVICE + str(i) for i in range(n_devices)]

def main():
    parser = arg_parse()

    in_steps = gen_data.real_timesteps
    out_steps = gen_data.n_out

    mode = parser.mode
    sequence_data_dir = parser.data_dir
    #gpus = max(min(parser.gpus, len(get_available_gpus())), 1)
    gpus = 4
    num_blocks = parser.num_blocks
    batch_size = parser.batch_size
    train_epochs = parser.train_epochs
    learning_rate = parser.lr
    verbose_step = parser.verbose_step

    n_ports = gpus
    devices = get_devices(n_ports)

    train_filenames = get_list_filenames(sequence_data_dir, 'train')
    val_filenames = get_list_filenames(sequence_data_dir, 'val')
    test_filenames = get_list_filenames(sequence_data_dir, 'test')
    assert os.path.isfile(train_filenames[0])
    assert os.path.isfile(val_filenames[0])
    assert os.path.isfile(test_filenames[0])

    sample_data = restore_data(train_filenames[0])[0]
    img_shape = sample_data.shape[1:]

    ###################
    filters = [32, 16, 8, 1]
    ###################

    PORTS = [str(START_PORT + i) for i in range(n_ports)]
    task_idx = 0
    workers = [IP_ADDRESS + ":" + PORT for PORT in PORTS]
    cluster_spec = tf.train.ClusterSpec({'worker': workers})

    server = tf.train.Server(cluster_spec, job_name='worker', task_index=task_idx)
    print(server.server_def)


    if mode == 'train':
        eclm_1 = ECLM(in_steps, out_steps, img_shape, devices[:2], filters,
                      num_blocks=num_blocks, starter_learning_rate=learning_rate,
                      verbose_step=verbose_step)
        eclm_2 = ECLM(in_steps, out_steps, img_shape, devices[2:], filters,
                      num_blocks=num_blocks, starter_learning_rate=learning_rate,
                      verbose_step=verbose_step)
        eclm_1.build_graph()
        eclm_2.build_graph()
        with tf.device(devices[0]):
            init = tf.global_variables_initializer()
        train(eclm_1, eclm_2, train_filenames, val_filenames, server, init, train_epochs, batch_size)
    else:
        eclm = ECLM(in_steps, out_steps, img_shape, devices[:2], filters,
                    mode="inference", num_blocks=num_blocks,
                    starter_learning_rate=learning_rate,
                    verbose_step=verbose_step)
        eclm.build_graph()
        saver = tf.train.Saver()
        with tf.Session(server.target) as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoint/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            inference_dir = os.path.join('inference', str(out_steps))
            inference_all(sess, eclm, test_filenames, inference_dir=inference_dir)


if __name__ == '__main__':
    main()

