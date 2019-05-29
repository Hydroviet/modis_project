import os
import sys
import math
import numpy as np
import tensorflow as tf
from datetime import datetime

import keras
from keras.models import Model
from keras.layers import Input
from keras.utils import Sequence
from keras.layers import ConvLSTM2D, Lambda, Add, BatchNormalization, Activation
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import adam, SGD
from alt_model_checkpoint import AltModelCheckpoint

from modis_utils.misc import cache_data, restore_data
from modis_utils.image_processing import get_pixel_weights
from modis_utils.misc import restore_data, cache_data, normalize_data
from modis_utils.misc import get_data_paths, get_target_paths, get_data_from_data_file, get_target_from_target_file
from eclm_model import Eclm2D
import gen_data


def lambda_scale(x, scale_range):
    min_x = tf.reduce_min(x)
    max_x = tf.reduce_max(x)
    l = scale_range[1] - scale_range[0]
    return Lambda(lambda x: (x - min_x)/(max_x - min_x)*l + scale_range[0])(x)

class MyGenerator(Sequence):
    def __init__(self, data_filenames, batch_size):
        self.data_filenames = data_filenames
        self.batch_size = batch_size

    def __len__(self):
        n = len(self.data_filenames)
        return math.ceil(n/self.batch_size)

    def __getitem__(self, idx):
        batch_X = []

        batch_input_imgs = []
        batch_target_img = []
        batch_input_pixel_weights = []
        batch_target_pixel_weight = []
        for i in range(idx*self.batch_size, (idx + 1)*self.batch_size):
            data = restore_data(self.data_filenames[i])
            batch_input_imgs.append(data[0][np.newaxis, :, :, :, np.newaxis])
            batch_target_img.append(data[1][np.newaxis, :, :, :, np.newaxis])
            batch_input_pixel_weights.append(data[2][np.newaxis, :, :, :, np.newaxis])
            batch_target_pixel_weight.append(data[3][np.newaxis, :, :, :, np.newaxis])

        batch_X = [np.vstack(batch_input_imgs), np.vstack(batch_input_pixel_weights)]
        batch_y = np.concatenate([np.vstack(batch_target_img), np.vstack(batch_target_pixel_weight)], axis=-1)
        return batch_X, batch_y


def mse_with_pixel_weights_tf(y_true_and_pw, y_pred):
    y_true, pw = tf.split(y_true_and_pw, 2, axis=-1)
    square_error = ((y_true - y_pred)**2)*pw
    return square_error

def hadamard_product(tensors):
    return tensors[0] * tensors[1]

def hadamard_product_output_shape(input_shapes):
    shape1 = list(input_shapes[0])
    shape2 = list(input_shapes[1])
    assert shape1 == shape2  # else hadamard product isn't possible
    return tuple(shape1)


def block(x, pixel_weights, kernel_size, n_out, strides, padding):
    x = Lambda(hadamard_product, hadamard_product_output_shape)([x, pixel_weights])
    x = BatchNormalization()(x)
    x = lambda_scale(x, (-2, 2))

    x = print_layer(x, "input: ")
    x = Eclm2D(filters=32, kernel_size=kernel_size,
               strides=strides, padding=padding, return_sequences=True)(x)
    x = print_layer(x, "layer_1: ")

    x = Eclm2D(filters=1, kernel_size=kernel_size, n_out=n_out,
               strides=strides, padding=padding, return_sequences=True)(x)
    x = print_layer(x, "layer_2: ")

    return x


def create_model(input_shape, n_out=12):
    inputs = [Input(shape=input_shape), Input(shape=input_shape)]
    x = inputs[0]
    pixel_weights = inputs[1]

    x = block(x, pixel_weights, 3, n_out, 1, 'same')

    # Model
    model = Model(inputs=inputs, outputs=x)

    model.compile(loss=mse_with_pixel_weights_tf, optimizer='adam')
    return model

def double_tanh(x):
    x = Activation('tanh')(x)
    return Lambda(lambda x: 2*x)(x)

def print_layer(layer, message, first_n=3, summarize=1024):
    return Lambda((
        lambda x: tf.Print(x, [tf.reduce_min(x), tf.reduce_max(x)],
            message=message,
            first_n=first_n,
            summarize=summarize)))(layer)


def create_model_1(input_shape, n_out=12):
    inputs = [Input(shape=input_shape), Input(shape=input_shape)]
    x = inputs[0]
    x = print_layer(x, "input: ")

    x = Eclm2D(filters=32, kernel_size=3, strides=1,
               padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    #x = double_tanh(x)
    x = lambda_scale(x, (-2,2))
    x = print_layer(x, "layer_1: ")

    x = Eclm2D(filters=1, kernel_size=3, strides=1, n_out=n_out,
               padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    #x = double_tanh(x)
    x = lambda_scale(x, (-2,2))
    x = print_layer(x, "output: ")

    # Model
    model = Model(inputs=inputs, outputs=x)

    model.compile(loss=mse_with_pixel_weights_tf, optimizer='adam')
    return model


def set_checkpoint(base_model, weights_dir):
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    filepath = os.path.join(weights_dir, "weights-{epoch:03d}.h5")
    checkpoint = AltModelCheckpoint(
    filepath, base_model, monitor='val_loss', mode='min', verbose=1, period=5)
    csv_logger = CSVLogger(os.path.join(
    weights_dir, 'log.csv'), append=True, separator=';')
    callbacks_list = [checkpoint, csv_logger]
    if lr_reducer is not None:
        callbacks_list = [lr_reducer] + callbacks_list

    timestamps = datetime.now()
    timestamps = str(timestamps)
    timestamps = timestamps[:timestamps.find('.')]
    timestamps = timestamps.replace(' ', '_')
    tensorboard_logdir = 'logs/{}'.format(timestamps)
    tensorboard = TensorBoard(log_dir=tensorboard_logdir)
    callbacks_list.append(tensorboard)
    return callbacks_list


if __name__ == '__main__':

    #Params
    water_threshold = gen_data.water_threshold
    timesteps = gen_data.real_timesteps
    n_out = gen_data.n_out

    batch_size = 1
    epochs = 2
    optimizer = adam(lr=0.001)

    if len(sys.argv) == 3:
        sequence_data_dir = sys.argv[1]
        gpus = sys.argv[2]
    else:
        sequence_data_dir = 'sequence_data'
        gpus = 1
    sequence_data_dir = os.path.join(sequence_data_dir, str(n_out))

    weights_dir = 'weights'
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=5, min_lr=0.001)
    #End params

    train_data_filenames = [os.path.join(sequence_data_dir, 'train', filename)
            for filename in os.listdir(os.path.join(sequence_data_dir, 'train'))]
    val_data_filenames = [os.path.join(sequence_data_dir, 'val', filename)
            for filename in os.listdir(os.path.join(sequence_data_dir, 'val'))]
    assert os.path.isfile(train_data_filenames[0])
    assert os.path.isfile(val_data_filenames[0])

    train_batch_generator = MyGenerator(train_data_filenames, batch_size)
    val_batch_generator = MyGenerator(val_data_filenames, batch_size)

    input_shape = (timesteps, 513, 513, 1)
    #base_model = create_model(input_shape, n_out=n_out)
    base_model = create_model_1(input_shape, n_out=n_out)

    if gpus > 1:
        model = multi_gpu_model(base_model, gpus=gpus)
    else:
        model = base_model
    model.compile(loss=mse_with_pixel_weights_tf, optimizer=optimizer)

    model.summary()
    callbacks = set_checkpoint(base_model, weights_dir)
    history = model.fit_generator(generator=train_batch_generator,
                                  validation_data=val_batch_generator,
                                  validation_steps=len(val_batch_generator),
                                  epochs=epochs,
                                  callbacks=callbacks)

