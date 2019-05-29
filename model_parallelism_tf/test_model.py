import os
import math
import numpy as np
from modis_utils.misc import cache_data, restore_data

import keras
from keras.models import Model
from keras.layers import Input

import tensorflow as tf
from keras.utils import Sequence
import keras.backend as K

from eclm_model import Eclm2D
from keras.layers import ConvLSTM2D, Lambda, Add, BatchNormalization

from tensorflow.python import debug as tf_debug
#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)

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

def lambda_scale(x, scale_range):
    min_x = tf.reduce_min(x)
    max_x = tf.reduce_max(x)
    l = scale_range[1] - scale_range[0]
    return Lambda(lambda x: (x - min_x)/(max_x - min_x)*l + scale_range[0])(x)

def hadamard_product(tensors):
    return tensors[0] * tensors[1]

def hadamard_product_output_shape(input_shapes):
    shape1 = list(input_shapes[0])
    shape2 = list(input_shapes[1])
    assert shape1 == shape2  # else hadamard product isn't possible
    return tuple(shape1)


def print_layer(layer, message, first_n=3, summarize=1024):
    return Lambda((
    	lambda x: tf.Print(x, [tf.reduce_min(x), tf.reduce_max(x)],
            message=message,
            first_n=first_n,
            summarize=summarize)))(layer)


def block(x, pixel_weights, filters, kernel_size, n_out, strides, padding):
    x = Lambda(hadamard_product, hadamard_product_output_shape)([x, pixel_weights])
    x = BatchNormalization()(x)
    x = lambda_scale(x, (-2,2))

    x = print_layer(x, "input_1: ")
    x = Eclm2D(filters=32, kernel_size=kernel_size,
            strides=strides, padding=padding, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = print_layer(x, "layer_1: ")
    x = lambda_scale(x, (-2,2))

    x = print_layer(x, "input_2: ")
    x = Eclm2D(filters=16, kernel_size=kernel_size,
            strides=strides, padding=padding, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = print_layer(x, "layer_2: ")
    x = lambda_scale(x, (-2,2))

    x = print_layer(x, "input_3: ")
    x = Eclm2D(filters=1, kernel_size=kernel_size, n_out=n_out,
            strides=strides, padding=padding, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = print_layer(x, "layer_3: ")
    x = lambda_scale(x, (-2,2))

    return x

def create_model(input_shape, filters=32, n_out=12):
    inputs = [Input(shape=input_shape), Input(shape=input_shape)]
    x = inputs[0]
    pixel_weights = inputs[1]

    x = block(x, pixel_weights, filters, 3, n_out, 1, 'same')
    x = block(x, pixel_weights, filters, 3, n_out, 1, 'same')

    # Model
    model = Model(inputs=inputs, outputs=x)

    model.compile(loss=mse_with_pixel_weights_tf, optimizer='adam')
    return model

def create_model_1(input_shape, filters=32, n_out=2):
    inputs = [Input(shape=input_shape), Input(shape=input_shape)]
    x = inputs[0]
    pixel_weights = inputs[1]

    x = block(x, pixel_weights, filters, 3, n_out, 1, 'same')

    model = Model(inputs=inputs, outputs=x)

    model.compile(loss=mse_with_pixel_weights_tf, optimizer='adam')
    return model



def main():

    ##################Params#############################
    timesteps = 3
    n_out = 2
    batch_size = 1
    epochs = 2
    sequence_data_dir = 'sequence_data'


    ##################End Params#########################


    train_data_filenames = [os.path.join(sequence_data_dir, 'train', filename)
            for filename in os.listdir(os.path.join(sequence_data_dir, 'train'))]
    assert os.path.isfile(train_data_filenames[0])

    train_batch_generator = MyGenerator(train_data_filenames, batch_size)
    for x in train_batch_generator[0]:
        try:
            print('train_batch_generator[0]: [', x[0].shape, x[1].shape, ']', end=', ')
        except:
            print(x.shape)

    input_shape = (timesteps, 513, 513, 1)
    #model = create_model(input_shape, n_out=n_out)
    model = create_model_1(input_shape, n_out=n_out)

    model.summary()
    model.fit_generator(generator=train_batch_generator,
                        epochs=epochs)


if __name__ == '__main__':
    main()
