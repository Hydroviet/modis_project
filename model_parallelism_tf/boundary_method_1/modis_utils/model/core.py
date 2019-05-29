import os
import numpy as np

import tensorflow as tf

'''
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import LSTM, ConvLSTM2D, Activation
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import BatchNormalization, TimeDistributed
'''

import tensorflow.keras as keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers import LSTM, ConvLSTM2D, Activation
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.python.keras.layers import BatchNormalization, TimeDistributed


from modis_utils.model.loss_function import mse_with_mask
from modis_utils.misc import get_data_test, get_target_test
from modis_utils.generators import OneOutputGenerator, MultipleOutputGenerator
from modis_utils.model.eval import predict_and_visualize_by_data_file_one_output



def conv_lstm_2D(
        filters, kernel_size, strides, padding='same', 
        activation='tanh', return_sequences=True, dilation_rate=(1,1)):
    return ConvLSTM2D(
        filters=filters, kernel_size=kernel_size, strides=strides, activation=activation,
        padding=padding, return_sequences=return_sequences,
        dilation_rate=dilation_rate)


def conv_2D(
        filters, kernel_size, strides, padding='valid',
        activation='tanh', dilation_rate=(1,1)):
    return Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        activation=activation, dilation_rate=dilation_rate)


def conv_2D_transpose(
        filters, kernel_size, strides, padding='valid', 
        activation='tanh', dilation_rate=(1,1)):
    return Conv2DTranspose(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        activation=activation, dilation_rate=dilation_rate)


def create_encoder():
    net = Sequential()
    #net.add(conv_2D(filters=16, kernel_size=3, strides=2))
    #net.add(conv_2D(filters=32, kernel_size=3, strides=2))
    net.add(conv_2D(filters=48, kernel_size=3, strides=2))
    net.add(conv_2D(filters=64, kernel_size=3, strides=2))
    net.add(conv_2D(filters=80, kernel_size=3, strides=2))
    return net


def create_decoder():
    net = Sequential()
    net.add(conv_2D_transpose(filters=80, kernel_size=3, strides=2))
    net.add(conv_2D_transpose(filters=64, kernel_size=3, strides=2))
    net.add(conv_2D_transpose(filters=48, kernel_size=3, strides=2))
    #net.add(conv_2D_transpose(filters=32, kernel_size=3, strides=2))
    #net.add(conv_2D_transpose(filters=16, kernel_size=3, strides=2))
    return net


def compile_model(model, compile_params):
    optimizer = keras.optimizers.SGD(lr=1e-4)
    loss = 'mse'
    metrics = ['mse']
    loss_weights = None

    if 'optimizer' in compile_params.keys():
        optimizer = compile_params['optimizer']
    if 'loss' in compile_params.keys():
        loss = compile_params['loss']
    if 'metrics' in compile_params.keys():
        metrics= compile_params['metrics']
    if 'loss_weights' in compile_params.keys():
        loss_weights = compile_params["loss_weights"]

    model.compile(
        optimizer=optimizer, loss=loss,
        loss_weights=loss_weights, metrics=metrics)
    return model


def create_model(model_params, compile_params):
    # Prepair
    input_shape = model_params['input_shape']
    batch_norm = model_params['batch_norm']
    encoder = create_encoder()
    decoder = create_decoder()
    output_shape = (input_shape[1], input_shape[2], input_shape[3])

    dim_flaten = 1
    for x in output_shape:
        dim_flaten *= x
    
    # Model architecture
    source = keras.Input(
        name='seed', shape=input_shape, dtype=tf.float32)
    encoder_conv2d = TimeDistributed(encoder)(source)
    encoder_convlstm = conv_lstm_2D(filters=32, kernel_size=3, strides=1, padding='same')(encoder_conv2d)
    if batch_norm:
        encoder_convlstm = BatchNormalization()(encoder_convlstm)
    encoder_convlstm = conv_lstm_2D(filters=64, kernel_size=3, strides=1, padding='same')(encoder_convlstm)
    if batch_norm:
        encoder_convlstm = BatchNormalization()(encoder_convlstm)
    encoder_convlstm = conv_lstm_2D(filters=80, kernel_size=3, strides=1, padding='same')(encoder_convlstm)
    if batch_norm:
        encoder_convlstm = BatchNormalization()(encoder_convlstm)

    decoder_convlstm = conv_lstm_2D(filters=80, kernel_size=3, strides=1, padding='same')(encoder_convlstm)
    if batch_norm:
        decoder_convlstm = BatchNormalization()(decoder_convlstm)
    decoder_convlstm = conv_lstm_2D(filters=64, kernel_size=3, strides=1, padding='same')(decoder_convlstm)
    if batch_norm:
        decoder_convlstm = BatchNormalization()(decoder_convlstm)
    decoder_convlstm = conv_lstm_2D(filters=32, kernel_size=3, strides=1, padding='same')(decoder_convlstm)
    if batch_norm:
        decoder_convlstm = BatchNormalization()(decoder_convlstm)
    decoder_conv2d = TimeDistributed(decoder)(decoder_convlstm)
    reconstruct_conv2d = conv_2D(filters=1, kernel_size=3, strides=1, padding='same')
    reconstruct_imgs = TimeDistributed(reconstruct_conv2d)(decoder_conv2d)

    predict_img = conv_lstm_2D(filters=1, kernel_size=3, strides=1, padding='same',
                               return_sequences=False)(encoder_convlstm)
    predict_img = Flatten()(predict_img)
    predict_img = Dense(units=4096)(predict_img)
    predict_img = Activation('tanh')(predict_img)
    predict_img = Dense(units=dim_flaten)(predict_img)
    predict_img = Reshape(target_shape=output_shape)(predict_img)
    predict_img = Activation('tanh')(predict_img)

    model = keras.Model(inputs=[source], outputs=[reconstruct_imgs, predict_img])
    
    # Compile parameters
    model = compile_model(model, compile_params)
    return model
