import os
import numpy as np
import tensorflow as tf

'''
import keras
from keras.models import Model
from keras.layers import BatchNormalization, TimeDistributed
from keras.layers import Conv2D, ConvLSTM2D, Conv2DTranspose
from keras.layers import Input, Add, Lambda, Dense, Activation
'''

import tensorflow.keras as keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import BatchNormalization, TimeDistributed
from tensorflow.python.keras.layers import Conv2D, ConvLSTM2D, Conv2DTranspose
from tensorflow.python.keras.layers import Input, Add, Lambda, Dense, Activation

from modis_utils.generators import OneOutputGenerator
from modis_utils.misc import get_data_test, get_target_test, cache_data
from modis_utils.model.core import compile_model, conv_lstm_2D, conv_2D
from modis_utils.model.loss_function import mse_with_mask, mse_with_mask_batch
from modis_utils.model.eval import predict_and_visualize_by_data_file_one_output
from modis_utils.image_processing import mask_lake_img


class SkipConvLSTMSingleOutput:
    
    def get_generator(data_filenames, batch_size, original_batch_size, pretrained):
        return OneOutputGenerator(data_filenames, batch_size, original_batch_size, pretrained)

    def create_model(modis_utils_obj):
        crop_size = modis_utils_obj._crop_size
        input_timesteps = modis_utils_obj._input_timesteps
        output_timesteps = modis_utils_obj._output_timesteps
        compile_params = modis_utils_obj._compile_params
        return SkipConvLSTMSingleOutput._create_model(
            crop_size, crop_size, input_timesteps, compile_params)
    
    def _create_model(img_height, img_width, input_timesteps, compile_params):
        input_shape = (input_timesteps, img_height, img_width, 1)
        x = Input(shape=input_shape, name='input')

        encoder_input_shape = input_shape[1:]
        encode_block = SkipConvLSTMSingleOutput._create_encoder(encoder_input_shape)
        net = TimeDistributed(encode_block)(x)

        net = ConvLSTM2D(filters=128, kernel_size=3, padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        hidden = ConvLSTM2D(filters=80, kernel_size=3, padding='same', return_sequences=False)(net)
        hidden = BatchNormalization()(hidden)

        decode_block = SkipConvLSTMSingleOutput._create_decoder(hidden.shape[1:], encode_block)
        net = decode_block([hidden, Lambda(lambda x: x[:,-1,:,:,:])(x)])
        net = Activation('sigmoid')(net)

        model = Model(inputs=x, outputs=net, name='skip_conv_single_output')
        model = compile_model(model, compile_params)
        return model 
        

    def inference(modis_utils_obj, data_type, idx):
        file_path = modis_utils_obj._data_files[data_type]['data']
        data_test = get_data_test(file_path, idx)
        data_test = np.expand_dims(np.expand_dims(data_test, axis=0), axis=-1)
        output = modis_utils_obj.inference_model.predict(data_test)
        output = np.squeeze(np.squeeze(output, axis=0), axis=-1)
        return output

    def eval(modis_utils_obj, data_type, idx, metric):
        if metric is None:
            metric = mse_with_mask
        target_path = modis_utils_obj._data_files[data_type]['target']
        target = get_target_test(target_path, idx)
        mask_path = modis_utils_obj._data_files[data_type]['mask']
        mask = get_target_test(mask_path, idx)
        predict = modis_utils_obj.get_inference(data_type, idx)
        return metric(target, predict, mask)

    def visualize_result(modis_utils_obj, data_type, idx):
        data_file_path = modis_utils_obj._data_files[data_type]['data']
        target_file_path = modis_utils_obj._data_files[data_type]['target']
        pred = modis_utils_obj.get_inference(data_type, idx)
        predict_and_visualize_by_data_file_one_output(
            data_file_path, target_file_path, pred, idx, modis_utils_obj._result_dir)

    def create_predict_mask_lake(modis_utils_obj, data_type='test'):
        for idx in range(modis_utils_obj.get_n_tests(data_type)):
            pred = modis_utils_obj.get_inference(data_type, idx)
            pred = modis_utils_obj._preprocess_strategy_context.inverse(pred)
            predict_mask_lake_path = os.path.join(
                modis_utils_obj._predict_mask_lake_dir, data_type, '{}.dat'.format(idx))
            cache_data(mask_lake_img(pred), predict_mask_lake_path)


    def _create_encoder(input_shape):
        x = Input(shape=input_shape, name='encoder_input')
        net = Conv2D(filters=48, kernel_size=3, strides=2, name='conv1')(x)
        net = BatchNormalization()(net)
        net = Conv2D(filters=64, kernel_size=3, strides=2, name='conv2')(net)
        net = BatchNormalization()(net)
        net = Conv2D(filters=80, kernel_size=3, strides=2, name='conv3')(net)
        net = BatchNormalization()(net)
        net = Model(inputs=x, outputs=net, name='encoder')
        return net


    def _create_decoder(input_shape, encode_block):
        conv3 = encode_block.get_layer('conv3').output
        x = Input(shape=input_shape, name='decoder_input')
        net = Add()([x, conv3])
        
        conv2 = encode_block.get_layer('conv2').output
        net = Conv2DTranspose(filters=64, kernel_size=3, strides=2, name='dconv1')(x)
        net = Add()([net, conv2])
        net = BatchNormalization()(net)
        
        net = Conv2DTranspose(filters=32, kernel_size=3, strides=2, name='dconv2')(net)
        net = BatchNormalization()(net)

        encoder_input = encode_block.get_layer('encoder_input').output
        net = Conv2DTranspose(filters=1, kernel_size=3, strides=2, name='dconv3')(net)
        net = Add()([net, encoder_input])
        net = BatchNormalization()(net)

        net = Model(inputs=[x, encode_block.input], outputs=net, name='decoder')
        return net

    
