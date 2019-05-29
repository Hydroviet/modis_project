import os
import numpy as np
import tensorflow as tf

'''
import keras
from keras.layers import BatchNormalization, TimeDistributed
'''

import tensorflow.keras as keras
from tensorflow.python.keras.layers import BatchNormalization, TimeDistributed


from modis_utils.generators import OneOutputGenerator
from modis_utils.misc import get_data_test, get_target_test, cache_data
from modis_utils.model.core import compile_model, conv_lstm_2D, conv_2D
from modis_utils.model.loss_function import mse_with_mask, mse_with_mask_batch
from modis_utils.model.eval import predict_and_visualize_by_data_file_one_output
from modis_utils.image_processing import mask_lake_img


class ConvLSTMSimpleOneTimeStepsOutput:
    
    def get_generator(data_filenames, batch_size, original_batch_size, pretrained):
        return OneOutputGenerator(data_filenames, batch_size, original_batch_size, pretrained)

    def create_model(modis_utils_obj):
        crop_size = modis_utils_obj._crop_size
        input_timesteps = modis_utils_obj._input_timesteps
        output_timesteps = modis_utils_obj._output_timesteps
        compile_params = modis_utils_obj._compile_params
        return ConvLSTMSimpleOneTimeStepsOutput._create_model(
            crop_size, crop_size, input_timesteps, compile_params)
    
    def _create_model(img_height, img_width, input_timesteps, compile_params):
        # Prepair
        input_shape = (input_timesteps, img_height, img_width, 1)
        
        # Model architecture
        source = keras.Input(
            name='seed', shape=input_shape, dtype=tf.float32)
        model = conv_lstm_2D(filters=128, kernel_size=3, strides=1, padding='same')(source)
        model = BatchNormalization()(model)
        model = conv_lstm_2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization()(model)
        model = conv_lstm_2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization()(model)
        model = conv_lstm_2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization()(model)
        model = conv_lstm_2D(filters=128, kernel_size=3, strides=1, padding='same', return_sequences=False)(model)
        model = BatchNormalization()(model)
        predict_img = conv_2D(filters=1, kernel_size=3, strides=1, padding='same')(model)
        model = keras.Model(inputs=[source], outputs=[predict_img])
        
        # Compile model
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


class ConvLSTMSimpleSequenceTimeStepsOutput:
    
    def get_generator(data_filenames, batch_size, original_batch_size, pretrained):
        return OneOutputGenerator(data_filenames, batch_size, original_batch_size, pretrained)

    def create_model(modis_utils_obj):
        crop_size = modis_utils_obj._crop_size
        input_timesteps = modis_utils_obj._input_timesteps
        output_timesteps = modis_utils_obj._output_timesteps
        compile_params = modis_utils_obj._compile_params
        return ConvLSTMSimpleSequenceTimeStepsOutput._create_model(
            crop_size, crop_size, input_timesteps, compile_params)
    
    def _create_model(img_height, img_width, input_timesteps, compile_params):
        # Prepair
        input_shape = (input_timesteps, img_height, img_width, 1)
        
        # Model architecture
        source = keras.Input(
            name='seed', shape=input_shape, dtype=tf.float32)
        model = conv_lstm_2D(filters=128, kernel_size=3, strides=1, padding='same')(source)
        model = BatchNormalization()(model)
        model = conv_lstm_2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization()(model)
        model = conv_lstm_2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization()(model)
        model = conv_lstm_2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization()(model)
        model = conv_lstm_2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization()(model)
        predict_imgs = TimeDistributed(conv_2D(filters=1, kernel_size=3, strides=1, padding='same'))(model)
        model = keras.Model(inputs=[source], outputs=[predict_imgs])
        
        # Compile model
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
            metric = mse_with_mask_batch
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
        predict_and_visualize_by_data_file_sequence_output(
            data_file_path, target_file_path, pred, idx, modis_utils_obj._result_dir)

    def create_predict_mask_lake(modis_utils_obj, data_type='test'):
        for idx in range(modis_utils_obj.get_n_tests(data_type)):
            pred = modis_utils_obj.get_inference(data_type, idx)
            pred = pred[0]
            pred = modis_utils_obj._preprocess_strategy_context.inverse(pred)
            predict_mask_lake_path = os.path.join(
                modis_utils_obj._predict_mask_lake_dir, data_type, '{}.dat'.format(idx))
            cache_data(mask_lake_img(pred), predict_mask_lake_path)

    