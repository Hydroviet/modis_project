import os
import numpy as np
from scipy import misc
import shutil
from datetime import datetime
from shutil import make_archive
from matplotlib import pyplot as plt
import tensorflow as tf

'''
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import LearningRateScheduler, CSVLogger
'''

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.callbacks import LearningRateScheduler, CSVLogger

from modis_utils.preprocessing.preprocess_strategy_context import PreprocessStrategyContext
from modis_utils.image_processing import create_water_cloud_mask, create_groundtruth_mask_lake, create_one_only_mask
from modis_utils.misc import create_data_file_continuous_years, get_data_file_path
from modis_utils.preprocessing.random_crop import augment_one_reservoir_without_cache
from modis_utils.preprocessing.random_crop import merge_data_augment
from modis_utils.misc import get_data_test, get_target_test, cache_data, restore_data, get_data_paths, get_target_paths
from modis_utils.model.core import compile_model
from modis_utils.model.model_utils_factory import get_model_utils


class ModisUtils:

    def __init__(self,
                 raw_data_dir='../raw_data',
                 modis_product='MOD13Q1',
                 reservoir_index=0,
                 preprocessed_type='normalized_div',
                 used_band='NDVI',
                 crop_size=32,
                 n_samples=50,
                 input_timesteps=12,
                 output_timesteps=1,
                 year_range=(2000, 2018),
                 model_name='convlstm_simple',
                 batch_size=32,
                 model_keras=True,
                 compile_params=None,
                 original_batch_size=1024,
                 TPU_FLAG=False,
                 training=True,
                 monitor=None,
                 monitor_mode='min',
                 resize_input=None,
                 pretrained=False,
                 lr_reducer=None):
        # Define parameters
        self._modis_product = modis_product
        self._reservoir_index = reservoir_index
        self._raw_data_dir = os.path.join(
            raw_data_dir, self._modis_product,
            str(self._reservoir_index))
        self._preprocessed_type = preprocessed_type
        self._used_band = used_band
        self._crop_size = crop_size
        self._n_samples = n_samples
        self._input_timesteps = input_timesteps
        self._output_timesteps = output_timesteps
        self._year_range = year_range
        self._original_batch_size = original_batch_size
        self._TPU_FLAG = TPU_FLAG
        self._resize_input = resize_input
        self._pretrained = pretrained
        self._training = training
        if self._pretrained:
            assert self._crop_size == 224
        self._lr_reducer = lr_reducer

        if self._crop_size == -1:
            self._n_samples = 1
        
        # Dir and Dir prefix
        self._dir_prefix = os.path.join(
            self._modis_product, str(self._reservoir_index), 
            self._used_band, self._preprocessed_type)
        
        self._preprocessed_data_dir_prefix = os.path.join(
            'preprocessed_data', self._modis_product, 
            str(self._reservoir_index), self._used_band)
        
        self._preprocessed_data_dir = os.path.join(
            self._preprocessed_data_dir_prefix, self._preprocessed_type)
        
        self._mask_data_dir = os.path.join(
            'masked_data', self._modis_product, 
            str(self._reservoir_index), self._used_band)
        
        self._data_augment_dir_prefix = os.path.join(
            self._dir_prefix, str(self._input_timesteps), 
            str(self._output_timesteps), str(self._crop_size))
        self._data_augment_dir = os.path.join(
            'data_augment', self._data_augment_dir_prefix)

        if self._crop_size == -1:
            self._data_augment_merged_dir = self._data_augment_dir
        else:
            self._data_augment_merged_dir = os.path.join(
                'data_augment_merged', self._data_augment_dir_prefix)
        
        # Other parameters
        self._day_period = 8 if self._modis_product == 'ALL' else 16
        self._n_data_per_year = 365//self._day_period + 1
        self._list_years = list(range(year_range[0], year_range[1] + 1))
        self._list_years_train = self._list_years[:-7]
        self._list_years_val = self._list_years[-7:-4]
        self._list_years_test = self._list_years[-4:]
    
        self._data_files = self._get_data_files()
        
        # Model parameters
        self._batch_size = batch_size
        self._model_name = model_name
        self._train_filenames = None
        self._val_filenames = None
        self._train_batch_generator = None
        self._val_batch_generator = None
        self._num_training_samples = None
        self._num_validation_samples = None

        self.model_utils = get_model_utils(self._model_name, self._output_timesteps)

        if os.path.exists(self._data_augment_merged_dir):
            self._set_generator()
        if model_keras:
            self.model_path = '{}.h5'.format(self._model_name)
        self._model = None
        self._compile_params = compile_params
        
        self._model_prefix = os.path.join(
            self._data_augment_dir_prefix, self._model_name)
        self._weights_dir = os.path.join('weights', self._model_prefix)
        self._result_dir = os.path.join('result', self._model_prefix)
        self._predict_dir = os.path.join('predict', self._model_prefix)
        
        self._monitor = monitor
        self._monitor_mode = monitor_mode
        self._filepath = None
        self._checkpoint = None
        self._csv_logger = None
        self._callbacks_list = None
        
        self.history = None
        if training:
            self._set_checkpoint()

        self.inference_model = None


        # Strategy objects
        self._preprocess_strategy_context = PreprocessStrategyContext(self)

        # Mask lake
        self._groundtruth_mask_lake_dir = os.path.join(
            'groundtruth_mask_lake', self._dir_prefix)
        self._predict_mask_lake_dir = os.path.join(
            'predict_mask_lake', self._model_prefix)
        
        
    def _get_data_files(self):
        data_types = ['train', 'val', 'test']
        file_types = ['data', 'target', 'mask']
        outputs = {}
        for data_type in data_types:
            outputs[data_type] = {}
            for file_type in file_types:
                outputs[data_type][file_type] = get_data_file_path(
                    self._preprocessed_data_dir, self._used_band, 
                    self._input_timesteps, self._output_timesteps,
                    data_type, file_type)
        return outputs
                

    def create_water_cloud_mask(self):
        if self._preprocessed_type == 'not_preprocessed':
            shutil.copytree('../sequence_output/masked_data', 'masked_data')
        elif self._preprocessed_type == 'Zhang':
            create_one_only_mask(
                self._raw_data_dir, '', self._year_range,
                self._n_data_per_year, self._day_period, self._mask_data_dir,
                self._resize_input)
        else:
            create_water_cloud_mask(
                self._raw_data_dir, self._used_band, self._year_range,
                self._n_data_per_year, self._day_period, self._mask_data_dir,
                self._resize_input)

    
    def preprocess_data(self):
        self._preprocess_strategy_context.preprocess_data(self)

    
    def make_archive_masked_data(self):
        make_archive('masked_data', 'zip', '.', self._mask_data_dir)
        
    def make_archive_preprocessed_data(self):
        make_archive('preprocessed_data', 'zip', '.', self._preprocessed_data_dir)
    
    
    def create_data_file(self):
        outputs = create_data_file_continuous_years(
            self._preprocessed_data_dir, self._input_timesteps,
            self._output_timesteps, self._list_years_train, self._list_years_val,
            self._list_years_test, self._mask_data_dir)
        make_archive('data_file', 'zip', '.', 'data_file')
        return outputs
    
    
    def augment_data(self):
        for data_type in ['train', 'val']:
            data_augment_dir = os.path.join(self._data_augment_dir, data_type)
            augment_one_reservoir_without_cache(
                self._data_files, data_augment_dir, 
                self._crop_size, self._n_samples, data_type,
                self._input_timesteps, self._output_timesteps)
            
            if self._crop_size != -1:
                data_augment_merged_dir = os.path.join(self._data_augment_merged_dir, data_type)
                merge_data_augment(data_augment_dir, data_augment_merged_dir, self._original_batch_size)

        self._set_generator()
        self._num_training_samples = len(self._train_filenames)*self._original_batch_size
        self._num_validation_samples = len(self._val_filenames)*self._original_batch_size
    
        
    def make_archive_augment_data(self):
        make_archive('data_augment_merged', 'zip', '.', self._data_augment_merged_dir)
    
    
    def _set_generator(self):        
        train_dir = os.path.join(self._data_augment_merged_dir, 'train')
        self._train_filenames = [os.path.join(train_dir, data_index)
                                 for data_index in os.listdir(train_dir)]
        self._train_batch_generator = self.model_utils.get_generator(
            self._train_filenames, self._batch_size,
            self._original_batch_size, self._pretrained)
        
        val_dir = os.path.join(self._data_augment_merged_dir, 'val')
        self._val_filenames = [os.path.join(val_dir, data_index)
                               for data_index in os.listdir(val_dir)]
        self._val_batch_generator = self.model_utils.get_generator(
            self._val_filenames, self._batch_size,
            self._original_batch_size, self._pretrained)
        self._num_training_samples = len(self._train_filenames)*self._original_batch_size
        self._num_validation_samples = len(self._val_filenames)*self._original_batch_size
        
    def get_train_generator(self):
        if self._train_batch_generator is None:
            self._set_generator
        return self._train_batch_generator
    
    def get_val_generator(self):
        if self._train_batch_generator is None:
            self._set_generator
        return self._val_batch_generator
    
    def create_model(self):
        K.clear_session()
        self._model = self.model_utils.create_model(self)
        return self._model
    
    def plot_model(self):
        if self._model is not None:
            plot_model(self._model, to_file='{}.png'.format(
                self._model_name), show_shapes=True)
            model_plot = misc.imread('{}.png'.format(self._model_name))
            plt.figure(figsize=(15,15))
            plt.imshow(model_plot)
            plt.show()
            
    def summary_model(self):
        if self._model is not None:
            self._model.summary()
            
    def plot_inference_model(self):
        if self.inference_model is not None:
            plot_model(self.inference_model, to_file='{}_inference.png'.format(
                self._model_name), show_shapes=True)
            model_plot = misc.imread('{}_inference.png'.format(self._model_name))
            plt.figure(figsize=(15,15))
            plt.imshow(model_plot)
            plt.show()
            
    def summary_inference_model(self):
        if self.inference_model is not None:
            self.inference_model.summary()
            
    def _set_checkpoint(self):
        if not os.path.exists(self._weights_dir):
            os.makedirs(self._weights_dir)
        self._filepath = os.path.join(self._weights_dir, "weights-{epoch:03d}.h5")
        self._checkpoint = ModelCheckpoint(
            self._filepath, monitor=self._monitor, mode=self._monitor_mode, verbose=1, period=5)
        self._csv_logger = CSVLogger(os.path.join(
            self._weights_dir, 'log.csv'), append=True, separator=';')
        self._callbacks_list = [self._checkpoint, self._csv_logger]
        if self._lr_reducer:
            self._callbacks_list = [self._lr_reducer] + self._callbacks_list

        timestamps = datetime.now()
        timestamps = str(timestamps)
        timestamps = timestamps[:timestamps.find('.')]
        timestamps = timestamps.replace(' ', '_')
        tensorboard_logdir = 'logs/{}'.format(timestamps)
        tensorboard = TensorBoard(log_dir=tensorboard_logdir)
        self._callbacks_list.append(tensorboard)

    def train(self, epochs=50, TPU_WORKER=None):
        if self._TPU_FLAG and TPU_WORKER is not None:
            self._model = tf.contrib.tpu.keras_to_tpu_model(
                self._model,
                strategy=tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

        self.history = self._model.fit_generator(
            generator=self._train_batch_generator,
            steps_per_epoch=(self._num_training_samples // self._batch_size),
            epochs=epochs,
            validation_data=self._val_batch_generator,
            validation_steps=(self._num_validation_samples // self._batch_size),
            callbacks=self._callbacks_list)
        self._model.save(self.model_path)
    
    def get_inference_model(self, img_height, img_width):
        K.clear_session()
        if os.path.exists(self.model_path):
            self.inference_model = self.model_utils._create_model(
                img_height, img_width, self._input_timesteps,
                self._compile_params, self._pretrained, 3)
            self.inference_model.load_weights(self.model_path)
            return self.inference_model
        return None
    
    def _get_inference_path(self, data_type='test', idx=0):
        return os.path.join(
            self._predict_dir, data_type, '{}.dat'.format(idx))

    def set_inference_model(self, inference_model):
        self.inference_model = inference_model

    def get_n_tests(self, data_type='test'):
        file_path = self._data_files[data_type]['data']
        return len(get_data_paths(file_path))

    def inference(self, data_type='test', idx=0, model=None):
        if self.inference_model is None:
            if model is None:
                model = self._model
            self.inference_model = model
        assert self.inference_model is not None
        
        output = self.model_utils.inference(self, data_type, idx)
        cache_data(output, self._get_inference_path(data_type, idx))
        return output


    def inference_all(self, data_types=None, model=None):
        if data_types is None:
            data_types = ['train', 'val', 'test']
        for data_type in data_types:
            if self.inference_model is None:
                if model is None:
                    model = self._model
                self.inference_model = model
            assert self.inference_model is not None
            n = self.get_n_tests(data_type)
            for idx in range(n):
                self.inference(data_type, idx)

    def get_inference(self, data_type='test', idx=0, model=None):
        inference_path = self._get_inference_path(data_type, idx)
        if not os.path.exists(inference_path):
            self.inference(data_type, idx)
        return restore_data(inference_path)

    def _get_yearday(self, data_type, idx):
        file_path = self._data_files[data_type]['target']
        filename = get_target_paths(file_path)[idx][0]
        yearday = filename.split('/')[-2]
        return yearday

    def get_groundtruth(self, data_type, idx):
        target_file_path = self._data_files[data_type]['target']
        return get_target_test(target_file_path, idx)

    def get_water_cloud_mask(self, data_type, idx):
        mask_file_path = self._data_files[data_type]['mask']
        return get_target_test(mask_file_path, idx)
    
    def eval(self, data_type='test', idx=0, metric=None):
        return self.model_utils.eval(self, data_type, idx, metric)
    
    def eval_all(self, data_type='test', metric=None):
        eval_list = []
        n = self.get_n_tests(data_type)
        for i in range(n):
            eval_list.append(self.eval(data_type, i, metric))
        return eval_list

    def visualize_result(self, data_type='test', idx=0):
        return self.model_utils.visualize_result(self, data_type, idx)

    def create_groundtruth_mask_lake(self):
        if self._preprocessed_type == 'Zhang':
            self._groundtruth_mask_lake_dir = self._preprocessed_data_dir
            return
        if not os.path.exists(self._groundtruth_mask_lake_dir):
            create_groundtruth_mask_lake(
                self._raw_data_dir, self._used_band, self._year_range, self._n_data_per_year,
                self._day_period, self._groundtruth_mask_lake_dir, self._resize_input)

    def get_groundtruth_mask_lake(self, data_type='test', idx=0):
        yearday = self._get_yearday(data_type, idx)
        year = yearday[:-3]
        groundtruth_mask_lake_path = os.path.join(
            self._groundtruth_mask_lake_dir, year, yearday, 'masked.dat')
        if self._preprocessed_type == 'Zhang':
            groundtruth_mask_lake_path = os.path.join(
            self._groundtruth_mask_lake_dir, year, yearday, 'preprocessed.dat')
        if os.path.exists(groundtruth_mask_lake_path):
            return restore_data(groundtruth_mask_lake_path)
        else:
            return None

    def create_predict_mask_lake(self, data_type='test'):
        if self._preprocessed_type == 'not_preprocessed' or self._preprocessed_type == 'Zhang':
            shutil.copytree('predict', 'predict_mask_lake')
        else:
            self.model_utils.create_predict_mask_lake(self, data_type)

    def get_predict_mask_lake(self, data_type='test', idx=0):
        predict_mask_lake_path = os.path.join(
            self._predict_mask_lake_dir, data_type, '{}.dat'.format(idx))
        if os.path.exists(predict_mask_lake_path):
            return restore_data(predict_mask_lake_path)
        else:
            return None

    def make_archive_predict(self):
        filename = 'predict_{}'.format(self._model_name)
        make_archive(filename, 'zip', '.', self._predict_dir)
        return filename

    def make_archive_result(self):
        filename = 'result_{}'.format(self._model_name)
        make_archive(filename, 'zip', '.', self._result_dir)
        return filename

