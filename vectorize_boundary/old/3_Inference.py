import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import shutil
import datetime
import tensorflow as tf
from shutil import unpack_archive, make_archive

from modis_utils.misc import restore_data
from modis_utils.modis_utils import ModisUtils
from modis_utils.model.loss_function import PSNRLoss, lossSSIM, SSIM, step_decay
from modis_utils.model.loss_function import mse_with_mask_tf, mse_with_mask_tf_1, mse_with_mask


# Parameters
config_path = 'config.dat'
config_params = restore_data(os.path.join(gdrive_dir, config_path))

# sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9)
#lrate = LearningRateScheduler(step_decay)

lr = config_params['lr']

training = False
crop_size = config_params['crop_size']
input_timesteps = config_params['input_timesteps']
output_timesteps = config_params['output_timesteps']
batch_size = config_params['batch_size']
compile_params = config_params['compile_params']
model_name = config_params['model_name']
preprocessed_type = config_params['preprocessed_type']
modis_product = config_params['modis_product']
monitor = config_params['monitor']
monitor_mode = config_params['monitor_mode']
resize_input = config_params['resize_input']

raw_data_dir = config_params['raw_data_dir']
reservoir_index = config_params['reservoir_index']
used_band = config_params['used_band']
year_range = config_params['year_range']
model_keras = config_params['model_keras']
original_batch_size = config_params['original_batch_size']

TPU_FLAG = False
# End Parameters


modis_utils = ModisUtils(
    raw_data_dir=raw_data_dir,
    modis_product=modis_product,
    reservoir_index=reservoir_index,
    preprocessed_type=preprocessed_type,
    used_band=used_band,
    crop_size=crop_size,
    input_timesteps=input_timesteps,
    output_timesteps=output_timesteps,
    year_range=year_range,
    model_name=model_name,
    batch_size=batch_size,
    model_keras=model_keras,
    compile_params=compile_params,
    original_batch_size=original_batch_size,
    TPU_FLAG=TPU_FLAG,
    training=training,
    monitor=monitor,
    monitor_mode=monitor_mode,
    resize_input=resize_input)

inference_model = modis_utils.get_inference_model(513, 513)
modis_utils.plot_inference_model()

a = modis_utils.get_inference('test')
assert a is not None

modis_utils.inference_all(['test'])

predict_file = modis_utils.make_archive_predict()

