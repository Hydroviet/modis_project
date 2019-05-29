import os
import sys
import h5py
import pickle
import subprocess
import numpy as np
from time import time
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Process
import subprocess
from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler

import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, LSTM, BatchNormalization
from keras.models import Model, load_model

sys.path.append('..')

from modis_utils.misc import cache_data, restore_data

gpu_id = int(sys.argv[1])
data_path = sys.argv[2]
epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])
timesteps = int(sys.argv[5])
units = int(sys.argv[6])
n = int(sys.argv[7])
model_dir = sys.argv[8]
training_fig_dir = sys.argv[9]

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

data = restore_data(data_path)

def create_model(timesteps, units):
    input_shape = (timesteps, units)
    inputs = Input(input_shape)
    x = LSTM(units*4, return_sequences=True)(inputs)
    x = LSTM(units*4, return_sequences=True)(inputs)
    x = LSTM(units)(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='mse', optimizer='adam')
    return model

def train(data, i, epochs, batch_size, timesteps, units, model_dir, training_fig_dir):
    K.clear_session()
    model_path = os.path.join(model_dir, '{}.dat'.format(i))
    model = create_model(timesteps, units)
    history = model.fit(data['train_X'][:, :, i : i+1], data['train_y'][:, i : i+1],
                        epochs=epochs, batch_size=batch_size, verbose=1,
                        validation_data=(data['val_X'][:, :, i : i+1], data['val_y'][:, i : i+1]))
    plt.figure()
    plt.plot(history.history['loss'], color='r', label='train loss')
    plt.plot(history.history['val_loss'], color='b', label='val loss')
    plt.legend()
    plt.title('Training history feature {}'.format(i))
    plt.savefig(os.path.join(training_fig_dir, '{}.png'.format(i)))
    model.save(model_path)
    return model_path

def main():
    m = n // 4
    if gpu_id < 3:
        list_idx = list(range(gpu_id*m, (gpu_id + 1)*m))
    else:
        list_idx = list(range(gpu_id*m, n))
    for idx in list_idx:
        train(data, idx, epochs, batch_size, timesteps, units, model_dir, training_fig_dir)
    
if __name__ == '__main__':
    main()