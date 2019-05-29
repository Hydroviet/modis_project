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
input_path = sys.argv[2]
batch_size = int(sys.argv[3])
steps = int(sys.argv[4])
model_paths_path = sys.argv[5]
n = int(sys.argv[6])

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

input_test = restore_data(input_path)
model_paths = restore_data(model_paths_path)

def predict_multisteps_single_point(input_single_point, point_id, steps, batch_size, gpu_id):
    K.clear_session()
    model = load_model(model_paths[point_id])
    n_test = len(input_single_point)
    res = np.zeros((n_test, steps))
    inputs = input_single_point.copy()
    for i in range(steps):
        predict = model.predict(inputs, batch_size=batch_size)
        inputs = np.concatenate([inputs[:, 1:, :], predict.reshape(-1, 1, 1)], axis=1)
        res[:, i:i+1] = predict
    return res

def main():
    m = n // 4
    if gpu_id < 3:
        list_idx = list(range(gpu_id*m, (gpu_id + 1)*m))
    else:
        list_idx = list(range(gpu_id*m, n))
    res = []
    for idx in list_idx:
        res_idx = predict_multisteps_single_point(input_test[:, :, idx : idx+1], idx, steps, batch_size, gpu_id)
        res.append(np.expand_dims(res_idx, axis=-1))
        with open('tmp/log_{}.txt'.format(gpu_id), 'a') as f:
            f.write(str(idx) + '\n')
    cache_data(res, 'tmp/out_{}.dat'.format(gpu_id))
    
if __name__ == '__main__':
    main()