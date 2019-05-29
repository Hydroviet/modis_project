import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from modis_utils.misc import cache_data, restore_data

def restore_data_batch(list_path):
    res = [[], [], [], []]
    for path in list_path:
        data = restore_data(path)
        for x, t in zip(data, res):
            t.append(np.expand_dims(np.expand_dims(x, axis=0), axis=-1))
    res_1 = []
    for t in res:
        res_1.append(np.vstack(t))
    return res_1

def tf_scale(x, scale_range):
    min_x = tf.reduce_min(x)
    max_x = tf.reduce_max(x)
    l = scale_range[1] - scale_range[0]
    return (x - min_x)/(max_x - min_x)*l + scale_range[0]

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_block_idx_per_gpu(num_blocks, gpus):
    res = []
    n = num_blocks // gpus
    m = num_blocks % gpus
    current_idx = 0
    for i in range(m):
        res.append([current_idx, current_idx + n + 1])
        current_idx += n + 1
    for i in range(gpus - m):
        res.append([current_idx, current_idx + n])
        current_idx += n
    return res

def get_list_filenames(data_dir, data_type):
    list_filenames = os.listdir(os.path.join(data_dir, data_type))
    list_filenames = sorted(list_filenames, key=lambda x: int(os.path.basename(x)[:-4]))
    return [os.path.join(data_dir, data_type, filename) for filename in list_filenames]
