import os
import numpy as np

from modis_utils.image_processing import mask_cloud_and_water
from modis_utils.misc import get_predict_dir, get_predict_mask_dir
from modis_utils.misc import cache_data, restore_data, get_im
from modis_utils.misc import get_target_test, get_data_file_path, get_target_paths
from modis_utils.misc import get_reservoir_min_max, get_reservoir_mean_std
from modis_utils.image_processing import mask_lake_img


def get_predict_mask_lake(data_dir, used_band, crop_size, time_steps, filters,
                          kernel_size, n_hidden_layers, mask_cloud_loss,
                          reservoir_index, test_index):
    predict_dir = get_predict_dir(data_dir, reservoir_index, used_band, crop_size, time_steps,
                                  filters, kernel_size, n_hidden_layers, mask_cloud_loss)
    predict = restore_data(os.path.join(predict_dir, '{}.dat'.format(test_index)))
    if 'div' not in data_dir:
        reservoir_min, reservoir_max = get_reservoir_min_max(data_dir, reservoir_index)
        mean, std = get_reservoir_mean_std(data_dir, reservoir_index)
        predict = np.interp(predict, (np.min(predict), np.max(predict)), 
                            (reservoir_min, reservoir_max))
        predict = predict*std + mean
    else:
        predict = np.interp(predict, (np.min(predict), np.max(predict)), (-2001, 10000))
    predict_mask = mask_lake_img(predict, band=used_band)
    return predict_mask


def get_groundtruth_mask_lake(data_dir, used_band, time_steps,
                              reservoir_index, test_index, data_type='test'):
    target_file_path = get_data_file_path(data_dir, reservoir_index, used_band,
                                          time_steps, data_type, 'target')
    test_path = get_target_paths(target_file_path)[test_index]
    token = test_path.split('/')
    raw_path = '/'.join(['raw_data'] + token[3:])
    img = get_im(raw_path[:-4] + '.tif')
    return mask_lake_img(img, band=used_band)
