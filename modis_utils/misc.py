import os
import csv
import h5py
import pickle
import numpy as np
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt

#List of years for MOD13Q1 only
TRAIN_LIST_YEARS_DEFAULT = [2000, 2001, 2002, 2003, 2004,
                            2005, 2006, 2007, 2008, 2009,
                            2010, 2011]
VAL_LIST_YEARS_DEFAULT = [2012, 2013, 2014]
TEST_LIST_YEARS_DEFAULT = [2015, 2016, 2017]

'''
# List of year for combined MOD13Q1 and MYD13Q1
TRAIN_LIST_YEARS_DEFAULT = [2003, 2004, 2005, 2006, 2007,
                            2008, 2009, 2010, 2011]
VAL_LIST_YEARS_DEFAULT = [2012, 2013, 2014]
TEST_LIST_YEARS_DEFAULT = [2015, 2016, 2017]
'''

def to_str(x):
    if not isinstance(x, list):
        return str(x)
    else:
        b = [str(y) for y in x]
        return '_'.join(b)


def get_data_file_dir(data_dir, used_band, input_time_steps, output_time_steps):
    return os.path.join('data_file', data_dir, str(input_time_steps),
                        str(output_time_steps))


def get_data_file_path(data_dir, used_band, input_time_steps,
                       output_time_steps, data_type, file_type):
    return os.path.join(get_data_file_dir(data_dir, used_band,
                                          input_time_steps, output_time_steps),
                        '{}_{}.csv'.format(data_type, file_type))


def get_predict_dir(data_dir, used_reservoir, used_band, crop_size,
                    time_steps, filters, kernel_size, n_hidden_layers,
                    mask_cloud_loss):
    return os.path.join('predict', get_result_dir_suffix(
        data_dir, used_reservoir, used_band, crop_size, time_steps,
        to_str(filters), to_str(kernel_size), to_str(n_hidden_layers), mask_cloud_loss))


def get_predict_mask_dir(data_dir, used_reservoir, used_band, crop_size,
                        time_steps, filters, kernel_size, n_hidden_layers,
                        mask_cloud_loss):
    return os.path.join('predict_mask', get_result_dir_suffix(
        data_dir, used_reservoir, used_band, crop_size, time_steps,
        filters, kernel_size, n_hidden_layers, mask_cloud_loss))


# Data
def cache_data(data, path):
    """Save data (numpy array) to disk."""
    dir_prefix = os.path.dirname(path)
    if dir_prefix != '' and not os.path.exists(dir_prefix):
        os.makedirs(dir_prefix)

    if not isinstance(data, dict):
        h5f = h5py.File(path, 'w')
        if not isinstance(data, tuple) and not isinstance(data, list):
            data = [data]
        for i, x in enumerate(data):
            h5f.create_dataset(str(i), data=x)
        h5f.close()
    else:
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()



def to_float32(x):
    if not isinstance(x, tuple):
        return np.float32(x)
    b = []
    for a in x:
        b.append(to_float32(a))
    return tuple(b)


def restore_data(path, convert_to_float32=False):
    if path[-4:] != '.dat' and path[-3:] != '.h5':
        return get_im(path)
    try:
        h5f = h5py.File(path, 'r')
        list_of_names = []
        h5f.visit(list_of_names.append)
        data = []
        for name in list_of_names:
            data.append(h5f[name][:])
            if convert_to_float32 and data:
                data = to_float32(data)
        h5f.close()
    except:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    if len(data) == 1:
        return data[0]
    return data


def _get_data_path(data_file):
    with open(data_file, "r") as data_f:
        reader = csv.reader(data_f)
        data_paths = [row for row in reader]
    return data_paths


def get_data_paths(data_file):
    return _get_data_path(data_file)


def _get_target_mask_path(target_file):
    with open(target_file, "r") as target_f:
        reader = csv.reader(target_f)
        target_paths = [row for row in reader]
    return target_paths


def get_target_paths(target_file):
    return _get_target_mask_path(target_file)


def get_mask_paths(mask_file):
    return _get_target_mask_path(mask_file)


def get_im(path, reduce_size=None):
    # reduce_size type is a tuple, example: (128, 96)
    img = misc.imread(path)
    # Reduce size
    if reduce_size is not None:
        img = misc.imresize(img, reduce_size)
    return img


def normalize_data(data, mean=0.0, std=1.0):
    if std == 0:
        return data - mean
    return (data - mean) / std

def scale_data(data, original_range=(-1.0,1.0), range=(-0.2001,1.0)):
    return np.interp(data, original_range, range)

def scale_normalized_data(normalized_data, range=(-1.0,1.0)):
    return np.interp(normalized_data, (-0.2001, 1.0), range)

def scale_data_tf(data_tf, original_range=(-1.0, 1.0), output_range=(-0.2001, 1.0)):
    original_diff = original_range[1] - original_range[0]
    output_diff = output_range[1] - output_range[0]
    data_zero_one_scale = tf.divide(tf.subtract(data_tf, original_range[0]), original_diff)
    return tf.add(tf.multiply(data_zero_one_scale, output_diff), output_range[0])

def scale_data_with_scaler(data, scaler):
    """Scale data to [0, 1].

    Args:
        data: Data need to be scaled, prefers numpy array.
        scaler: MinMaxScaler object

    Returns:
    Data after scaled.
    """
    if (data.ndim == 5):
        x = data.reshape(data.shape[0]*data.shape[1]*data.shape[2],
                         data.shape[3]*data.shape[4])
    else:
        x = data.reshape(data.shape[0]*data.shape[1],
                         data.shape[2]*data.shape[3])
    scale_x = scaler.transform(x)
    return scale_x.reshape(data.shape)


def _get_list_data_in_one_year(data_dir, used_band, year, mask=False):
    if mask:
        filename = 'masked.dat'
    else:
        filename = 'preprocessed.dat'
    year_dir = os.path.join(data_dir, str(year))
    list_folders = os.listdir(year_dir)
    list_folders = sorted(list_folders, key=lambda x: int(x))
    return [os.path.join(year_dir, x, filename) for x in list_folders]


def _create_data_file_continuous_years(data_dir,
                                       input_time_steps,
                                       output_time_steps,
                                       list_years,
                                       data_type,
                                       mask_data_dir,
                                       used_band=''):
    input_file = get_data_file_path(data_dir, used_band, input_time_steps,
                                    output_time_steps, data_type, 'data')
    target_file = get_data_file_path(data_dir, used_band, input_time_steps,
                                     output_time_steps, data_type, 'target')
    mask_file = get_data_file_path(data_dir, used_band, input_time_steps,
                                   output_time_steps, data_type, 'mask')
    input_f = open(input_file, 'w')
    target_f = open(target_file, 'w')
    mask_f = open(mask_file, 'w')
    writer_input = csv.writer(input_f)
    writer_target = csv.writer(target_f)
    writer_mask = csv.writer(mask_f)

    list_data = []
    list_mask = []

    if list_years[0] > 2000:
        list_data_prev_year = _get_list_data_in_one_year(
            data_dir, used_band, list_years[0] - 1)
        list_data += list_data_prev_year[-input_time_steps:]
        list_mask_prev_year = _get_list_data_in_one_year(
            mask_data_dir, used_band, list_years[0] - 1, True)
        list_mask += list_mask_prev_year[-input_time_steps:]

    for year in list_years:
        list_data += _get_list_data_in_one_year(data_dir, used_band, year)
        list_mask += _get_list_data_in_one_year(mask_data_dir, used_band, year, True)

    if list_years[-1] < 2018 and output_time_steps > 1:
        list_data_next_year = _get_list_data_in_one_year(
            data_dir, used_band, list_years[-1] + 1)
        list_data += list_data_next_year[:output_time_steps - 1]
        list_mask_next_year = _get_list_data_in_one_year(
            mask_data_dir, used_band, list_years[-1] + 1, True)
        list_mask += list_mask_next_year[:output_time_steps - 1]

    n_data = len(list_data) - (input_time_steps + output_time_steps) + 1
    for i in range(input_time_steps, n_data + input_time_steps):
        list_data_in_window = list_data[i - input_time_steps : i]
        list_target_in_window = list_data[i : i + output_time_steps]
        list_mask_in_window = list_mask[i : i + output_time_steps]
        writer_input.writerow(list_data_in_window)
        writer_target.writerow(list_target_in_window)
        writer_mask.writerow(list_mask_in_window)

    input_f.close()
    target_f.close()
    mask_f.close()


def create_data_file_continuous_years(data_dir='raw_data/MOD13Q1',
                                      input_time_steps=12,
                                      output_time_steps=1,
                                      list_years_train=None,
                                      list_years_val=None,
                                      list_years_test=None,
                                      mask_data_dir='masked_data/MOD13Q1',
                                      used_band=''):
    """Create files containing path of images.

    If you already created those files but now you change list_years,
    you must remove all old files and recreate them.

    Example:
        create_data_file_continuous_years(data_dir='raw_data/MOD13Q1/0',
                                          used_band='NDVI',
                                          input_time_steps=12,
                                          output_time_steps=1,
                                          list_years_train=None,
                                          list_years_val=None,
                                          list_years_test=None,
                                          mask_data_dir='masked_data/MOD13Q1'):

    Args:
        data_dir: Directory where stores image data.
        input_time_steps: Input Time steps (length) of LSTM sequence.
        output_time_steps: Output Time steps (length) of LSTM sequence.
        list_years_train: List years of data used for train, use None if
            want to use default range.
        list_years_val: List years of data used for validation.
        list_years_test: List years of data used for test.
        mask_data_dir: Directory where stores masked images.
        used_band: A string represents name of used band.

    Returns:
        A dictionary stores paths of data files.
    """
    list_years = {}
    data_types = ['train', 'val', 'test']
    file_types = ['data', 'target']
    if mask_data_dir is not None:
        file_types.append('mask')

    if list_years_train is None:
        list_years['train'] = TRAIN_LIST_YEARS_DEFAULT
    else:
        list_years['train'] = list_years_train

    if list_years_val is None:
        list_years['val'] = VAL_LIST_YEARS_DEFAULT
    else:
        list_years['val'] = list_years_val

    if list_years_test is None:
        list_years['test'] = TEST_LIST_YEARS_DEFAULT
    else:
        list_years['test'] = list_years_test

    data_file_dir = get_data_file_dir(data_dir, used_band,
                                      input_time_steps, output_time_steps)

    outputs = {}
    for data_type in data_types:
        outputs[data_type] = {}
        for file_type in file_types:
            outputs[data_type][file_type] = get_data_file_path(
                data_dir, used_band, input_time_steps,
                output_time_steps, data_type, file_type)

    # Check whether all needed files are created
    if os.path.isdir(data_file_dir):
        if mask_data_dir is None:
            return outputs
        elif os.path.isfile(outputs['train']['mask']):
            return outputs

    # Not all files are created, Create/Recreate them.
    try:
        os.makedirs(data_file_dir)
    except:
        pass
    for data_type in data_types:
        _create_data_file_continuous_years(data_dir, input_time_steps,
                                           output_time_steps, list_years[data_type],
                                           data_type, mask_data_dir, '')
    return outputs


def get_data_merged_from_paths(data_paths, target_paths, mask_paths):
    list_data = []
    for data_path in data_paths:
        list_data.append(np.expand_dims(restore_data(data_path), axis=0))
    for target_path in target_paths:
        list_data.append(np.expand_dims(restore_data(target_path), axis=0))
    for mask_path in mask_paths:
        list_data.append(np.expand_dims(restore_data(mask_path), axis=0))
    data_merged = np.concatenate(list_data, axis=0)
    data_merged = np.expand_dims(data_merged, axis=0)
    return data_merged


def get_target_from_target_file(target_file_path, which):
    target_paths_list = _get_target_mask_path(target_file_path)
    target_paths = target_paths_list[which]
    if len(target_paths) == 1:
        return restore_data(target_paths[0])
    else:
        list_target = []
        for path in target_paths:
            list_target.append(np.expand_dims(restore_data(path), axis=0))
        return np.concatenate(list_target, axis=0)


def get_data_from_data_file(data_file_path, which):
    data_paths_list = _get_data_path(data_file_path)
    data_paths = data_paths_list[which]
    list_data = []
    for path in data_paths:
        list_data.append(np.expand_dims(restore_data(path), axis=0))
    return np.concatenate(list_data, axis=0)


def get_data_test(data_file_path, which):
    return get_data_from_data_file(data_file_path, which)


def get_target_test(target_file_path, which):
    return get_target_from_target_file(target_file_path, which)


def get_reservoir_min_max(data_dir, reservoir_index):
    min_max_path = os.path.join('min_max', data_dir, 'min_max.dat')
    min_max = restore_data(min_max_path)
    reservoir_min = min_max[reservoir_index]['min']
    reservoir_max = min_max[reservoir_index]['max']
    return reservoir_min, reservoir_max


def get_reservoir_mean_std(data_dir, reservoir_index):
    data_dir = data_dir.strip('/')
    token = data_dir.split('/')
    preprocessed_dir = '/'.join(token[:2])
    modis_product = token[-1]
    mean_std_path = os.path.join('mean_std', preprocessed_dir, 'change_fill_value', modis_product,
                                 'mean_std.dat')
    mean_std_dict = restore_data(mean_std_path)
    mean_std_dict_reservoir = mean_std_dict[reservoir_index]
    mean = mean_std_dict_reservoir['mean']
    std = mean_std_dict_reservoir['std']
    return mean, std


def get_max_value(data_dir, used_band, list_years, n_data_per_year, day_period):
    MAX = 0
    for year in list_years:
        for d in range(n_data_per_year):
            day = d*day_period + 1
            prefix = os.path.join(str(year), str(year) + str(day).zfill(3))
            current_data_dir = os.path.join(data_dir, prefix)
            try:
                list_imgs = os.listdir(current_data_dir)
                filename = list(filter(lambda x: used_band in x, list_imgs))[0]
                img = restore_data(os.path.join(current_data_dir, filename))
                MAX = max(MAX, np.max(img))
            except:
                pass
    return MAX


def get_water_area(mask_lake, img_type='modis'):
    if img_type == 'modis':
        return np.sum(mask_lake)*0.25*0.25
    elif img_type == 'sar':
        return np.sum(mask_lake)*0.01*0.01
    else:
        return np.sum(mask_lake)


def plot_two_arrays(arr1, arr2, legend=None, title=None):
    if legend is None or not isinstance(legend, list) or len(legend) < 2:
        legend = ['predict', 'groundtruth']
    fig, ax = plt.subplots(1,1,figsize=(20,8))
    ax.plot(arr1, marker='o', c='r', linestyle='-', label=legend[0])
    ax.plot(arr2, marker='o', c='b', linestyle='-', label=legend[1])
    ax.legend()
    ax.set_xlabel('Day')
    ax.set_ylabel('Water Area (km^2)')

    if title is None:
        title = 'Groundtruth-Predict Water Area 2018'
    ax.set_title(title)
    plt.savefig(title)
    plt.show()
