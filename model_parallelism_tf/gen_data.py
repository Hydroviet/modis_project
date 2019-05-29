import os
import sys
import csv
import math
import numpy as np
from datetime import datetime
from shutil import copytree, ignore_patterns, unpack_archive

from modis_utils.misc import cache_data, restore_data
from modis_utils.image_processing import get_pixel_weights
from modis_utils.misc import restore_data, cache_data, normalize_data
from modis_utils.misc import get_data_paths, get_target_paths, get_data_from_data_file, get_target_from_target_file


if os.path.isfile('requirement.txt'):
    os.system('pip install -r requirement.txt')

water_threshold = -0.9680795202031212
timesteps = 47
real_timesteps = 3 + (timesteps - 3)//4
n_out = 12
n_data_per_year = 46
n_samples = 10000

'''
water_threshold = -0.9680795202031212
timesteps = 7
real_timesteps = 3 + (timesteps - 3)//4
n_out = 2
n_data_per_year = 46
n_samples = 5
'''

preprocess_data_dir = 'preprocessed_data'
pixel_weights_dir = 'pixel_weights'
sequence_data_dir = 'sequence_data'
sequence_data_dir = os.path.join(sequence_data_dir, str(n_out))

list_filenames = sorted(os.listdir(preprocess_data_dir))
pws = [os.path.join(pixel_weights_dir, filename) for filename in list_filenames]
list_filenames = [os.path.join(preprocess_data_dir, filename) for filename in list_filenames]

n_val = n_data_per_year
n_test = n_data_per_year*2 - n_out + 1
n_train = len(list_filenames) - 3*n_data_per_year - timesteps

train_id = (0, n_train + timesteps)
val_id = (n_train, n_train + n_val + timesteps)
test_id = (n_train + n_val, len(list_filenames) - n_out + 1)
idx = {}
idx['train'] = train_id
idx['val'] = val_id
idx['test'] = test_id


def get_inputs_targets_pw(list_filenames, pws, idx_data_type, n_out):
    list_filenames_data_type = list_filenames[idx_data_type[0]:idx_data_type[1] + n_out - 1]
    pws_data_type = pws[idx_data_type[0]:idx_data_type[1] + n_out - 1]
    list_inputs = []
    list_targets = []
    list_pw_inputs = []
    list_pw_targets = []
    for i in range(timesteps, len(list_filenames_data_type) - n_out + 1):
        list_inputs.append(list_filenames_data_type[i - timesteps : i])
        list_targets.append(list_filenames_data_type[i : i + n_out])
        list_pw_inputs.append(pws_data_type[i - timesteps : i])
        list_pw_targets.append(pws_data_type[i : i + n_out])
    return list_inputs, list_targets, list_pw_inputs, list_pw_targets

def create_data_file():
    data_file_dir = 'data_file'
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)

    return_paths = {}
    for data_type in ('train', 'val', 'test'):
        input_filename = os.path.join(data_file_dir, '{}_input.csv'.format(data_type))
        target_filename = os.path.join(data_file_dir, '{}_target.csv'.format(data_type))
        pw_input_filename = os.path.join(data_file_dir, '{}_pw_input.csv'.format(data_type))
        pw_target_filename = os.path.join(data_file_dir, '{}_pw_target.csv'.format(data_type))

        list_inputs, list_targets, list_pw_inputs, list_pw_targets = \
            get_inputs_targets_pw(list_filenames, pws, idx[data_type], n_out)

        input_f = open(input_filename, 'w')
        input_writer = csv.writer(input_f)
        target_f = open(target_filename, 'w')
        target_writer = csv.writer(target_f)
        pw_input_f = open(pw_input_filename, 'w')
        pw_input_writer = csv.writer(pw_input_f)
        pw_target_f = open(pw_target_filename, 'w')
        pw_target_writer = csv.writer(pw_target_f)

        for row in list_inputs:
            input_writer.writerow(row)
        input_f.close()
        for row in list_targets:
            target_writer.writerow(row)
        target_f.close()
        for row in list_pw_inputs:
            pw_input_writer.writerow(row)
        pw_input_f.close()
        for row in list_pw_targets:
            pw_target_writer.writerow(row)
        pw_target_f.close()

        return_paths[data_type] = {'input': input_filename,
                                   'target': target_filename,
                                   'pw_input': pw_input_filename,
                                   'pw_target': pw_target_filename}
    return return_paths

def select_img(list_imgs):
    n = len(list_imgs)
    res = list_imgs[0].copy()
    for img in list_imgs[1:]:
        res += img
    return res/n

def select_data(sequence_data):
    res = []
    for i in range(0, len(sequence_data) - 3, 4):
        selected_img = select_img(sequence_data[i : i+4])
        res.append(np.expand_dims(selected_img, axis=0))
    res = np.vstack(res)
    res = np.vstack([res, sequence_data[-3:]])
    return res

def create_sequence_data(data_file_paths, n_samples=10000):
   if not os.path.exists(sequence_data_dir):
      os.makedirs(sequence_data_dir)

   for data_type, n_data in zip(('train', 'val', 'test'), (n_train, n_val, n_test)):
        sequence_data_type_dir = os.path.join(sequence_data_dir, data_type)
        if not os.path.exists(sequence_data_type_dir):
            os.makedirs(sequence_data_type_dir)
        data_type_file_paths = data_file_paths[data_type]
        input_file = data_type_file_paths['input']
        target_file = data_type_file_paths['target']
        pw_target_file = data_type_file_paths['pw_target']
        for i in range(min(n_data, n_samples)):
            inputs = select_data(get_data_from_data_file(input_file, i))
            target = get_data_from_data_file(target_file, i)
            input_pixel_weights = np.array(list(map(lambda x: get_pixel_weights(x, water_threshold), inputs)))
            target_pixel_weights = get_data_from_data_file(pw_target_file, i)
            cache_data((inputs, target, input_pixel_weights, target_pixel_weights),
                       os.path.join(sequence_data_type_dir, '{}.dat'.format(i)))

'''
def create_sample_sequence_data(n_samples):
    for data_type, n_data in zip(('train',), (n_samples,)):
        sequence_data_type_dir = os.path.join(sequence_data_dir, data_type)
        if not os.path.exists(sequence_data_type_dir):
            os.makedirs(sequence_data_type_dir)
        data_type_file_paths = data_file_paths[data_type]
        input_file = data_type_file_paths['input']
        target_file = data_type_file_paths['target']
        pw_input_file = data_type_file_paths['pw_input']
        pw_target_file = data_type_file_paths['pw_target']
        for i in range(min(n_data, n_samples)):
            inputs = get_data_from_data_file(input_file, i)[-sample_timesteps:]
            target = get_data_from_data_file(target_file, i)
            input_pixel_weights = get_data_from_data_file(pw_input_file, i)[-sample_timesteps:]
            target_pixel_weights = get_data_from_data_file(pw_target_file, i)
            cache_data((inputs, target, input_pixel_weights, target_pixel_weights),
                       os.path.join(sequence_data_type_dir, '{}.dat'.format(i)))
'''


def main():
    print(n_train, n_val, n_test)
    print(idx)

    if not os.path.exists(sequence_data_dir):
        print('Generating data')
        data_file_paths = create_data_file()
        create_sequence_data(data_file_paths, n_samples)


if __name__ == '__main__':
    main()
