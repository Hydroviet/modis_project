import os
import numpy as np
import multiprocessing as mp
from skimage.segmentation import find_boundaries

from modis_utils.misc import cache_data, restore_data
from modis_utils.misc import get_data_merged_from_paths
from modis_utils.misc import get_data_paths, get_target_paths, get_mask_paths


def _random_crop_func(x, offset_x, offset_y, crop_size=32, random_crop=True):
    return x[:,:, offset_y : offset_y+crop_size, offset_x : offset_x+crop_size]


def _merge_data_augment(n_data, data_index_shuffle, list_data, merge_data_dir,
                        batch_size, thread_id, n_threads):
    m = n_data//batch_size
    k = m//n_threads
    i_range_of_cur_thread = range(thread_id*k, (thread_id+1)*k)
    for i in i_range_of_cur_thread:
        merge_data = []
        merge_target_mask = []
        for j in data_index_shuffle[i*batch_size : (i+1)*batch_size]:
            data = restore_data(list_data[j])
            merge_data.append(np.expand_dims(data[0], axis=0))
            if data[1].shape[0] > 1:
                data_1 = np.expand_dims(data[1], axis=0)
            else:
                data_1 = data[1]
            merge_target_mask.append(data_1)
        if len(merge_data) == batch_size:
            merge_data = np.vstack(merge_data)
            merge_target_mask = np.vstack(merge_target_mask)
            merge_data_path = os.path.join(merge_data_dir, '{}.dat'.format(i))
            cache_data((merge_data, merge_target_mask), merge_data_path)
        

def _merge_last_data_augment(n_data, data_index_shuffle, list_data, merge_data_dir, 
                             batch_size, thread_id, n_threads):
    m = n_data//batch_size
    k = m - m % n_threads
    i = k + thread_id
    merge_data = []
    merge_target_mask = []
    for j in data_index_shuffle[i*batch_size : (i+1)*batch_size]:
        data = restore_data(list_data[j])
        merge_data.append(np.expand_dims(data[0], axis=0))
        if data[1].shape[0] > 1:
            data_1 = np.expand_dims(data[1], axis=0)
        else:
            data_1 = data[1]
        merge_target_mask.append(data_1)
    if len(merge_data) == batch_size:
        merge_data = np.vstack(merge_data)
        merge_target_mask = np.vstack(merge_target_mask)
        merge_data_path = os.path.join(merge_data_dir, '{}.dat'.format(i))
        cache_data((merge_data, merge_target_mask), merge_data_path)
    

def _merge_data_with_last(merge_data_dir, n_data, data_index_shuffle, list_data, 
                          batch_size, n_threads):
    try:
        os.makedirs(merge_data_dir)
    except:
        pass
    
    processes = [mp.Process(target=_merge_data_augment, 
                            args=(n_data, data_index_shuffle, list_data,
                                  merge_data_dir, batch_size, thread_id, 
                                  n_threads))
                 for thread_id in range(n_threads)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    processes = [mp.Process(target=_merge_last_data_augment,
                            args=(n_data, data_index_shuffle, list_data, 
                                  merge_data_dir, batch_size, thread_id,
                                  n_threads))
                 for thread_id in range(n_threads)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def merge_data_augment(data_augment_dir, data_augment_merged_dir,
                       batch_size=1024, n_threads=mp.cpu_count() - 2):
    list_data = [os.path.join(data_augment_dir, data_index) for data_index 
                  in os.listdir(data_augment_dir)]
    n_data = len(list_data)
    index_shuffle = np.random.permutation(n_data)
    
    _merge_data_with_last(data_augment_merged_dir, n_data, index_shuffle,
                          list_data, batch_size, n_threads)


def augment_one_reservoir_without_cache(data_files, data_augment_dir,
                                        crop_size, n_samples, data_type,
                                        input_time_steps, output_timesteps):
    """Generate random crop augmentation on reservoir.
    Returns:
        Boolean, True if augment successfully and False vice versa.
    """
    data_file = data_files[data_type]['data']
    data_paths = get_data_paths(data_file)

    target_file = data_files[data_type]['target']
    target_paths = get_target_paths(target_file)

    mask_file = data_files[data_type]['mask']
    mask_paths = get_mask_paths(mask_file)

    if crop_size == -1:
        _generate_whole_image(data_paths, target_paths, mask_paths,
                              data_augment_dir, input_time_steps,
                              output_timesteps)
    elif crop_size <= 80:
        _generate_on_boundaries(data_paths, target_paths, mask_paths,
                                data_augment_dir, crop_size, n_samples,
                                input_time_steps, output_timesteps)
    else:
        _generate(data_paths, target_paths, mask_paths,
                  data_augment_dir, crop_size, n_samples,
                  input_time_steps, output_timesteps)
    return True


def _generate_on_boundaries(data_paths, target_paths, mask_paths,
                            data_augment_dir, crop_size, n_samples,
                            input_time_steps, output_timesteps):
    n_data = len(data_paths)
    cnt = 0
    half_crop_size = crop_size//2
    for k in range(n_data):
        data_merged = get_data_merged_from_paths(data_paths[k], target_paths[k],
                                                 mask_paths[k])
        target_img = data_merged[0, -2*output_timesteps, :, :]
        h, w = data_merged.shape[2:4]
        already = set()
        boundaries = find_boundaries(target_img)
        pos = np.where(boundaries)
        n_pos = len(pos[0])
        ii = 1
        while n_pos < n_samples:
            target_img = data_merged[0, -2*output_timesteps + ii, :, :]
            already = set()
            boundaries = find_boundaries(target_img)
            pos = np.where(boundaries)
            n_pos = len(pos[0])
            ii += 1
        for i in range(n_samples):
            while True:
                offset_x = pos[0][np.random.randint(n_pos)]
                offset_y = pos[1][np.random.randint(n_pos)]
                if offset_x + half_crop_size + 1 < w and offset_y + half_crop_size + 1 < h and \
                  offset_x - half_crop_size >= 0 and offset_y - half_crop_size >= 0 and \
                  (offset_x, offset_y) not in already:
                    break
            already.add((offset_x, offset_y))
            batch = _random_crop_func(data_merged, offset_x - half_crop_size,
                    offset_y - half_crop_size, crop_size=crop_size)
            for j in range(batch.shape[0]):
                cur = batch[j]
                data = np.expand_dims(cur[:input_time_steps], axis=-1)
                target = np.expand_dims(cur[-2*output_timesteps:-output_timesteps], axis=-1)
                mask = np.expand_dims(cur[-output_timesteps:], axis=-1)
                target_mask = np.concatenate((target, mask), axis=-1)
                if not os.path.isdir(data_augment_dir):
                    os.makedirs(data_augment_dir)
                file_path = os.path.join(data_augment_dir,
                                         '{}.dat'.format(cnt))
                cnt += 1
                cache_data((data, target_mask), file_path)


def _generate(data_paths, target_paths, mask_paths,
              data_augment_dir, crop_size, n_samples,
              input_time_steps, output_timesteps):
    n_data = len(data_paths)
    cnt = 0
    for k in range(n_data):
        data_merged = get_data_merged_from_paths(data_paths[k], target_paths[k],
                                                 mask_paths[k])
        for i in range(n_samples):
            batch = _random_crop_func_1(data_merged, crop_size)
            for j in range(batch.shape[0]):
                cur = batch[j]
                data = np.expand_dims(cur[:input_time_steps], axis=-1)
                target = np.expand_dims(cur[-2*output_timesteps:-output_timesteps], axis=-1)
                mask = np.expand_dims(cur[-output_timesteps:], axis=-1)
                target_mask = np.concatenate((target, mask), axis=-1)
                if not os.path.isdir(data_augment_dir):
                    os.makedirs(data_augment_dir)
                file_path = os.path.join(data_augment_dir,
                                         '{}.dat'.format(cnt))
                cnt += 1
                cache_data((data, target_mask), file_path)


def _random_crop_func_1(x, crop_size=32, random_crop=True):
    '''x.shape = (n_data, time_step, img_height, img_width, channels) '''
    h, w = x.shape[2:4]
    offset_y = np.random.randint(h - crop_size)
    offset_x = np.random.randint(w - crop_size)
    return x[:,:, offset_y : offset_y+crop_size, offset_x : offset_x+crop_size]


def _generate_whole_image(data_paths, target_paths, mask_paths,
                          data_augment_dir, input_time_steps,
                          output_timesteps):
    n_data = len(data_paths)
    for k in range(n_data):
        data_merged = get_data_merged_from_paths(data_paths[k], target_paths[k],
                                                 mask_paths[k])
        data = data_merged[:, :-2*output_timesteps, :, :]
        data = np.expand_dims(data, axis=-1)
        target = data_merged[:, -2*output_timesteps:-output_timesteps, :, :]
        target = np.expand_dims(data, axis=-1)
        mask = data_merged[:, -output_timesteps:, :, :]
        mask = np.expand_dims(data, axis=-1)
        if target.shape[1] > 1:
            target = target.squeeze(axis=1)
            mask = mask.squeeze(axis=1)
        target_mask = np.concatenate((target, mask), axis=-1)
        if not os.path.isdir(data_augment_dir):
            os.makedirs(data_augment_dir)
        file_path = os.path.join(data_augment_dir, '{}.dat'.format(k))
        cache_data((data, target_mask), file_path)
