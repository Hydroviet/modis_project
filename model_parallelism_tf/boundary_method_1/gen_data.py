# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from modis_utils.misc import cache_data, restore_data


# In[3]:


input_dir = '../sequence_data/12/'
output_dir = 'sequence_data/12'

center_point_xs = np.arange(16, 513, 32)
center_point_xs


# In[8]:


water_threshold = -0.9680795202031212


# In[9]:


from modis_utils.image_processing import mask_lake_img


# In[10]:


from skimage.segmentation import find_boundaries


# In[11]:


def find_boundaries_mask_lake(x, water_threshold):
    x1 = mask_lake_img(x, offset=water_threshold)
    return find_boundaries(x1)

percentile = restore_data('../percentile/0.dat')


permanent_water_area = np.where(percentile > 0.8, 1, 0)


# In[17]:


def find_boundaries_mask_lake(x, water_threshold):
    x1 = mask_lake_img(x, offset=water_threshold)
    x1 = np.logical_or(x1, permanent_water_area)
    return find_boundaries(x1)


# In[122]:


def get_pos(boundaries_img, center_point_xs):
    a = np.where(boundaries_img)
    res = []
    for x in center_point_xs:
        b = np.where(a[1] == x)[0]
        if len(b) > 1:
            choose_x1 = b[0]
            choose_x2 = b[-1]
            res.append((a[0][choose_x1], a[1][choose_x1]))
            res.append((a[0][choose_x2], a[1][choose_x2]))
    return res


# In[22]:


def get_patch_coor(center_pos, sz):
    sz /= 2
    res = (center_pos[0] - sz, center_pos[0] + sz - 1, center_pos[1] - sz, center_pos[1] + sz - 1)
    return list(map(lambda x: int(x), res))


# In[138]:


def gen_data_1(sequence_data, permanent_water_area, water_threshold, patch_size=32, output_path=None):
    inputs, targets, inputs_pw, targets_pw = sequence_data
    last_inputs_pw = inputs[-1]
    a = find_boundaries_mask_lake(last_inputs_pw, water_threshold)
    list_center_pos = get_pos(a, center_point_xs)

    patches_inputs = []
    patches_targets = []
    patches_inputs_pw = []
    patches_targets_pw = []

    outputs = [patches_inputs, patches_targets, patches_inputs_pw, patches_targets_pw]

    def padding(x, sz):
        res = np.zeros((sz, sz))
        res[:x.shape[0], :x.shape[1]] = x
        return res

    for center_pos in list_center_pos:
        if len(center_pos) < 2:
            continue
        x1, x2, y1, y2 = get_patch_coor(center_pos, patch_size)
        for origin, patches in zip(sequence_data, outputs):
            patch = origin[:, x1 : x2+1, y1 : y2+1]
            if x2 - x1 < patch_size - 1 or y2 - y1 < patch_size - 1:
                patch = padding(patch, patch_size)
            patches.append(np.expand_dims(patch, axis=0))
    res = []
    for patches in outputs:
        res.append(np.vstack(patches))
    if output_path is not None:
        cache_data(res, output_path)
    return res


def gen_data(input_dir, list_filenames, data_type, permanent_water_area, water_threshold, patch_size, output_dir):
    input_dir = os.path.join(input_dir, data_type)
    output_dir = os.path.join(output_dir, data_type)
    for filename in list_filenames:
        output_path = os.path.join(output_dir, filename)
        input_path = os.path.join(input_dir, filename)
        sequence_data = restore_data(input_path)
        gen_data_1(sequence_data, permanent_water_area, water_threshold, patch_size, output_path)

'''
path = '../sequence_data/12/train/0.dat'
a = restore_data(path)
res = gen_data_1(a, permanent_water_area, water_threshold)
for x in res:
    print(x.shape)
'''

import multiprocessing as mp

def main():
    n_cores = 24
    patch_size = 32

    for data_type in ('train', 'val', 'test'):
        processes = []
        input_data_type_dir = os.path.join(input_dir, data_type)
        output_data_type_dir = os.path.join(output_dir, data_type)
        if not os.path.exists(output_data_type_dir):
            os.makedirs(output_data_type_dir)
        list_filenames_1 = os.listdir(input_data_type_dir)
        n = len(list_filenames_1)
        m = n // n_cores
        r = n % n_cores
        start_pos = 0
        for i in range(n_cores):
            q = m + 1 if i < r else m
            list_filenames = list_filenames_1[start_pos : start_pos + q]
            p = mp.Process(target=gen_data, args=(input_dir, list_filenames, data_type, permanent_water_area,
                                                  water_threshold, patch_size, output_dir))
            processes.append(p)
            start_pos += q

        for p in processes:
            p.start()
        for p in processes:
            p.join()

if __name__ == '__main__':
    main()
