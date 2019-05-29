import os
import sys
import numpy as np
from skimage.segmentation import find_boundaries

sys.path.append('../../../../')
from modis_utils.misc import cache_data, restore_data
from modis_utils.image_processing import mask_lake_img


water_threshold = (0.1 + 0.2001)/1.2001
percentile = restore_data('../../percentile/0.dat')
center_point_xs = np.arange(16, 513, 32)

permanent_water_area = np.where(percentile > 0.8, 1, 0)


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


def find_boundaries_mask_lake(x, water_threshold):
    x1 = mask_lake_img(x, offset=water_threshold)
    x1 = np.logical_or(x1, permanent_water_area)
    return find_boundaries(x1)


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


def get_patch_coor(center_pos, sz):
    sz /= 2
    res = (center_pos[0] - sz, center_pos[0] + sz - 1, center_pos[1] - sz, center_pos[1] + sz - 1)
    return list(map(lambda x: int(x), res))


def gen_boundary_patch(inputs, patch_size, water_threshold=water_threshold):
    last_inputs = inputs[-1]
    a = find_boundaries_mask_lake(last_inputs, water_threshold)
    list_center_pos = get_pos(a, center_point_xs)
    
    patches = []
    coords = []
    
    for center_pos in list_center_pos:
        if len(center_pos) < 2:
            continue
        coord = get_patch_coor(center_pos, patch_size)
        coords.append(coord)
        x1, x2, y1, y2 = coord
        patch = inputs[:, x1 : x2+1, y1 : y2+1]
        if patch.shape[1] < patch_size - 1 or patch.shape[2] < patch_size - 1:
            continue
        patches.append(np.expand_dims(patch, axis=0))
    return np.vstack(patches), coords
