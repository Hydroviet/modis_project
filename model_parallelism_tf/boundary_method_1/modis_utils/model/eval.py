import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from modis_utils.misc import get_data_test, get_target_test


def predict_and_visualize_by_data_file_one_output(
        data_file_path, target_file_path, pred_img, which=0, 
        result_dir=None, model=None):
    example = get_data_test(data_file_path, which)
    time_steps = example.shape[0]
    groundtruth = get_target_test(target_file_path, which)
    
    plt.figure(figsize=(10, 10))
    if model is not None and pred is None:
        pred = model.predict(example[np.newaxis, :, :, :, np.newaxis])
    
    G = gridspec.GridSpec(2, time_steps)
    pred_img
    
    for i, img in enumerate(example):
        axe = plt.subplot(G[0, i])
        axe.imshow(img)

    ax_groundtruth = plt.subplot(G[1, :time_steps//2])
    ax_groundtruth.imshow(groundtruth)
    ax_groundtruth.set_title('groundtruth')
    
    ax_pred = plt.subplot(G[1, time_steps//2:2*(time_steps//2)])
    ax_pred.imshow(pred_img)
    ax_pred.set_title('predict')

    if result_dir is not None:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        plt.savefig(os.path.join(result_dir, '{}.png'.format(which))) 
    
    return (groundtruth, pred_img)


def predict_and_visualize_by_data_file_sequence_output(
        data_file_path, target_file_path, pred, which=0, 
        result_dir=None, model=None):
    pass


def predict_and_visualize_by_data_file_one_output_and_gridding(
        data_file_path, target_file_path, model, which=0, 
        result_dir=None):
    input_seq = get_data_test(data_file_path, which)
    if real_time_steps is not None:
        time_steps = real_time_steps
        input_seq = input_seq[input_seq.shape[0] - real_time_steps:, :, :]
    else:
        time_steps = input_seq.shape[0]
    groundtruth = get_target_test(target_file_path, which)
    if mask_file_path is not None:
        mask_groundtruth = get_target_test(mask_file_path, which)
        mask_groundtruth[mask_groundtruth == -1] = 0
    groundtruth = scale_normalized_data(groundtruth, groundtruth_range)
    
    plt.figure(figsize=(10, 10))
    
    offset_x = input_seq.shape[1] % crop_size
    offset_y = input_seq.shape[2] % crop_size
    input_seq = input_seq[:, offset_x//2:-(offset_x - offset_x//2), \
                          offset_y//2:-(offset_y - offset_y//2)]
    groundtruth = groundtruth[offset_x//2:-(offset_x - offset_x//2), \
                              offset_y//2:-(offset_y - offset_y//2)]

    pred_img = np.zeros_like(groundtruth)

    for i in range(input_seq.shape[1] // crop_size):
        for j in range(input_seq.shape[2] // crop_size):
            pred = model.predict(input_seq[np.newaxis, :, \
                                 i*crop_size:(i+1)*crop_size, \
                                 j*crop_size:(j+1)*crop_size, np.newaxis])
            pred_img[i*crop_size:(i+1)*crop_size, \
                     j*crop_size:(j+1)*crop_size] = pred[1][0,:,:,0]
    
    G = gridspec.GridSpec(2, time_steps)
    
    for i, img in enumerate(input_seq):
        axe = plt.subplot(G[0, i])
        axe.imshow(img)

    ax_groundtruth = plt.subplot(G[1, :time_steps//2])
    ax_groundtruth.imshow(groundtruth)
    ax_groundtruth.set_title('groundtruth')
    
    ax_pred = plt.subplot(G[1, time_steps//2:2*(time_steps//2)])
    ax_pred.imshow(pred_img)
    ax_pred.set_title('predict')

    if result_dir is not None:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        plt.savefig(os.path.join(result_dir, '{}.png'.format(which))) 
    return (groundtruth, pred_img)


def predict_and_visualize_by_data_file_sequence_output_and_gridding(
        data_file_path, target_file_path, pred, which=0, 
        result_dir=None, model=None):
    pass