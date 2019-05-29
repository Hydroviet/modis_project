#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import os.path
import time
import numpy as np
import tensorflow as tf
import cv2
import sys
import random
import functools
from tqdm import tqdm
from pathlib import Path
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt

from inference_whole_img_utils import InferenceMTConvLSTMWholeImg
from gen_data_inference_utils import gen_boundary_patch, select_data


# In[2]:


sys.path.append('../../../../')


# In[3]:


from modis_utils.misc import cache_data, restore_data


# In[4]:


os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# In[5]:


if not os.path.exists('inferences'):
    os.makedirs('inferences')


# In[6]:


data_dir = '../data_patch'
model_name = 'predrnn_pp'
save_dir = 'result/model'
input_length = 14
output_length = 1
img_width = 32
img_channel = 1
stride = 1
filter_size = 3
num_hidden = [16, 16, 16, 1]
num_layers = len(num_hidden)
patch_size = 4
layer_norm = True
lr = 0.001
reverse_input = False
batch_size = 8
max_iterations = 80000
display_interval = 1
test_interval = 2000
snapshot_interval = 10000

save_checkpoints_steps = 100
whole_img_width = 513
batch_norm_decay = 0.997
batch_norm_epsilon = 1e-5


# In[7]:


params = {
    "data_dir" : data_dir,
    "model_name" :  model_name,
    "save_dir" : save_dir,
    "input_length" : input_length,
    "output_length" : output_length,
    "seq_length" : input_length + output_length,
    "img_width" : img_width,
    "img_channel" : img_channel,
    "stride" : stride,
    "filter_size" : filter_size,
    "num_hidden" : num_hidden,
    "num_layers" : num_layers,
    "patch_size" : patch_size,
    "layer_norm" : layer_norm,
    "lr" : lr,
    "reverse_input" : reverse_input,
    "batch_size" : batch_size,
    "max_iterations" : max_iterations,
    "display_interval" : display_interval,
    "test_interval" : test_interval,
    "snapshot_interval" : snapshot_interval,
    "whole_img_width" : whole_img_width,
    "batch_norm_decay" : batch_norm_decay,
    "batch_norm_epsilon" : batch_norm_epsilon
}


# In[8]:


timesteps = 47


# In[ ]:





# In[9]:


inputs_np_whole_img = restore_data('../../multiscale_predrnn/data/sequence_data/test/0.dat')[0]
inputs_np_whole_img.shape


# In[10]:


inference_mt_convlstm_whole_img = InferenceMTConvLSTMWholeImg(params)


# In[11]:


np_input_dir = '../../multiscale_predrnn/data/sequence_data'
steps_ahead = 80

for subset in ('test', 'val'):
    inference_dir = 'inferences/{}'.format(subset)
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
    np_input_dir_subset = os.path.join(np_input_dir, subset)
    n = len(os.listdir(np_input_dir_subset))
    for i in tqdm(range(n)):
        inputs_np = restore_data(os.path.join(np_input_dir_subset, '{}.dat'.format(i)))[0]
        inputs_np = inputs_np[-timesteps:]
        res = inference_mt_convlstm_whole_img.get_inference_from_np_array(inputs_np, steps_ahead)
        cache_data(res, os.path.join(inference_dir, '{}.dat'.format(i)))
