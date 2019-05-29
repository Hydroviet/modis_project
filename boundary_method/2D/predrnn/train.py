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
from pathlib import Path
from skimage.measure import compare_ssim

import predrnn_pp
import dataset_utils
from nets import models_factory
from data_provider import datasets_factory
from utils import preprocess
from utils import metrics


# In[2]:


os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
NUM_GPUS = 2

# In[3]:


data_dir = '../one_output/data_patch'
model_name = 'predrnn_pp'
save_dir = 'results/predrnn_pp'
input_length = 14
output_length = 1
img_width = 32
img_channel = 1
stride = 1
filter_size = 5
num_hidden = [128, 64, 64, 1]
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


# In[4]:


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
    "snapshot_interval" : snapshot_interval
}


# In[5]:


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    predictions = predrnn_pp.rnn(features, params["num_layers"], params["num_hidden"], params["filter_size"],
                                 params["stride"], params["seq_length"], params["input_length"], params["layer_norm"])
    predictions = predictions[:, params["input_length"]-1:]
    print("predictions.shape =", predictions.shape)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
        metrics = {
            'mse': tf.metrics.mean_squared_error(labels=labels, predictions=predictions)
        }

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=params["lr"])
                .minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            raise NotImplementedError()


# In[6]:


def input_fn(data_dir, subset, batch_size,
             use_distortion_for_training=True):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = dataset_utils.ConvLSTMDataSet(data_dir, subset, use_distortion)
    return dataset.make_batch(batch_size)

def input_fn_multigpus(data_dir, subset, batch_size,
             use_distortion_for_training=True):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = dataset_utils.ConvLSTMDataSet(data_dir, subset, use_distortion)
    dataset = dataset.make_batch(batch_size, True)
    return dataset.prefetch(NUM_GPUS)


# In[7]:


train_inpf = functools.partial(input_fn_multigpus, data_dir, 'train', batch_size)
eval_inpf = functools.partial(input_fn_multigpus, data_dir, 'val', batch_size)

strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
cfg = tf.estimator.RunConfig(save_checkpoints_steps=save_checkpoints_steps,
                             train_distribute=strategy)
estimator = tf.estimator.Estimator(model_fn, save_dir, cfg, params)

tf.logging.set_verbosity(tf.logging.INFO)

Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
hook = tf.contrib.estimator.stop_if_no_increase_hook(
    estimator, 'mse', 10000, min_steps=10000, run_every_secs=600)
train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)


# In[ ]:


tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# In[ ]:




