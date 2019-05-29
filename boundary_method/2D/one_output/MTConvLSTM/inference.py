import os
import sys
import functools
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
from tensorflow.contrib import predictor

import utils
import dataset_utils
import mt_convlstm_model


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    if isinstance(features, dict):
        features = features['feature']

    model = mt_convlstm_model.MTConvLSTM(
                filters=params['num_hidden'],
                is_training=is_training,
                data_format='channels_last',
                batch_norm_decay=params["batch_norm_decay"],
                batch_norm_epsilon=params["batch_norm_epsilon"])
    predictions = model.forward_pass(features)
    print("predictions.shape =", predictions.shape)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
        tensors_to_log = {'loss': loss}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)
        train_hooks = [logging_hook]

        metrics = {
            'mse': tf.metrics.mean_squared_error(labels=labels, predictions=predictions)
        }

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=params["lr"])\
                .minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            raise NotImplementedError()


def input_fn(data_dir, subset, batch_size,
             use_distortion_for_training=True):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = dataset_utils.ConvLSTMDataSet(data_dir, subset, use_distortion)
    return dataset.make_batch(batch_size)


class InferenceMTConvLSTM:

    def __init__(self, params):
        self.params = params

        self.test_inpf = functools.partial(input_fn, params['data_dir'], 'test', params['batch_size'])
        self.val_inpf = functools.partial(input_fn, params['data_dir'], 'val', params['batch_size'])
        self.train_inpf = functools.partial(input_fn, params['data_dir'], 'train', params['batch_size'])
        self.input_fn = {
            'train': self.train_inpf,
            'val': self.val_inpf,
            'test': self.test_inpf
        }

        cfg = tf.estimator.RunConfig()
        self.estimator = tf.estimator.Estimator(model_fn, params['save_dir'], cfg, params)

        def serving_input_receiver_fn():
            inputs = tf.placeholder(dtype=tf.float32,
                shape=[params['batch_size'], params['input_length'], params['img_width'],
                    params['img_width'], params['img_channel']], name='inputs')
            receiver_tensors = {'feature': inputs}
            features = inputs
            return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

        export_dir = 'saved_model'
        self.estimator.export_saved_model(export_dir, serving_input_receiver_fn)

        subdirs = [x for x in Path(export_dir).iterdir()
            if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        self.predict_fn = predictor.from_saved_model(latest)


    def evaluate(self, subset='test'):
        return self.estimator.evaluate(self.input_fn[subset],
                   steps=dataset_utils.ConvLSTMDataSet.num_examples_per_epoch(subset) // self.params['batch_size'])


    def get_inference_from_tfrecord(self, subset='test'):
        results = self.estimator.predict(self.input_fn[subset])
        inferences = []
        i = 0
        for result in tqdm(results):
            inferences.append(result)
            i += 1
            if i == dataset_utils.ConvLSTMDataSet.num_examples_per_epoch(subset):
                break
        return np.vstack(inferences).squeeze()


    def get_inference_from_np_array(self, inputs_np):
        # inputs_np: shape = (BxTxHxW)
        batch_size = self.params['batch_size']
        n = len(inputs_np)
        r = n % batch_size
        m = n // batch_size
        inputs_np = np.expand_dims(inputs_np, axis=-1)
        if r > 0:
            inputs_np = np.vstack([inputs_np, inputs_np[:batch_size - r]]) # Padding to have shape as multiple of batch_size
            m += 1
        results = []
        for i in range(m):
            result = self.predict_fn({'feature': inputs_np[i*batch_size : (i+1)*batch_size]})['output']
            results.append(result)
        results = np.vstack(results).squeeze()
        if r > 0:
            return results[: r-batch_size]
        return results
