import os
import argparse
import functools
import numpy as np
import tensorflow as tf
from pathlib import Path

import utils
import dataset_utils
import convlstm_model


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = convlstm_model.ConvLSTM(
                is_training=is_training,
                data_format='channels_last',
                batch_norm_decay=params["batch_norm_decay"],
                batch_norm_epsilon=params["batch_norm_epsilon"])
    predictions = model.forward_pass(features)

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
            train_op = tf.train.AdamOptimizer(learning_rate=params["starter_learning_rate"])\
                .minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            raise NotImplementedError()


def input_fn(data_dir, subset, batch_size,
             use_distortion_for_training=True):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = dataset_utils.ConvLSTMDataSet(data_dir, subset, use_distortion)
    return dataset.make_batch(batch_size)

#def mse(labels, predictions):
#    return tf.metrics.mean_squared_error(labels=labels, predictions=predictions)


def main(gpu_id, data_dir, batch_size, use_distortion_for_training,
         save_checkpoints_steps, checkpoint_dir, **params):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    train_inpf = functools.partial(input_fn, data_dir, 'train', batch_size)
    eval_inpf = functools.partial(input_fn, data_dir, 'val', batch_size)

    cfg = tf.estimator.RunConfig(save_checkpoints_steps=save_checkpoints_steps)
    estimator = tf.estimator.Estimator(model_fn, checkpoint_dir, cfg, params)

    tf.logging.set_verbosity(tf.logging.INFO)

    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'mse', 10000, min_steps=10000, run_every_secs=600)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--data_dir', default='../data_patch')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_distortion_for_training', type=bool, default=False)
    parser.add_argument('--save_checkpoints_steps', type=int, default=100)
    parser.add_argument('--checkpoint_dir', default='result/model')
    parser.add_argument('--batch_norm_decay', type=float, default=0.997)
    parser.add_argument('--batch_norm_epsilon', type=float, default=1e-5)
    parser.add_argument('--starter_learning_rate', type=float, default=1e-4)

    args = parser.parse_args()

    main(**vars(args))
