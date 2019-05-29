# In[1]:


import argparse
import functools
import itertools
import os


# In[2]:


import bcl
import bcl_model
import numpy as np
import tensorflow as tf


# In[3]:


tf.logging.set_verbosity(tf.logging.INFO)
used_gpus = [0, 1, 2, 3]
s = str(used_gpus[0])
for used_gpu in used_gpus[1:]:
    s += ',{}'.format(used_gpu)

s1 = "\"{}\"".format(s)
s1


# In[4]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# In[5]:


import cifar10_utils
import six


# In[6]:


def get_model_fn(num_gpus, variable_strategy, num_workers):
    """Returns a function that will build the resnet model."""

    def _bcl_model_fn(features, labels, mode, params):
        """Resnet model body.
        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
           manages gradient updates.
        Args:
          features: a list of tensors, one for each tower
          labels: a list of tensors, one for each tower
          mode: ModeKeys.TRAIN or EVAL
          params: Hyperparameters suitable for tuning
        Returns:
          A EstimatorSpec object.
        """
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        weight_decay = params.weight_decay
        momentum = params.momentum

        tower_inputs = features
        tower_groundtruths = labels
        tower_losses = []
        tower_gradvars = []
        tower_preds = []

        # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
        # on CPU. The exception is Intel MKL on CPU which is optimal with
        # channels_last.
        data_format = params.data_format
        if not data_format:
            if num_gpus == 0:
                data_format = 'channels_last'
            else:
                data_format = 'channels_first'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = cifar10_utils.local_device_setter(
                    worker_device=worker_device)
            elif variable_strategy == 'GPU':
                device_setter = cifar10_utils.local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                        num_gpus, tf.contrib.training.byte_size_load_fn))
            with tf.variable_scope('bcl', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, gradvars, preds = _tower_fn(
                            is_training, weight_decay, tower_inputs[i], tower_groundtruths[i],
                            data_format, params.num_layers, params.batch_norm_decay,
                            params.batch_norm_epsilon)
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            # Only trigger batch_norm moving mean and variance update from
                            # the 1st tower. Ideally, we should grab the updates from all
                            # towers but these stats accumulate extremely fast so we can
                            # ignore the other stats from the other towers without
                            # significant detriment.
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                           name_scope)

        # Now compute global loss and gradients.
        gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                gradvars.append((avg_grad, var))

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):
            # Suggested learning rate scheduling from
            # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
            num_batches_per_epoch = bcl.BCLDataSet.num_examples_per_epoch(
                'train') // (params.train_batch_size * num_workers)
            boundaries = [
                num_batches_per_epoch * x
                for x in np.array([82, 123, 300], dtype=np.int64)
            ]
            staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.002]]

            learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                                        boundaries, staged_lr)

            loss = tf.reduce_mean(tower_losses, name='loss')

            examples_sec_hook = cifar10_utils.ExamplesPerSecondHook(
                params.train_batch_size, every_n_steps=10)

            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=params.learning_rate, momentum=momentum)

            tensors_to_log = {'loss': loss}

            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=100)

            train_hooks = [logging_hook, examples_sec_hook]

            if params.sync:
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer, replicas_to_aggregate=num_workers)
                sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
                train_hooks.append(sync_replicas_hook)

            # Create single grouped train op
            train_op = [
                optimizer.apply_gradients(
                    gradvars, global_step=tf.train.get_global_step())
            ]
            train_op.extend(update_ops)
            train_op = tf.group(*train_op)

            predictions = tf.concat(tower_preds, axis=0)
            groundtruths = tf.concat(labels, axis=0)
            metrics = {
                'mse':
                    tf.metrics.mean_squared_error(groundtruths, predictions)
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            training_hooks=train_hooks,
            eval_metric_ops=metrics)

    return _bcl_model_fn


# In[7]:


def _tower_fn(is_training, weight_decay, inputs, groundtruths, data_format,
              num_layers, batch_norm_decay, batch_norm_epsilon):
    """Build computation tower (Resnet).
    Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    data_format: channels_last (NHWC) or channels_first (NCHW).
    num_layers: number of layers, an int.
    batch_norm_decay: decay for batch normalization, a float.
    batch_norm_epsilon: epsilon for batch normalization, a float.
    Returns:
    A tuple with the loss for the tower, the gradients and parameters, and
    predictions.
    """
    model = bcl_model.BCL(
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        is_training=is_training,
        data_format=data_format)
    #tower_pred = model.forward_pass(inputs, input_data_format='channels_last')
    tower_pred = model.forward_pass(inputs)

    tower_loss = tf.losses.mean_squared_error(
        labels=groundtruths, predictions=tower_pred)
    tower_loss = tf.reduce_mean(tower_loss)

    model_params = tf.trainable_variables()
    #tower_loss += weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in model_params])

    tower_grad = tf.gradients(tower_loss, model_params)

    return tower_loss, zip(tower_grad, model_params), tower_pred


# In[8]:


def input_fn(data_dir,
             subset,
             num_shards,
             batch_size,
             use_distortion_for_training=True):
    """Create input graph for model.
        Args:
        data_dir: Directory where TFRecords representing the dataset are located.
        subset: one of 'train', 'validate' and 'eval'.
        num_shards: num of towers participating in data-parallel training.
        batch_size: total batch size for training to be divided by the number of
        shards.
        use_distortion_for_training: True to use distortions.
        Returns:
        two lists of tensors for features and labels, each of num_shards length.
    """
    with tf.device('/cpu:0'):
        use_distortion = subset == 'train' and use_distortion_for_training
        dataset = bcl.BCLDataSet(data_dir, subset, use_distortion)
        image_batch, label_batch = dataset.make_batch(batch_size)
        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            return [image_batch], [label_batch]

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.
        image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
        image_shards = [[] for i in range(num_shards)]
        label_shards = [[] for i in range(num_shards)]
        for i in range(batch_size):
            idx = i % num_shards
            image_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
        image_shards = [tf.parallel_stack(x) for x in image_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        return image_shards, label_shards


# In[9]:


def get_experiment_fn(data_dir,
                      num_gpus,
                      variable_strategy,
                      use_distortion_for_training=True):
    """Returns an Experiment function.
    Experiments perform training on several workers in parallel,
    in other words experiments know how to invoke train and eval in a sensible
    fashion for distributed training. Arguments passed directly to this
    function are not tunable, all other arguments should be passed within
    tf.HParams, passed to the enclosed function.
    Args:
      data_dir: str. Location of the data for input_fns.
      num_gpus: int. Number of GPUs on each worker.
      variable_strategy: String. CPU to use CPU as the parameter server
      and GPU to use the GPUs as the parameter server.
      use_distortion_for_training: bool. See cifar10.Cifar10DataSet.
    Returns:
      A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->
      tf.contrib.learn.Experiment.
      Suitable for use by tf.contrib.learn.learn_runner, which will run various
      methods on Experiment (train, evaluate) based on information
      about the current runner in `run_config`.
    """

    def _experiment_fn(run_config, hparams):
        """Returns an Experiment."""
        # Create estimator.
        train_input_fn = functools.partial(
            input_fn,
            data_dir,
            subset='train',
            num_shards=num_gpus,
            batch_size=hparams.train_batch_size,
            use_distortion_for_training=use_distortion_for_training)

        val_input_fn = functools.partial(
            input_fn,
            data_dir,
            subset='val',
            batch_size=hparams.val_batch_size,
            num_shards=num_gpus)

        train_steps = hparams.train_steps

        estimator = tf.estimator.Estimator(
            model_fn=get_model_fn(num_gpus, variable_strategy,
                                  run_config.num_worker_replicas or 1),
                                  config=run_config,
                                  params=hparams)

        #train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps)
        #eval_spec = tf.estimator.EvalSpec(input_fn=val_input_fn)
        #tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        # Create experiment.
        return tf.contrib.learn.Experiment(
            estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=val_input_fn,
            train_steps=train_steps,
            eval_steps=100,
            min_eval_frequency=10)

    return _experiment_fn


# In[17]:


def main(job_dir, data_dir, num_gpus, variable_strategy,
         use_distortion_for_training, log_device_placement, num_intra_threads,
         **hparams):
    # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    print('hparams:', hparams)

    # Session configuration.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=log_device_placement,
        intra_op_parallelism_threads=num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = cifar10_utils.RunConfig(
        session_config=sess_config, model_dir=job_dir,
        save_checkpoints_steps=hparams["eval_steps"])
    tf.contrib.learn.learn_runner.run(
        get_experiment_fn(data_dir, num_gpus, variable_strategy,
                          use_distortion_for_training),
        run_config=config,
        schedule="train_and_evaluate",
        hparams=tf.contrib.training.HParams(
            is_chief=config.is_chief,
            **hparams))


# In[18]:


args = {
    "data_dir": "sequence_data/12",
    "job_dir": "tmp/bcl",
    "variable_strategy": "CPU",
    "num_gpus": len(used_gpus),
    "num_layers": 4,
    "train_steps": 15000,
    "eval_steps": 100,
    "train_batch_size": 128,
    "val_batch_size": 128,
    "momentum": 0.9,
    "weight_decay": 2e-4,
    "learning_rate": 0.001,
    "use_distortion_for_training": False,
    "sync": False,
    "num_intra_threads": 0,
    "num_inter_threads": 0,
    "data_format": "channels_last",
    "log_device_placement": False,
    "batch_norm_decay": 0.997,
    "batch_norm_epsilon": 1e-5
}


# In[19]:


main(**args)


# In[ ]:




