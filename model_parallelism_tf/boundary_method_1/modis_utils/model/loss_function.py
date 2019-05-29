import numpy as np
import tensorflow as tf

# PSNR
def PSNRLoss(y_true_and_mask, y_pred):
    y_true, y_mask = tf.split(y_true_and_mask, 2, axis=-1)
    square_error = (y_true - y_pred)**2
    tf_mask = tf.where(tf.equal(y_mask, 0),
                       tf.fill(tf.shape(y_mask), 0),
                       tf.fill(tf.shape(y_mask), 1))
    tf_mask = tf.to_float(tf_mask)
    
    return -10. * tf.log(
        tf.divide(
            tf.reduce_sum(
                tf.multiply(tf_mask, square_error)),
                    tf.maximum(tf.reduce_sum(tf_mask), 1.0))) / tf.log(10.)
  
# SSIM
    
def lossSSIM(y_true, y_pred):
    return 1.0 - SSIM(y_true, y_pred)
  
def SSIM(y_true_and_mask, y_pred):
    y_true, y_mask = tf.split(y_true_and_mask, 2, axis=-1)
    ssim_tf = tf.image.ssim(y_true, y_pred, 1.0)
    return tf.reduce_mean(ssim_tf)

# Decay learning rate
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.3
    epochs_drop = 50 
    lrate = initial_lrate * np.power(drop, 
                                     np.floor((1+epoch)/epochs_drop))
    return lrate
  
def mse_with_mask_tf(y_true_and_mask, y_pred):
    y_true, y_mask = tf.split(y_true_and_mask, 2, axis=-1)
    square_error = (y_true - y_pred)**2
    tf_mask = tf.where(tf.equal(y_mask, 0),
                       tf.fill(tf.shape(y_mask), 0),
                       tf.fill(tf.shape(y_mask), 1))
    tf_mask = tf.to_float(tf_mask)
    return tf.divide(tf.reduce_sum(tf.multiply(tf_mask, square_error)),
                     tf.maximum(tf.reduce_sum(tf_mask), 1.0))

def mse_with_mask_tf_1(y_mask, y_pred):
    tf_mask = tf.where(tf.equal(y_mask, 1),
                       tf.fill(tf.shape(y_mask), 1),
                       tf.fill(tf.shape(y_mask), 0))
    tf_mask = tf.to_float(tf_mask)
    return tf.reduce_mean((tf_mask - y_pred)**2)


def mse_with_mask(groundtruth, predict, mask=None, mask_cloud=0):
    square_error = ((groundtruth - predict)**2)
    cloud_mask = np.where(mask == mask_cloud, 0.0, 1.0) #0
    return np.sum(np.multiply(cloud_mask, square_error))/np.maximum(
        np.sum(cloud_mask), 1.0)

def mse_with_mask_batch(groundtruth, predict, mask=None, mask_cloud=0):
    loss = []
    for i in range(predict.shape[0]):
        loss.append(mse_with_mask(groundtruth[i], predict[i], mask, mask_cloud))
    return [loss[0], np.mean(np.asarray(loss, dtype=np.float32))]


def sum_loss_tf(y_true_and_mask, y_pred):
    y_true, y_mask = tf.split(y_true_and_mask, 2, axis=-1)
    y_true = tf.to_float(y_true)
    y_pred = tf.to_float(y_pred)
    sum_true = tf.reduce_sum(y_true, axis=[1,2,3])
    sum_pred = tf.reduce_sum(y_pred, axis=[1,2,3])
    return tf.reduce_mean(tf.sqrt(tf.square(sum_true - sum_pred)))
