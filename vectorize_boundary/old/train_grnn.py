#!/usr/bin/env python
# coding: utf-8

# In[1]:


colab = False
if colab:
    from google.colab import drive
    drive.mount('gdrive')
    gdrive_dir = 'cache'


# In[2]:


import os
import h5py
import numpy as np
from scipy import misc
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[3]:


from modis_utils.misc import cache_data, restore_data


# In[4]:


data = restore_data(os.path.join('cache', 'boundary_vectors_ALL.h5'))


# In[5]:


train_boundary_vectors = data[0]
val_boundary_vectors = data[1]
test_boundary_vectors = data[2]


# In[6]:


n_points = train_boundary_vectors.shape[1]
print(n_points)


# In[7]:


train_boundary_vectors.shape, val_boundary_vectors.shape, test_boundary_vectors.shape


# In[63]:


def transform(data, scaler):
    old_shape = data.shape
    data = data.reshape(old_shape[0], -1)
    if scaler is not None:
        data = scaler.transform(data.astype(np.float))
    #return data.reshape(old_shape)
    return data

def transform_standardize(data, mean, std):
    old_shape = data.shape
    data = data.reshape(old_shape[1]*old_shape[2], -1)
    data = (data - mean)/std
    return data.reshape(old_shape)
    #return data

def find_mean_std(data):
    old_shape = data.shape
    data = data.reshape(old_shape[1]*old_shape[2], -1)
    mean = np.mean(data, axis=-1).reshape(-1, 1)
    std = np.std(data, axis=-1).reshape(-1, 1)
    std[std == 0] = 1
    #mean = mean.reshape(-1, old_shape[-1])
    #std = std.reshape(-1, old_shape[-1])
    return mean, std


# In[64]:


scaler = None
scale_data = False
if scale_data:
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_boundary_vectors.reshape(train_boundary_vectors.shape[0], -1))

mean, std = find_mean_std(train_boundary_vectors)
train_boundary_vectors_1 = transform_standardize(train_boundary_vectors, mean, std)
val_boundary_vectors_1 = transform_standardize(val_boundary_vectors, mean, std)
test_boundary_vectors_1 = transform_standardize(test_boundary_vectors, mean, std)


# In[10]:


# normalize the dataset
'''
train_boundary_vectors_scale_1 = transform(train_boundary_vectors, scaler, flatten=False)
val_boundary_vectors_scale_1 = transform(val_boundary_vectors, scaler, flatten=False)
test_boundary_vectors_scale_1 = transform(test_boundary_vectors, scaler, flatten=False)
'''


# In[12]:


def create_dataset(boundary_vectors_scale, timesteps):
    data_X = []
    data_Y = []
    for i in range(len(boundary_vectors_scale) - timesteps):
        data_x = boundary_vectors_scale[i:(i+timesteps)]
        data_y = boundary_vectors_scale[i + timesteps]
        data_X.append(data_x)
        data_Y.append(data_y)
    return np.asarray(data_X), np.asarray(data_Y)


# In[65]:


timesteps = 50
train_X, train_Y = create_dataset(train_boundary_vectors_1, timesteps)
val_X, val_Y = create_dataset(np.concatenate(
    [train_boundary_vectors_1[-timesteps:], val_boundary_vectors_1]),
                              timesteps)
test_X, test_Y = create_dataset(np.concatenate(
    [val_boundary_vectors_1[-timesteps:], test_boundary_vectors_1]),
                                timesteps)


# In[66]:


print(train_X.shape, train_Y.shape, val_X.shape, val_Y.shape, test_X.shape, test_Y.shape)


# # Visualize data points

# In[67]:


a = train_Y.reshape(-1, 2)
b = val_Y.reshape(-1, 2)
c = test_Y.reshape(-1, 2)

fig, ax = plt.subplots(3, 1, figsize=(10,10))
ax[0].scatter(a[:,0], a[:,1], color='r', label='train')
ax[1].scatter(b[:,0], b[:,1], color='g', label='val')
ax[2].scatter(c[:,0], c[:,1], color='b', label='test')

plt.show()


# In[68]:


a = train_Y.reshape(-1, 2)
b = val_Y.reshape(-1, 2)
c = test_Y.reshape(-1, 2)

fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.scatter(a[:,0], a[:,1], color='r', label='train')
ax.scatter(b[:,0], b[:,1], color='g', label='val')
ax.scatter(c[:,0], c[:,1], color='b', label='test')

ax.legend()

plt.show()


# In[76]:


def split_data(data, axis=0, n=4):
    data_split = []
    n_data_in_one_part = data.shape[axis]//n
    out_shape = list(data.shape)
    out_shape[axis] = n_data_in_one_part
    out_shape[0] = out_shape[0]*n
    data = data.reshape(-1, data.shape[axis])
    for i in range(n):
        data_split.append(data[:, n_data_in_one_part*i : n_data_in_one_part*(i+1)])
    data_split = np.vstack(data_split)
    data_split = data_split.reshape(out_shape)
    return data_split


# In[77]:


n_split = 4
a = split_data(train_X, axis=2, n=n_split)
a.shape


# In[78]:


train_X = split_data(train_X, axis=2, n=n_split)
train_Y = split_data(train_Y, axis=1, n=n_split)

val_X = split_data(val_X, axis=2, n=n_split)
val_Y = split_data(val_Y, axis=1, n=n_split)

test_X = split_data(test_X, axis=2, n=n_split)
test_Y = split_data(test_Y, axis=1, n=n_split)

print(train_X.shape, train_Y.shape, val_X.shape, val_Y.shape, test_X.shape, test_Y.shape)


# In[79]:


def create_graph_matrix(n_points_on_boundary):
    def calc_arc_distance(a, b, n):
        diff = np.abs(a-b)
        if diff > n//2:
            diff = n - diff
        return diff

    n = n_points_on_boundary
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i,j] = calc_arc_distance(i, j, n)
    return mat.astype(np.float32)

def create_graph_matrix_1(n_points_on_boundary):
    def calc_arc_distance(a, b, n):
        diff = np.abs(a-b)
        if diff > n//2:
            diff = n - diff
        return diff

    n = n_points_on_boundary
    mat = np.zeros((2*n, 2*n))
    for i in range(n):
        for j in range(n):
            mat[i,j] = calc_arc_distance(i, j, n)
    mat[n:2*n, n:2*n] = mat[:n, :n]
    for i in range(n):
        for j in range(n, 2*n):
            mat[i,j] = mat[i, j - n]
    mat[n:2*n, :n] = mat[:n, n:2*n]
    return mat.astype(np.float32)


# In[88]:


mat = create_graph_matrix(n_points)
print(mat.shape)


# In[89]:


A = np.divide(mat, n_points)


# In[82]:


A.shape


# In[96]:


from grnn_keras import GRNN
from keras import backend as K
from keras.models import Model, Input
from keras.optimizers import adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, CSVLogger
from keras.utils import plot_model, multi_gpu_model


# In[97]:

learning_rate = 0.001

def create_model(timesteps, n_nodes, n_dims, n_hiddens, dropout=0., recurrent_dropout=0.):
    K.clear_session()
    input_main = Input(shape=(timesteps, n_nodes, n_dims))
    input_aux = Input(shape=(n_nodes, n_nodes), name='A')
    inputs = [input_main, input_aux]

    x = GRNN(n_nodes, n_dims, n_hiddens, keep_dims=False, return_sequences=True,
             dropout=dropout, recurrent_dropout=recurrent_dropout)(inputs)
    x = GRNN(n_nodes, n_hiddens, n_dims, keep_dims=False,
             dropout=dropout, recurrent_dropout=recurrent_dropout)([x, input_aux])
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='mse', optimizer=adam(lr=learning_rate))
    return model


# In[ ]:


start_time = time()

batch_size = 64
timesteps = train_X.shape[1]
n_nodes = n_points // n_split
ndims = train_X.shape[-1]
n_hidden = 2*ndims
dropout = 0.5
recurrent_dropout = 0.3

model = create_model(timesteps, n_nodes, ndims, n_hidden, dropout, recurrent_dropout)
end_time = time()
print('Time to create model = {}s'.format(end_time - start_time))


# In[22]:


model.summary()


# In[90]:


A = np.expand_dims(A, axis=0)
A = np.tile(A, [train_X.shape[0]//n_split, 1, 1])
A_val = A[:val_X.shape[0]//n_split]
print(A.shape, A_val.shape)


# In[91]:


def split_A(A, n_split):
    list_A = []
    n_data_in_one_part = A.shape[1]//n_split
    for i in range(n_split):
        list_A.append(A[:, n_data_in_one_part*i : n_data_in_one_part*(i + 1),
                        n_data_in_one_part*i : n_data_in_one_part*(i + 1)])
    return np.vstack(list_A)


# In[92]:


A = split_A(A, n_split)
A_val = split_A(A_val, n_split)
print(A.shape, A_val.shape)


# In[24]:


weights_path = 'boundary_predict_GRNN_Keras_256.h5'
if os.path.isfile(weights_path):
    print('load last weights')
    model.load_weights(weights_path)


# In[25]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001, verbose=1)


# In[26]:


#parallel_model = multi_gpu_model(model, gpus=2)
#parallel_model.compile(loss='mse', optimizer=adam(lr=0.001))


# In[27]:


weights_dir = 'grnn'
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
filepath = os.path.join(weights_dir, "weights-{epoch:03d}.h5")
checkpoint = ModelCheckpoint(
    filepath, mode='min', verbose=1,
    period=5, save_weights_only=True)
csv_logger = CSVLogger(os.path.join(
    weights_dir, 'log.csv'), append=True, separator=';')

callbacks_list = [checkpoint, csv_logger, reduce_lr]


# In[28]:


timestamps = datetime.now()
timestamps = str(timestamps)
timestamps = timestamps[:timestamps.find('.')]
timestamps = timestamps.replace(' ', '_')
tensorboard_logdir = 'logs/{}'.format(timestamps)
tensorboard = TensorBoard(log_dir=tensorboard_logdir)
callbacks_list.append(tensorboard)


# In[ ]:


if os.path.isfile(weights_path):
    print('load old weights')
    model.load_weights(weights_path)
history = model.fit(
    [train_X, A], train_Y, epochs=10, batch_size=64,
    validation_data=([val_X, A_val], val_Y), callbacks=callbacks_list)
model.save_weights(weights_path, overwrite=True)


# In[30]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


#parallel_model.save('boundary_predict_GRNN_Keras.h5', overwrite=True)

