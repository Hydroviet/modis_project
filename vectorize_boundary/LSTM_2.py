#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import h5py
import pickle
import subprocess
import numpy as np
from time import time
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler

import keras
import keras.backend as K
from keras.layers import Input, LSTM, BatchNormalization
from keras.models import Model, load_model


# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
sys.path.append('..')


# In[3]:


from modis_utils.misc import cache_data, restore_data


# In[4]:


data = restore_data(os.path.join('cache', 'boundary_vectors_ALL_1.dat'))


# In[5]:


for x in data:
    print(x.shape)


# In[6]:


train_boundary_vectors = data[0]
test_boundary_vectors = data[1]


# In[7]:


train_boundary_vectors.shape, test_boundary_vectors.shape


# In[8]:


n_points = train_boundary_vectors.shape[1]


# In[9]:


n_years = len(train_boundary_vectors)//46
n_years


# In[10]:


data_train = train_boundary_vectors[:0].copy()
data_test = test_boundary_vectors
for i in range(n_years):
    year = 2003 + i
    if year != 2011 and year != 2013:
        data_train = np.vstack([data_train, train_boundary_vectors[i*46 : (i + 1)*46]])
print(data_train.shape)


# In[11]:


data_train_1 = data_train.reshape(data_train.shape[0], -1)
data_train_1.shape


# In[12]:


data_test_1 = data_test.reshape(data_test.shape[0], -1)


# In[ ]:





# In[46]:


variants = []
for i in range(data_train_1.shape[1]):
    var = np.var(data_train_1[:, i])
    variants.append(var)


# In[47]:


variants_1 = variants.copy()


# In[48]:


variants_2 = np.asarray(variants_1)
variants_2


# In[49]:


list_idx = np.where(variants_2 > 1)[0]


# In[50]:


len(list_idx)


# In[51]:


data_train_2 = data_train_1[:, list_idx]
data_train_2.shape


# In[52]:


data_test_2 = data_test_1[:, list_idx]
data_test_2.shape


# In[53]:


scaler = MinMaxScaler()
scaler.fit(data_train_2)


# In[54]:


data_train_3 = scaler.transform(data_train_2)
data_test_3 = scaler.transform(data_test_2)


# In[ ]:





# In[55]:


def create_sequence_data(data_train, data_test, timesteps=12):
    data_all = np.vstack([data_train, data_test])
    len_val = 46
    len_train = len(data_train) - timesteps - len_val
    len_test = len(data_test)
    
    train_X = []
    train_y = []
    val_X = []
    val_y = []
    test_X = []
    test_y = []
    
    for i in range(timesteps, timesteps + len_train):
        X = np.expand_dims(data_all[i - timesteps : i], axis=0)
        y = np.expand_dims(data_all[i], axis=0)
        train_X.append(X)
        train_y.append(y)
        
    for i in range(timesteps + len_train, timesteps + len_train + len_val):
        X = np.expand_dims(data_all[i - timesteps : i], axis=0)
        y = np.expand_dims(data_all[i], axis=0)
        val_X.append(X)
        val_y.append(y)
        
    for i in range(timesteps + len_train + len_val, timesteps + len_train + len_val + len_test):
        X = np.expand_dims(data_all[i - timesteps : i], axis=0)
        y = np.expand_dims(data_all[i], axis=0)
        test_X.append(X)
        test_y.append(y)
        
    return np.vstack(train_X), np.vstack(train_y), np.vstack(val_X), np.vstack(val_y), np.vstack(test_X), np.vstack(test_y)


# In[56]:


train_X, train_y, val_X, val_y, test_X, test_y = create_sequence_data(data_train_3, data_test_3)


# In[57]:


data = {}
data['train_X'] = train_X
data['train_y'] = train_y
data['val_X'] = val_X
data['val_y'] = val_y
data['test_X'] = test_X
data['test_y'] = test_y


# In[58]:


for k, v in data.items():
    print(v.min(), v.max())


# In[ ]:





# In[59]:


def save_data_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_data_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def create_empty_list(n):
    res = []
    for _ in range(n):
        res.append(None)
    return res

def mse(x, y):
    return np.mean((x - y)**2)


# In[60]:


training_fig_dir = 'visualize/lstm_2/training'
if not os.path.exists(training_fig_dir):
    os.makedirs(training_fig_dir)


# In[61]:


class LSTM_2:

    def __init__(self, data, data_train_full, list_idx=None, scaler=None, mode='train'):
        self.model_dir = 'lstm'
        self.mode = mode
        self.model = None
        
        self.data = data
        
        self.units = data['train_X'].shape[2]
        self.timesteps = data['train_X'].shape[1]
        self.full_shape = data_train_full.shape[-1]
        
        if list_idx is None:
            self.list_idx = np.arange(self.full_shape)
        self.list_idx = list_idx

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        if self.mode == 'inference':
            self.model = self.load_model(self.model_dir)
            
        self.list_small_variant_idx = np.setdiff1d(np.arange(self.full_shape), self.list_idx)
        self.default_values = {}
        for i in self.list_small_variant_idx:
            mean_value = np.mean(data_train_full[:, i])
            self.default_values[i] = int(mean_value)
            
        self.scaler = scaler
        self.model_paths = None
        
    def load_model(self, model_dir=None):
        if model_dir is None:
            model_dir=self.model_dir
        self.model_paths = {}
        for i in range(len(self.list_idx)):
            self.model_paths[i] = os.path.join(model_dir, '{}.dat'.format(i))
        return self.model_paths
    
    def _create_model(self, units=None):
        if units is None:
            units = self.units
        input_shape = (self.timesteps, units)
        inputs = Input(input_shape)
        x = LSTM(units*32, return_sequences=True)(inputs)
        x = LSTM(units*16, return_sequences=True)(inputs)
        x = LSTM(units*4, return_sequences=True)(inputs)
        x = LSTM(units)(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer='adam')
        return model
    
    def train(self, epochs=1, batch_size=1):
        for i in range(len(self.list_idx)):
            self._train(i, epochs, batch_size)
        self.load_model(self.model_dir)
        return self.model_paths

    def _train(self, i, epochs, batch_size):
        print('Training history feature {}'.format(i))
        K.clear_session()
        model_path = os.path.join(self.model_dir, '{}.dat'.format(i))
        model = self._create_model(1)
        history = model.fit(self.data['train_X'][:, :, i : i+1], self.data['train_y'][:, i : i+1],
                            epochs=epochs, batch_size=batch_size, verbose=0,
                            validation_data=(self.data['val_X'][:, :, i : i+1], self.data['val_y'][:, i : i+1]))
        plt.figure()
        plt.plot(history.history['loss'], color='r', label='train loss')
        plt.plot(history.history['val_loss'], color='b', label='val loss')
        plt.legend()
        plt.title('Training history feature {}'.format(i))
        plt.savefig(os.path.join(training_fig_dir, '{}.png'.format(i)))
        model.save(model_path)
        return model_path
    
    def _predict(self, input_test, batch_size):
        inferences = []
        for i in range(len(self.list_idx)):
            K.clear_session()
            model = load_model(self.model_paths[i])
            predict = model.predict(input_test[:, :, i : i+1], batch_size=batch_size)
            inferences.append(predict.reshape(-1, 1))
        return np.concatenate(inferences, axis=-1)
    
    def inference(self, input_test, return_full=None, return_original_range=False):
        if self.model is None:
            self.model = self.load_model()
        if len(input_test.shape) == 2:
            input_test = np.expand_dims(input_test, axis=0)
        inferences = self._predict(input_test, batch_size=64)
        if return_full:
            inferences = self.scaler.inverse_transform(inferences)
            outputs = []
            for idx, input_test_i in enumerate(input_test):
                predictions = inferences[idx]
                res = create_empty_list(self.full_shape)
                for i, prediction in zip(self.list_idx, predictions):
                    res[i] = prediction
                for i, default_value in self.default_values.items():
                    res[i] = default_value
                outputs.append(np.asarray(res))
            outputs = np.asarray(outputs)
            return outputs
        if return_original_range:
            inferences = self.scaler.inverse_transform(inferences)
        return inferences

    def eval(self, inputs=None, groundtruths=None, return_original_range=False):
        if inputs is None:
            inputs = self.data['test_X']
        if groundtruths is None:
            groundtruths = self.data['test_y']
        groundtruth_shape = groundtruths.shape[-1]
        return_full = (groundtruth_shape == self.full_shape)
        predictions = self.inference(inputs, return_full, return_original_range)
        if return_original_range:
            groundtruths = self.scaler.inverse_transform(groundtruths)
        
        loss = (groundtruths - predictions)**2
        return loss, predictions


# In[ ]:





# In[62]:


lstm_2 = LSTM_2(data, data_train_1, list_idx, scaler, mode='train')
model_path = lstm_2.train(epochs=30, batch_size=64)


# In[ ]:


lstm_2 = LSTM_2(data, data_train_1, list_idx, scaler, mode='inference')
losses, predictions = lstm_2.eval(return_original_range=True)
print(losses)
print(losses.mean())


# In[ ]:


inference_dir = 'inference'
if not os.path.exists(inference_dir):
    os.makedir(inference_dir)
cache_data(predictions, os.path.join(inference_dir, 'lstm_2.dat'))


# In[ ]:


inputs = lstm_1.data['test_X'][:1]
groundtruths = lstm_1.data['test_y'][:1]
inputs.shape, groundtruths.shape


# In[ ]:


predictions.shape


# In[ ]:


def get_predictions_multi_steps(lstm_1, n_steps=1):
    inputs = lstm_1.data['test_X'][:1]
    outputs = []
    for i in range(n_steps):
        predict = lstm_1.inference(inputs, return_full=False)
        outputs.append(predict)
        inputs = np.concatenate([inputs[:, 1:, :], np.expand_dims(predict, axis=0)], axis=1)
    return np.asarray(outputs)


# In[ ]:


lstm_1 = LSTM_2(data, data_train_1, list_idx, scaler, mode='inference')
predictions_multi_steps = get_predictions_multi_steps(lstm_1, 2)


# In[ ]:


def get_full_shape_predictions_multi_steps(lstm_1, n_steps=1):
    predictions = get_predictions_multi_steps(lstm_1, n_steps)
    predictions = lstm_1.scaler.inverse_transform(predictions.squeeze())
    res = np.zeros((predictions.shape[0], lstm_1.full_shape))
    for i, idx in enumerate(lstm_1.list_idx):
        res[:, idx] = predictions[:, i].astype(np.int)
    for i, default_value in lstm_1.default_values.items():
        res[:, i] = default_value
    return res


# In[ ]:


full_shape_predictions_multi_steps = get_full_shape_predictions_multi_steps(lstm_1, 80)
print(full_shape_predictions_multi_steps.shape)


# In[ ]:


def reshape_to_point(x):
    return x.reshape(x.shape[0], -1, 2)


# In[ ]:


full_shape_predictions_multi_steps = reshape_to_point(full_shape_predictions_multi_steps)
full_shape_predictions_multi_steps.shape


# In[ ]:


def convert_boundaries_to_image(boundary, img_width, img_height):
    img = np.zeros((img_width, img_height))
    for i in range(boundary.shape[0]):
        x = boundary[i][0].astype(np.int32)
        y = boundary[i][1].astype(np.int32)
        img[x, y] = 1
    return img


# In[ ]:


img1 = convert_boundaries_to_image(full_shape_predictions_multi_steps[0], 513, 513)
plt.imshow(img1)


# # Calculate Polygon area

# In[ ]:


full_shape_predictions_multi_steps.shape, data_test.shape


# In[ ]:


def find_border(data_points):
    x = data_points[:, 0]
    y = data_points[:, 1]
    return x.min(), y.min(), x.max(), y.max()

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.a = y1 - y2
        self.b = x2 - x1
        self.c = x1*y2 - x2*y1
    def calc(self, x, y):
        return self.a*x + self.b*y + self.c
    
def convert_boundary_vector_to_polygon(boundary_vector):
    x1, y1, x2, y2 = find_border(boundary_vector)
    line = Line(x1, y1, x2, y2)
    score_point = [line.calc(x, y) for x, y in boundary_vector]
    group_1 = []
    group_2 = []
    for i, p in enumerate(boundary_vector):
        if score_point[i] < 0:
            group_1.append(p)
        else:
            group_2.append(p)
    group_1 = sorted(group_1, key=itemgetter(0,1))
    group_2 = sorted(group_2, key=itemgetter(0,1), reverse=True)
    group = np.vstack([group_1, group_2])
    return Polygon(zip(group[:, 0], group[:, 1]))

region = [(10, 10), (200, 10), (512, 400), (512, 512), (300, 512), (10, 100), (10, 10)]

def check_point_in_region(p, lines, sample_scores):
    for line, sample_score in zip(lines, sample_scores):
        score = line.calc(p[0], p[1])
        if score/sample_score < 0:
            return False
    return True

def get_points_in_region(boundary, region):
    lines = []
    sample_point = (256, 256)
    for i in range(len(region) - 1):
        lines.append(Line(*(region[i] + region[i + 1])))
    res = []
    sample_scores = [line.calc(sample_point[0], sample_point[1]) for line in lines]
    for p in boundary:
        if check_point_in_region(p, lines, sample_scores):
            res.append(p)
    return np.vstack(res)

def calc_area(points, region):
    points = get_points_in_region(points, region)
    p = convert_boundary_vector_to_polygon(points)
    pp = p.buffer(0)
    return pp.area


# In[ ]:


print(calc_area(full_shape_predictions_multi_steps[0], region))
print(calc_area(data_test[0], region))


# In[ ]:


print(calc_area(full_shape_predictions_multi_steps[1], region))
print(calc_area(data_test[1], region))


# In[ ]:


area_lstm = []
for boundary_vector in full_shape_predictions_multi_steps:
    area = calc_area(boundary_vector, region)
    area_lstm.append(area)
    
area_test_truth = []
for boundary_vector in data_test[:80]:
    area = calc_area(boundary_vector, region)
    area_test_truth.append(area)

plt.plot(area_lstm, color='green', label='area_lstm')
plt.plot(area_test_truth, color='r', label='area_test_truth')
plt.legend()


# In[ ]:


K.clear_session()
model_path = os.path.join(lstm_2.model_dir, '{}.dat'.format(idx))
model = lstm_2._create_model(1)
history = model.fit(lstm_2.data['train_X'][:, :, idx : idx+1], lstm_2.data['train_y'][:, idx : idx+1],
                    epochs=epochs, batch_size=batch_size, verbose=0,
                    validation_data=(data['val_X'][:, :, idx : idx+1], lstm_2.data['val_y'][:, idx : idx+1]))

