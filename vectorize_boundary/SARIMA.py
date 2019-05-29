#!/usr/bin/env python
# coding: utf-8

# In[169]:


import os
import sys
import h5py
import pickle
import subprocess
import numpy as np
from time import time
from operator import itemgetter
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[3]:


sys.path.append('..')


# In[4]:


from modis_utils.misc import cache_data, restore_data


# In[5]:


data = restore_data(os.path.join('cache', 'boundary_vectors_ALL_1.dat'))


# In[8]:


for x in data:
    print(x.shape)


# In[9]:


train_boundary_vectors = data[0]
test_boundary_vectors = data[1]


# In[10]:


train_boundary_vectors.shape, test_boundary_vectors.shape


# In[11]:


n_points = train_boundary_vectors.shape[1]


# In[12]:


n_years = len(train_boundary_vectors)//46
n_years


# In[13]:


data_train = train_boundary_vectors[:0].copy()
data_test = test_boundary_vectors
for i in range(n_years):
    year = 2003 + i
    if year != 2011 and year != 2013:
        data_train = np.vstack([data_train, train_boundary_vectors[i*46 : (i + 1)*46]])
print(data_train.shape)


# In[17]:


data_train_1 = data_train.reshape(data_train.shape[0], -1)
data_train_1.shape


# In[36]:


data_test_1 = data_test.reshape(data_test.shape[0], -1)


# In[208]:





# In[214]:


variants = []
for i in range(data_train_1.shape[1]):
    var = np.var(data_train_1[:, i])
    variants.append((var))


# In[215]:


variants_1 = variants.copy()


# In[217]:


variants_2 = np.asarray(variants_1)
variants_2


# In[218]:


len(variants_2)


# In[227]:


list_idx = np.where(variants_2 > 1)[0]


# In[228]:


len(list_idx)


# In[229]:


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


# In[151]:


class VSARIMA:

    def __init__(self, data_train=None, data_test=None, list_idx=None, mode='train'):
        self.data_train = data_train
        self.data_test = data_test
        self.n_data = data_train.shape[-1]
        self.log = 'log.csv'
        self.model_dir = 'sarima'
        self.train_loss = None
        self.mean_train_loss = None

        if list_idx is None:
            self.list_idx = list(range(self.n_data))
        else:
            self.list_idx = list_idx

        self.list_small_variant_idx = np.setdiff1d(np.arange(self.n_data), self.list_idx)
        self.default_values = {}
        for i in self.list_small_variant_idx:
            mean_value = np.mean(data_train[:, i])
            self.default_values[i] = mean_value

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        with open(self.log, 'a') as f:
            f.write('running_time, idx, train_loss, test_loss\n')

        self.data_train_path = 'data_train_df.dat'
        self.data_test_path = 'data_test_df.dat'
        if self.data_train is not None:
            save_data_pickle(self.data_train, self.data_train_path)
        if self.data_test is not None:
            save_data_pickle(self.data_test, self.data_test_path)

        self.mode = mode
        if self.mode == 'inference':
            self.load_model(self.model_dir)


    def load_model(self, model_dir=None):
        if model_dir is None:
            model_dir=self.model_dir
        self.model_paths = []
        for i in range(self.n_data):
            self.model_paths.append(os.path.join(model_dir, '{}.dat'.format(i)))
        return self.model_paths

    def train(self):
        for idx in self.list_idx:
            self._train(self.data_train[:, idx], self.data_test[:, idx], idx)
        self.load_model(self.model_dir)
        return self.model_paths

    def _train(self, data_train_idx, data_test_idx, idx):
        start_time = time()
        mod = sm.tsa.statespace.SARIMAX(data_train_idx,
                                        order=(1, 0, 1),
                                        seasonal_order=(1, 1, 0, 46),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        model_fit = mod.fit()
        train_loss = self.calc_loss_idx(model_fit, data_train_idx, 'train', 150)
        test_loss = self.calc_loss_idx(model_fit, data_test_idx, 'test')

        end_time = time()
        running_time = end_time - start_time
        with open(self.log, 'a') as f:
            f.write('{:11.3f}s,{:5d},{:11.3f},{:10.3f}\n'.format(running_time, idx, train_loss, test_loss))
        print('{:11.3f}s,{:5d},{:11.3f},{:10.3f}'.format(running_time, idx, train_loss, test_loss))

        model_path = os.path.join(self.model_dir, '{}.dat'.format(idx))
        save_data_pickle(model_fit, model_path)
        return model_path

    def calc_loss_idx(self, model_fit, data_test, type='test', start=None):
        steps = data_test.shape[0]
        if type == 'train':
            forecast = model_fit.get_prediction(start=start)
            forecast = forecast.predicted_mean
            data_test = data_test[start:]
        else:
            forecast = model_fit.forecast(steps)
        loss = mse(forecast, data_test)
        return loss

    def inference(self, steps=1):
        assert len(os.listdir(self.model_dir)) >= data_test.shape[0]
        res = create_empty_list(self.n_data)
        for i in self.list_idx:
            model_path = self.model_paths[i]
            model = load_data_pickle(model_path)
            forecast = model.forecast(steps)
            res[i] = np.expand_dims(forecast, axis=-1)
        for i in self.list_small_variant_idx:
            mean_data = self.default_values[i]
            forecast = np.ones((steps, 1))*mean_data
            res[i] = forecast
        #print(shape_1, shape_2)
        return np.concatenate(res, axis=1)

    def eval(self, groundtruth):
        steps = groundtruth.shape[0]
        predictions = self.inference(steps)
        loss = (groundtruth - predictions)**2
        return loss, predictions


# In[129]:


def main():
    vsarima = VSARIMA(data_train_1, data_test_1, list_idx, mode='train')
    model_paths = vsarima.train()


# In[131]:


#main()


# In[132]:


vsarima = VSARIMA(data_train_1, data_test_1, list_idx, mode='train')
model_paths = vsarima.train()


# In[153]:


vsarima_1 = VSARIMA(data_train_1, data_test_1, list_idx, mode='inference')
losses, predictions = vsarima_1.eval(data_test_1)


# In[155]:


print(losses.mean())


# In[154]:


inference_dir = 'inference'
if not os.path.exists(inference_dir):
    os.makedirs(inference_dir)
cache_data(predictions, os.path.join(inference_dir, 'sarima.dat'))


# # Calculate Polygon area

# In[166]:


predictions = predictions.reshape(predictions.shape[0], -1, 2)


# In[167]:


predictions.shape, data_test.shape


# In[171]:


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


# In[172]:


area_sarima = []
for boundary_vector in predictions:
    area = calc_area(boundary_vector, region)
    area_sarima.append(area)

area_test_truth = []
for boundary_vector in data_test[:80]:
    area = calc_area(boundary_vector, region)
    area_test_truth.append(area)


visualize_dir = 'visualize/sarima'
if not os.path.exists(visualize_dir):
    os.makedirs(visualize_dir)


plt.figure()
plt.plot(area_sarima, color='blue', label='area_sarima')
plt.plot(area_test_truth, color='r', label='area_test_truth')
plt.legend()
plt.savefig(os.path.join(visualize_dir, 'area_predict_groundtruth.png'))


# In[174]:


prediction_1 = predictions[:, 0, 1]
groundtruth_1 = data_test[:, 0, 1]
prediction_1.shape, groundtruth_1.shape


# In[181]:


predictions_2 = predictions.reshape(predictions.shape[0], -1)
groundtruths_2 = data_test.reshape(data_test.shape[0], -1)


# In[185]:



# In[192]:


def visualize_single_point_groundtruth_predict(idx=1, save_file=False):
    prediction_1 = predictions_2[:, idx]
    groundtruth_1 = groundtruths_2[:, idx]
    plt.figure()
    plt.plot(prediction_1, color='blue', label='prediction')
    plt.plot(groundtruth_1, color='r', label='groundtruth')
    plt.legend()
    plt.title('SARIMA coordinate {}'.format(idx))
    if save_file:
        plt.savefig(os.path.join(visualize_dir ,'{}.png'.format(idx)))


# In[194]:


#for idx in list_idx:
#    visualize_single_point_groundtruth_predict(idx, True)


# In[196]:


cache_data(data_test, os.path.join(inference_dir, 'groundtruth.dat'))


# In[ ]:




