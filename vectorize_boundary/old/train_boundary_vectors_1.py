# In[1]:


import os
import sys
import h5py
import pickle
import subprocess
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[2]:


from modis_utils.misc import cache_data, restore_data


# In[4]:


scale_data = False


# In[5]:


data = restore_data(os.path.join('cache', 'boundary_vectors_ALL.h5'))


# In[6]:


train_boundary_vectors = data[0]
val_boundary_vectors = data[1]
test_boundary_vectors = data[2]


# In[7]:


train_boundary_vectors.shape, val_boundary_vectors.shape, test_boundary_vectors.shape


# In[8]:


n_points = train_boundary_vectors.shape[1]


# In[9]:


data_1 = np.concatenate(data, axis=0)


# In[10]:


train_boundary_vectors = data_1[:-92]
val_boundary_vectors = data_1[-92:-46]
test_boundary_vectors = data_1[-46:]


# In[11]:


train_boundary_vectors.shape, val_boundary_vectors.shape, test_boundary_vectors.shape


# In[12]:


def transform(data, scaler):
    old_shape = data.shape
    data = data.reshape(old_shape[0], -1)
    if scaler is not None:
        data = scaler.transform(data.astype(np.float))
    return data.reshape(old_shape)
    #return data

def transform_standardize(data, mean, std):
    old_shape = data.shape
    data = data.reshape(-1, old_shape[1]*old_shape[2])
    data = (data - mean)/std
    return data.reshape(old_shape)
    #return data

def find_mean_std(data):
    old_shape = data.shape
    data = data.reshape(-1, old_shape[1]*old_shape[2])
    mean = np.mean(data, axis=0).reshape(1, -1)
    std = np.std(data, axis=0).reshape(1, -1)
    std[std == 0] = 1
    #mean = mean.reshape(-1, old_shape[-1])
    #std = std.reshape(-1, old_shape[-1])
    return mean, std


# In[13]:


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


# In[14]:


def create_dataset(boundary_vectors_scale, timesteps):
    data_X = []
    data_Y = []
    for i in range(len(boundary_vectors_scale) - timesteps):
        data_x = boundary_vectors_scale[i:(i+timesteps)]
        data_y = boundary_vectors_scale[i + timesteps]
        data_X.append(data_x)
        data_Y.append(data_y)
    return np.asarray(data_X), np.asarray(data_Y)


# In[16]:


from numpy import polyfit


# In[17]:



# In[18]:


n_years = len(data_1)//46


# In[19]:


data_2 = data_1[:24].copy()
data_3 = data_1[24:]
for i in range(n_years):
    year = 2003 + i
    if year != 2011 and year != 2013:
        data_2 = np.vstack([data_2, data_3[i*46 : (i + 1)*46]])

data_1.shape, data_2.shape


# In[22]:


import statsmodels.api as sm
import itertools


# In[23]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[26]:


from time import time


# In[31]:


data = data_2.reshape(data_2.shape[0], -1)
data.shape


# In[39]:


n_tests = 92
data_train = data[:-n_tests]



# In[42]:


import pandas as pd


# In[43]:


data = data_2.reshape(data_2.shape[0], -1)
data.shape


# In[44]:


df = pd.DataFrame(data=data)
df1 = df.loc[:, (df != df.iloc[0]).any()]
print(df1.shape)

data_train = df1[:-n_tests]
data_test = df1[-n_tests:]
print(data_train.shape)
print(data_test.shape)
data_train_12 = data_train.iloc[:, :12]
data_test_12 = data_test.iloc[:, :12]

# In[55]:


def mse(x, y):
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.values
    return np.mean((x - y)**2)


# In[56]:

def save_data_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_data_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class VSARIMA:

    def __init__(self, data_train=None, data_test=None, mode='train'):
        self.data_train = data_train
        self.data_test = data_test
        self.n_data = data_train.shape[-1]
        self.log = 'log.csv'
        self.log_dir = 'sarima'
        self.train_loss = None
        self.mean_train_loss = None

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.data_train_path = 'data_train_df.dat'
        self.data_test_path = 'data_test_df.dat'
        if self.data_train is not None:
            save_data_pickle(self.data_train, self.data_train_path)
        if self.data_test is not None:
            save_data_pickle(self.data_test, self.data_test_path)

        self.mode = mode
        if self.mode == 'inference':
            self.load_model(self.log_dir)
        else:
            self.results = self._create_empty_list(self.n_data)

    def _create_empty_list(self, n):
        res = []
        for _ in range(n):
            res.append(None)
        return res

    def load_model(self, log_dir):
        self.results = []
        for i in range(self.n_data):
            self.results.append(os.path.join(log_dir, '{}.dat'.format(i)))
        return self.results

    def train(self):
        return self._train(self.data_train, self.data_test, [0, self.n_data])

    def _train(self, data_train, data_test, idx):
        start_time = time()
        train_loss = []
        for i in range(idx[0], idx[1]):
            subprocess.call(['python3.6', 'sarima.py', self.data_train_path,
                self.data_test_path, self.log, self.log_dir, str(i), str(start_time)])
            if i % 10 == 0:
                print(i)
        end_time = time()
        print("Training time = {:0.3f}s".format(end_time - start_time))
        return self.results

    def calc_loss(self, start_idx, end_idx, data_test):
        res = []
        steps = data_test.shape[0]
        for i in range(start_idx, end_idx + 1):
            results = self.results[i]
            if self.mode == 'inference':
                _, results, _, _ = load_data_pickle(results)
            forecast = results.forecast(steps)
            loss = mse(forecast, data_test.iloc[:,i])
            res.append(loss)
        return res

    def inference(self, steps=1):
        res = []
        for i in range(self.n_data):
            results = self.results[i]
            if self.mode == 'inference':
                _, results, _, _ = load_data_pickle(results)
            res.append(results.forecast(steps))
        return pd.concat(res, ignore_index=True, axis=1)

    def eval(self, groundtruth, metric=None):
        steps = groundtruth.shape[0]
        yhat = self.inference(steps)
        if metric is None:
            metric = mse
        return yhat, metric(groundtruth, yhat)


# In[57]:

if __name__ == '__main__':
    #vsarima = VSARIMA(data_train_12, data_test_12)
    vsarima = VSARIMA(data_train, data_test)

    # In[58]:


    vsarima.train()

