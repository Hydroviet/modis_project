
# coding: utf-8

# In[70]:


import os
import sys
import h5py
import pickle
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from o365_utils import O365Account


# In[3]:


from modis_utils.misc import cache_data, restore_data


account = O365Account()


# In[5]:


data = restore_data(os.path.join('cache', 'boundary_vectors_ALL_1.dat'))


# In[6]:


train_boundary_vectors = data[0]
test_boundary_vectors = data[1]


# In[7]:


train_boundary_vectors.shape, test_boundary_vectors.shape


# In[8]:


n_points = train_boundary_vectors.shape[1]


# In[12]:


n_years = len(train_boundary_vectors)//46
n_years


# In[31]:


data_train = train_boundary_vectors[:0].copy()
data_test = test_boundary_vectors
for i in range(n_years):
    year = 2003 + i
    if year != 2011 and year != 2013:
        data_train = np.vstack([data_train, train_boundary_vectors[i*46 : (i + 1)*46]])
print(data_train.shape)


# In[9]:


import statsmodels.api as sm
import itertools


# In[17]:


y = data_train[:, 0, 1]


# In[19]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[24]:


from time import time


# In[25]:


start_time = time()
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 0, 1),
                                seasonal_order=(1, 1, 0, 46),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
end_time = time()
print('Running time = {:0.1f}s'.format(end_time - start_time))
print(results.summary().tables[1])


# In[39]:


data_train = data_train.reshape(data_train.shape[0], -1)
data_train.shape


# In[40]:


n_tests = data_test.shape[0]
data_test = data_test.reshape(data_test.shape[0], -1)
n_tests


# In[41]:


import pandas as pd


# In[54]:


df = pd.DataFrame(data=np.vstack([data_train, data_test]))


# In[55]:


df = df.loc[:, (df != df.iloc[0]).any()]


# In[61]:


df_train = df[:-n_tests]
df_test = df[-n_tests:]
df_train_12 = df_train.iloc[:, :12]
df_test_12 = df_test.iloc[:, :12]


# In[62]:


df_train_12.shape, df_test_12.shape


# In[69]:


def mse(x, y):
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.values
    return np.mean((x - y)**2)

def save_data_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_data_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# In[84]:


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
            self.results[i] = os.path.join(self.log_dir, '{}.dat'.format(i))
            if i % 10 == 0:
                print(i, end=' ')
        end_time = time()
        print("\nTraining time = {:0.3f}s".format(end_time - start_time))
        return self.results

    def calc_loss(self, start_idx, end_idx, data_test):
        res = []
        steps = data_test.shape[0]
        for i in range(start_idx, end_idx + 1):
            results = self.results[i]
            if self.mode == 'inference' or isinstance(results, str):
                _, results, _, _ = load_data_pickle(results)
            forecast = results.forecast(steps)
            loss = mse(forecast, data_test.iloc[:,i])
            res.append(loss)
        return res

    def inference(self, steps=1):
        res = []
        for i in range(self.n_data):
            results = self.results[i]
            if self.mode == 'inference' or isinstance(results, str):
                _, results, _, _ = load_data_pickle(results)
            res.append(results.forecast(steps))
        return pd.concat(res, ignore_index=True, axis=1)

    def eval(self, groundtruth, metric=None):
        steps = groundtruth.shape[0]
        yhat = self.inference(steps)
        if metric is None:
            metric = mse
        return yhat, metric(groundtruth, yhat)


# In[71]:


vsarima = VSARIMA(df_train_12, df_test_12)
#vsarima = VSARIMA(df_train, df_test)


# In[72]:


vsarima.train()


# In[85]:

yhat, loss = vsarima.eval(df_test_12)
#yhat, loss = vsarima.eval(df_test)
print('mse =', loss)


cache_data(yhat, 'vsarima_inference.dat')
account.upload_file('vsarima_inference.dat', 'MODIS')
account.upload_file('log.csv', 'MODIS')
