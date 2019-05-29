# In[1]:


import os
import sys
import h5py
import pickle
import subprocess
import numpy as np
import pandas as pd
from time import time
import statsmodels.api as sm

def save_data_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_data_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def mse(x, y):
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.values
    return np.mean((x - y)**2)


# In[5]:


data_train_path = sys.argv[1]
data_test_path = sys.argv[2]
log = sys.argv[3]
log_dir = sys.argv[4]
i = int(sys.argv[5])
start_time = float(sys.argv[6])

data_train = load_data_pickle(data_train_path)
data_test = load_data_pickle(data_test_path)
data_train = data_train.iloc[:, i]
data_test = data_test.iloc[:, i]

class VSARIMA1:

    def __init__(self, data_train, data_test):
        self.data_train = data_train
        self.data_test = data_test
        self.results = None
        self.log = log
        self.log_dir = log_dir
        self.mean_train_loss = None

    def train(self):
        return self._train(self.data_train, self.data_test)

    def _train(self, data_train, data_test):
        mod = sm.tsa.statespace.SARIMAX(data_train,
                                        order=(1, 0, 1),
                                        seasonal_order=(1, 1, 0, 46),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.results = mod.fit()
        train_loss = self.calc_loss(data_train, 'train', 150)
        test_loss = self.calc_loss(data_test)

        end_time = time()
        running_time = end_time - start_time
        with open(self.log, 'a') as f:
            f.write('{:10.3f}s,{:5d},{:10.3f},{:10.3f}\n'.format(running_time, i, train_loss, test_loss))


        save_data_pickle((mod, self.results, train_loss, test_loss), os.path.join(self.log_dir, '{}.dat'.format(i)))
        return self.results

    def calc_loss(self, data_test, type='test', start=None):
        steps = data_test.shape[0]
        if type == 'train':
            forecast = self.results.get_prediction(start=start)
            forecast = forecast.predicted_mean
            data_test = data_test[start:]
        else:
            forecast = self.results.forecast(steps)
        loss = mse(forecast, data_test)
        return loss

    def inference(self, steps=1):
        return self.results.forecast(steps)

    def eval(self, groundtruth, metric=None):
        steps = groundtruth.shape[0]
        yhat = self.inference(steps)
        if metric is None:
            metric = mse
        return yhat, metric(groundtruth, yhat)


# In[57]:

vsarima1 = VSARIMA1(data_train, data_test)
vsarima1.train()

