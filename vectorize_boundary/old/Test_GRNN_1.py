
# coding: utf-8

# In[1]:


colab = False
if colab:
    from google.colab import drive
    drive.mount('gdrive')
    gdrive_dir = 'gdrive/My Drive/Colab'
else:
    gdrive_dir = '.'


# In[2]:


import os
import h5py
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[3]:


#!pip install livelossplot


# In[4]:


#!git clone https://github.com/lamductan/modis_utils


# In[5]:


from livelossplot import PlotLosses
from modis_utils.misc import cache_data, restore_data


# In[6]:


data = restore_data(os.path.join('cache', 'boundary_vectors_ALL.h5'))


# In[7]:


train_boundary_vectors = data[0]
val_boundary_vectors = data[1]
test_boundary_vectors = data[2]


# In[8]:


n_points = train_boundary_vectors.shape[1]
n_points


# In[9]:


train_boundary_vectors.shape, val_boundary_vectors.shape, test_boundary_vectors.shape


# In[10]:


def transform(data, scaler, flatten=True):
    old_shape = data.shape
    data = data.reshape(old_shape[0], -1)
    data = scaler.transform(data.astype(np.float))
    if not flatten:
        return data.reshape(old_shape)
    return data


# In[11]:


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_boundary_vectors.reshape(train_boundary_vectors.shape[0], -1))
train_boundary_vectors_scale = transform(train_boundary_vectors, scaler, flatten=True)
val_boundary_vectors_scale = transform(val_boundary_vectors, scaler, flatten=True)
test_boundary_vectors_scale = transform(test_boundary_vectors, scaler, flatten=True)


# In[12]:


# normalize the dataset
train_boundary_vectors_scale_1 = transform(train_boundary_vectors, scaler, flatten=False)
val_boundary_vectors_scale_1 = transform(val_boundary_vectors, scaler, flatten=False)
test_boundary_vectors_scale_1 = transform(test_boundary_vectors, scaler, flatten=False)


# In[13]:


train_boundary_vectors_scale_1.shape


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


# In[15]:


'''
timesteps = 50
train_X, train_Y = create_dataset(train_boundary_vectors_scale, timesteps)
val_X, val_Y = create_dataset(np.concatenate(
    [train_boundary_vectors_scale[-timesteps:], val_boundary_vectors_scale]),
                              timesteps)
test_X, test_Y = create_dataset(np.concatenate(
    [val_boundary_vectors_scale[-timesteps:], test_boundary_vectors_scale]),
                                timesteps)
'''


# In[16]:


'''
timesteps = 50
train_X_1, train_Y_1 = create_dataset(train_boundary_vectors_scale_1, timesteps)
val_X_1, val_Y_1 = create_dataset(np.concatenate(
    [train_boundary_vectors_scale_1[-timesteps:], val_boundary_vectors_scale_1]),
                              timesteps)
test_X_1, test_Y_1 = create_dataset(np.concatenate(
    [val_boundary_vectors_scale_1[-timesteps:], test_boundary_vectors_scale_1]),
                                timesteps)
'''


# In[17]:


#train_X.shape, train_Y.shape, val_X.shape, val_Y.shape, test_X.shape, test_Y.shape


# In[18]:


#train_X_1.shape, train_Y_1.shape, val_X_1.shape, val_Y_1.shape, test_X_1.shape, test_Y_1.shape


# In[19]:


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


# In[20]:


mat = create_graph_matrix(n_points)
mat.shape


# In[21]:


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)


# In[22]:


import random
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import sys
from grnn.model import GRNN


# In[23]:


opt = DotDict()
opt.nNode = n_points
opt.batchSize = 1
opt.dimHidden = 32
opt.dimFeature = 2
opt.truncate = 50
opt.nIter = 20
opt.cuda = False
opt.lr = 0.01


# In[24]:


def to_torch(data):
    data_np = np.expand_dims(train_boundary_vectors_scale_1, axis=0)
    return torch.from_numpy(data_np) 


# In[25]:


mat = np.divide(mat, n_points)


# In[26]:


#data = torch.from_numpy(data_np)
#A = torch.from_numpy(mat[np.newaxis, :, :]).double()


# In[28]:


criterion = nn.MSELoss()


# In[29]:


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)


# In[30]:


if opt.cuda:
    net = net.to(device)
    criterion = criterion.cuda()
    data = data.to(device)
    A = A.to(device)


# In[31]:


optimizer = optim.Adam(net.parameters(), lr=opt.lr)


# In[32]:


#hState = torch.randn(opt.batchSize, opt.dimHidden, opt.nNode).double()
yLastPred = 0


# In[33]:


def getTime(begin, end):
    timeDelta = end - begin
    return '%d h %d m %d.%ds' % (timeDelta.seconds // 3600, (timeDelta.seconds%3600) // 60, timeDelta.seconds % 60, timeDelta.microseconds)

timStart = datetime.datetime.now()


# In[34]:


point_plot_idx = 0
feature_plot_idx = 1


# In[35]:


def to_numpy(data):
    if not opt.cuda:
        data = data.cpu()
    return data.numpy()


# In[36]:


'''
for t in range(data.size(1) - opt.truncate):
    x = data[:, t:(t + opt.truncate), :, :]
    y = data[:, (t + 1):(t + opt.truncate + 1), :, :]
    prediction = 0

    for i in range(opt.nIter):
        process = '[Log] %d propogation, %d epoch. ' % (t + 1, i + 1)
        timStamp = datetime.datetime.now()
        prediction, hNew = net(x, hState, A)
        #print(prediction)
        print(process + 'Forward used: %s.' % getTime(timStamp, datetime.datetime.now()))
        hState = hState.data

        loss = criterion(prediction, y)
        optimizer.zero_grad()
        timStamp = datetime.datetime.now()
        loss.backward()
        
        print(process + 'Backward used: %s.' % getTime(timStamp, datetime.datetime.now()))
        
        optimizer.step()

    _, hState = net.propogator(x[:, 0, :, :], hState, A)
    hState = hState.data

    if t == 0:
        plt.plot([v for v in range(opt.truncate)], 
                 x[:, :, point_plot_idx, feature_plot_idx].data.numpy().flatten(), 'r-')
        plt.plot([v + 1 for v in range(opt.truncate)], 
                 to_numpy(prediction[:, :, point_plot_idx, feature_plot_idx].data).flatten(), 'b-')
    else:
        plt.plot([t + opt.truncate - 2, t + opt.truncate - 1], 
                 to_numpy(x[:, -2:, point_plot_idx, feature_plot_idx].data).flatten(), 'r-')
        plt.plot([t + opt.truncate - 1, t + opt.truncate],
                 [yLastPred, prediction[0, -1, point_plot_idx, feature_plot_idx]], 'b-')
        plt.plot([t + opt.truncate - 1, t + opt.truncate], 
                 to_numpy(x[:, -2:, point_plot_idx, feature_plot_idx].data).flatten(), 'r-')
    plt.draw()
    plt.pause(0.5)
    yLastPred = prediction[0, -1, point_plot_idx, feature_plot_idx]

plt.ioff()
plt.show()
'''


# In[37]:


'''
n_data = data.size(1) - opt.truncate
losses = []
visualize_data = False

for i in range(opt.nIter):
    running_loss = 0.0
    for t in range(n_data):
        x = data[:, t:(t + opt.truncate), :, :]
        y = data[:, (t + 1):(t + opt.truncate + 1), :, :]
        prediction = 0

        timStamp = datetime.datetime.now()
        prediction, hNew = net(x, hState, A)
        #print(prediction)
        hState = hState.data

        loss = criterion(prediction, y)
        optimizer.zero_grad()
        timStamp = datetime.datetime.now()
        loss.backward()
        
        running_loss += loss.item()
        optimizer.step()

    running_loss /= n_data
    print('Epoch %d, mean loss = %.3f' % (i + 1, running_loss))
    losses.append(running_loss)
    plt.plot(losses)

    if visualize_data:
        _, hState = net.propogator(x[:, 0, :, :], hState, A)
        hState = hState.data

        if t == 0:
            plt.plot([v for v in range(opt.truncate)], 
                     x[:, :, point_plot_idx, feature_plot_idx].data.numpy().flatten(), 'r-')
            plt.plot([v + 1 for v in range(opt.truncate)], 
                     to_numpy(prediction[:, :, point_plot_idx, feature_plot_idx].data).flatten(), 'b-')
        else:
            plt.plot([t + opt.truncate - 2, t + opt.truncate - 1], 
                     to_numpy(x[:, -2:, point_plot_idx, feature_plot_idx].data).flatten(), 'r-')
            plt.plot([t + opt.truncate - 1, t + opt.truncate],
                     [yLastPred, prediction[0, -1, point_plot_idx, feature_plot_idx]], 'b-')
            plt.plot([t + opt.truncate - 1, t + opt.truncate], 
                     to_numpy(x[:, -2:, point_plot_idx, feature_plot_idx].data).flatten(), 'r-')
        plt.draw()
        plt.pause(0.5)
        yLastPred = prediction[0, -1, point_plot_idx, feature_plot_idx]

plt.ioff()
plt.show()
'''


# In[38]:


num_processes = mp.cpu_count() - 2
num_processes


# In[43]:


data_of_phase = {
    "train": to_torch(train_boundary_vectors_scale_1),
    "validation": to_torch(val_boundary_vectors_scale_1)
}

n_data_of_phase = {}
for phase, data in data_of_phase.items():
    n_data_of_phase[phase] = data.size(1) - opt.truncate

liveloss = PlotLosses()

mp.set_start_method('spawn')
net = GRNN(opt)
#net.double();
net.share_memory()
processes = []


# In[45]:


def train(net, A, opt, data_of_phase, n_data_of_phase):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    hState = torch.randn(opt.batchSize, opt.dimHidden, opt.nNode).double()
    for i in range(opt.nIter):
        logs = {}
        for phase in ['train', 'validation']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            n_data = n_data_of_phase[phase]
            data = data_of_phase[phase]
            for t in range(n_data):
                x = data[:, t:(t + opt.truncate), :, :]
                y = data[:, (t + 1):(t + opt.truncate + 1), :, :]
                prediction = 0

                prediction, hNew = net(x, hState, A)
                hState = hState.data

                loss = criterion(prediction, y)
                optimizer.zero_grad()
                timStamp = datetime.datetime.now()
                loss.backward()

                current_loss = loss.item()
                running_loss += current_loss
                if phase == 'train' and t % 30 == 0:
                    print(current_loss)
                optimizer.step()

            '''
            running_loss /= n_data
            print('Epoch %d, mean loss = %.3f' % (i + 1, running_loss))
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'

            logs[prefix + 'log loss'] = running_loss
            liveloss.update(logs)
            liveloss.draw()
            '''


# In[47]:


for rank in range(num_processes):
    p = mp.Process(target=train, args=(net, A, opt, data_of_phase, n_data_of_phase))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

