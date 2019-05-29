import numpy as np
from modis_utils.misc import restore_data

#from keras.utils import Sequence
from tensorflow.python.keras.utils import Sequence


class MyGenerator(Sequence):
    def __init__(self, data_filenames, batch_size, original_batch_size, pretrained):
        assert batch_size <= original_batch_size
        
        self.data_filenames = data_filenames
        self.batch_size = batch_size
        self.original_batch_size = original_batch_size
        self.k = self.original_batch_size // self.batch_size
        self.pretrained = pretrained

    def __len__(self):
        return len(self.data_filenames)*self.k


class OneOutputGenerator(MyGenerator):
      
    def __getitem__(self, idx):         
        data = restore_data(self.data_filenames[idx // self.k])
        i = idx % self.k
        batch_X = data[0][i*self.batch_size:(i+1)*self.batch_size]
        batch_y = data[1][i*self.batch_size:(i+1)*self.batch_size]
        if self.pretrained:
            batch_X = np.tile(batch_X, 3)
        return batch_X.astype(np.float32), batch_y.astype(np.float32)


class MultipleOutputGenerator(MyGenerator):
      
    def __getitem__(self, idx):         
        data = restore_data(self.data_filenames[idx // self.k])
        i = idx % self.k
        batch_X = data[0][i*self.batch_size:(i+1)*self.batch_size].astype(np.float32)
        batch_y = data[1][i*self.batch_size:(i+1)*self.batch_size].astype(np.float32)
        if self.pretrained:
            batch_X = np.tile(batch_X, 3)
        return batch_X, [batch_y, batch_X]