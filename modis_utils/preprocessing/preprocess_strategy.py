import os
import numpy as np
from modis_utils.image_processing import change_fill_value
from modis_utils.misc import restore_data, cache_data, get_max_value


class PreprocessStrategy:
    def __init__(self):
        self.fn = None

    def _preprocess_data(self, data_dir, used_band, year_range, n_data_per_year,
                         day_period, preprocessed_data_dir, resize_input=None):
        for year in range(year_range[0], year_range[1] + 1):
            for d in range(n_data_per_year):
                day = d*day_period + 1
                prefix = os.path.join(str(year), str(year) + str(day).zfill(3))
                current_data_dir = os.path.join(data_dir, prefix)
                try:
                    list_imgs = os.listdir(current_data_dir)
                    filename = list(filter(lambda x: used_band in x, list_imgs))[0]
                    img = restore_data(os.path.join(current_data_dir, filename))
                    normalized_img = self.fn(img)
                    if resize_input:
                        normalized_img = normalized_img[:resize_input, :resize_input]
                    cur_dest_dir = os.path.join(preprocessed_data_dir, prefix)
                    if not os.path.exists(cur_dest_dir):
                        os.makedirs(cur_dest_dir)
                    cache_data(normalized_img, os.path.join(cur_dest_dir, 
                                                            'preprocessed.dat'))
                except:
                    print('Not found data {}{:03} in {}.'.format(
                        year, day, current_data_dir))

    def inverse(self, data):
        pass


class NormalizedDivStrategy(PreprocessStrategy):
    def __init__(self):
        self.fn = lambda x: x/10000
    
    def inverse(self, data):
        return data*10000

    def preprocess_data(self, modis_utils_obj):
        change_fill_value_data_dir = os.path.join(
            modis_utils_obj._preprocessed_data_dir_prefix, 'change_fill_value')
        change_fill_value(
            modis_utils_obj._raw_data_dir, modis_utils_obj._used_band,
            modis_utils_obj._year_range, modis_utils_obj._n_data_per_year,
            modis_utils_obj._day_period, change_fill_value_data_dir)
        super()._preprocess_data(
            change_fill_value_data_dir, '', modis_utils_obj._year_range,
            modis_utils_obj._n_data_per_year, modis_utils_obj._day_period,
            modis_utils_obj._preprocessed_data_dir, modis_utils_obj._resize_input)


class NormalizedStrategy(PreprocessStrategy):
    def __init__(self, modis_utils_obj):
        self.MIN = -2001
        self.MAX = get_max_value(
            modis_utils_obj._raw_data_dir, modis_utils_obj._used_band,
            modis_utils_obj._list_years_train, modis_utils_obj._n_data_per_year,
            modis_utils_obj._day_period)
        self.DIFF = self.MAX - self.MIN
        self.fn = lambda x: (x - self.MIN)/self.DIFF

    def inverse(self, data):
        return data*self.DIFF + self.MIN

    def preprocess_data(self, modis_utils_obj):
        change_fill_value_data_dir = os.path.join(
            modis_utils_obj._preprocessed_data_dir_prefix, 'change_fill_value')
        change_fill_value(
            modis_utils_obj._raw_data_dir, modis_utils_obj._used_band, 
            modis_utils_obj._year_range, modis_utils_obj._n_data_per_year,
            modis_utils_obj._day_period, change_fill_value_data_dir)
        super()._preprocess_data(
            change_fill_value_data_dir, '', modis_utils_obj._year_range,
            modis_utils_obj._n_data_per_year, modis_utils_obj._day_period,
            modis_utils_obj._preprocessed_data_dir, modis_utils_obj._resize_input)


class NotPreprocessStrategy(PreprocessStrategy):
    def __init__(self):
        self.fn = lambda x: x
    
    def inverse(self, data):
        return data

    def preprocess_data(self, modis_utils_obj):
        super()._preprocess_data(
            modis_utils_obj._raw_data_dir, '', modis_utils_obj._year_range,
            modis_utils_obj._n_data_per_year, modis_utils_obj._day_period,
            modis_utils_obj._preprocessed_data_dir, modis_utils_obj._resize_input)