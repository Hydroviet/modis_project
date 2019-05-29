from modis_utils.preprocessing.preprocess_strategy import NormalizedStrategy
from modis_utils.preprocessing.preprocess_strategy import NormalizedDivStrategy
from modis_utils.preprocessing.preprocess_strategy import NotPreprocessStrategy


class PreprocessStrategyContext:

    def __init__(self, modis_utils_obj):
        self.strategy = None
        if modis_utils_obj._preprocessed_type == 'normalized_div':
            self.strategy = NormalizedDivStrategy()
        elif modis_utils_obj._preprocessed_type == 'normalized_zero_one':
            self.strategy = NormalizedStrategy(modis_utils_obj)
        elif modis_utils_obj._preprocessed_type == 'not_preprocessed':
            self.strategy = NotPreprocessStrategy()
        elif modis_utils_obj._preprocessed_type == 'Zhang':
            self.strategy = NotPreprocessStrategy()
        else:
            raise ValueError

    def preprocess_data(self, modis_utils_obj):
        self.strategy.preprocess_data(modis_utils_obj)

    def inverse(self, data):
        return self.strategy.inverse(data)
