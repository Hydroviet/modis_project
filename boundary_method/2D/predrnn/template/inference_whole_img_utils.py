import os
import numpy as np

from inference import InferencePredRNN
from gen_data_inference_utils import gen_boundary_patch, select_data

class InferencePredRNNWholeImg:

    def __init__(self, params):
        self.params = params
        self.patch_size = params['img_width']
        self.img_width = params['whole_img_width']
        self.inference_predrnn = InferencePredRNN(params)


    def _get_inference_from_np_array_patch(self, inputs_np_patch):
        return self.inference_predrnn.get_inference_from_np_array(inputs_np_patch)


    def _get_inference_from_np_array_whole_img(self, inputs_np_whole_img):
        return inputs_np_whole_img[-1]


    def _merge(self, predicted_whole_img, predicted_patches, coords):
        res = predicted_whole_img
        for coord, patch in zip(coords, predicted_patches):
            x1, x2, y1, y2 = coord
            res[x1 : x2+1, y1 : y2+1] = patch
        return res


    def get_inference_from_np_array(self, full_timesteps_inputs_np, steps=1):
        results_steps = []
        for _ in range(steps):
            # inputs_np_whole_img: shape = (T, H, W)
            inputs_np_whole_img = select_data(full_timesteps_inputs_np)
            # inputs_np_patch: shape = (B, T, P, P)
            inputs_np_patch, coords = gen_boundary_patch(inputs_np_whole_img, self.patch_size)

            predicted_patches = self._get_inference_from_np_array_patch(inputs_np_patch) # (B, T, P, P)
            predicted_whole_img = self._get_inference_from_np_array_whole_img(inputs_np_whole_img) # (H, w)

            final_predicted = self._merge(predicted_whole_img, predicted_patches, coords) # (H, W)
            final_predicted = np.expand_dims(final_predicted, axis=0)
            results_steps.append(final_predicted)
            full_timesteps_inputs_np = np.vstack([full_timesteps_inputs_np[:-1], final_predicted])
        return np.vstack(results_steps) # (S, H, W)

