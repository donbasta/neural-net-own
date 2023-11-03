from src.layer.base import BaseLayer
from numpy.lib.stride_tricks import as_strided
import numpy as np


class LSTM(BaseLayer):
    """
    Defines an LSTM layer consisting of inputs and kernels.
    """

    def __init__(
        self,
        input_shape,
        padding,
        filter_count,
        kernel_shape,
        stride,
    ):
        self.input_shape = input_shape
        self.padding = padding
        self.stride = stride
        self.filter_count = filter_count
        self.kernel_shape = kernel_shape
        self.n_channels = input_shape[0]
        self.filters = np.array(
            generate_random_uniform_matrixes(
                self.filter_count, self.n_channels, self.kernel_shape
            )
        )
        self.bias = 1
        self.bias_weight = 0
        self.type = "convolutional"

    def run_convolution_stage(self, inputs: np.array):
        final_fmap = []
        filter_idx = 0
        for kernels in self.filters:
            fmap = []
            for channel_idx, input_channel in enumerate(inputs):
                padded = pad_array(input_channel, self.padding, 0)
                strided_views = generate_strides(
                    padded, self.kernel_shape, stride=self.stride
                )
                multiplied_views = np.array(
                    [np.multiply(view, kernels[channel_idx])
                     for view in strided_views]
                )
                conv_mult_res = np.array(
                    [[np.sum(view) for view in row]
                     for row in multiplied_views]
                )
                fmap.append(conv_mult_res)
            fmap = np.array(fmap)
            final_fmap.append(add_all_feature_maps(fmap))
            filter_idx += 1
        bias_weight = self.bias * self.bias_weight
        return np.array(final_fmap) + bias_weight

    def run(self, inputs: np.array):
        return self.run_convolution_stage(inputs)

    def get_type(self):
        return "conv2d"

    def to_object(self):
        return {
            "type": self.get_type(),
            "params": {
                "kernel": self.filters.tolist(),
                "bias": self.bias,
            },
        }


def pad_with(vector, pad_width, _, kwargs):
    pad_value = kwargs.get("padder", 10)
    vector[: pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def pad_array(mat: np.array, size: int, padder: int):
    padded = np.pad(mat, size, pad_with, padder=padder)
    return padded


def generate_strides(mat: np.array, kernel_size, stride):
    view_shape = tuple(np.subtract(mat.shape, kernel_size) + 1) + kernel_size
    view_strides = mat.strides + mat.strides
    result = as_strided(mat, strides=view_strides, shape=view_shape)[
        ::stride, ::stride]
    return result


def generate_random_uniform_matrixes(n_filter, n_channel, size):
    np.random.seed(42)
    return np.array(
        [
            [np.random.uniform(low=-1.0, high=1.0, size=size)
             for _ in range(n_channel)]
            for _ in range(n_filter)
        ]
    )


def add_all_feature_maps(feature_map_arr: np.array):
    res = np.zeros((feature_map_arr.shape[1], feature_map_arr.shape[2]))
    for feature_map in feature_map_arr:
        np.add(res, feature_map, out=res)
    return res
