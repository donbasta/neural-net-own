from src.layer.base import BaseLayer
from numpy.lib.stride_tricks import as_strided
import numpy as np


class Convolutional(BaseLayer):
    """
    Defines a convolutional layer consisting of inputs and kernels.
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
            generate_random_uniform_matrices(
                self.filter_count, self.n_channels, self.kernel_shape
            )
        )
        self.bias = 1
        self.bias_weight = 0
        self.type = "conv2d"

    @classmethod
    def load_from_file(cls, data):
        input_shape = tuple(data["params"]["input_shape"])
        padding = data["params"]["padding"]
        filter_count = data["params"]["filter_count"]
        kernel_shape = tuple(data["params"]["kernel_shape"])
        stride = data["params"]["stride"]
        layer = cls(input_shape, padding, filter_count, kernel_shape, stride)

        layer.filters = np.array(data["params"]["filters"])
        layer.bias = data["params"]["bias"]

        return layer

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
        return self.type

    def to_object(self):
        return {
            "type": self.get_type(),
            "params": {
                "input_shape": self.input_shape,
                "padding": self.padding,
                "filter_count": self.filter_count,
                "kernel_shape": self.kernel_shape,
                "stride": self.stride,
                "filters": self.filters.tolist(),
                "bias": self.bias,
            },
        }

    def get_total_params(self):
        return self.filter_count * \
            (self.kernel_shape[0] * self.kernel_shape[1]
             * self.input_shape[-1])

    def print_info(self):
        w = 1 + (self.input_shape[1] + 2 * self.padding -
                 self.kernel_shape[0]) // self.stride
        h = 1 + (self.input_shape[2] + 2 * self.padding -
                 self.kernel_shape[1]) // self.stride
        output_size = (w, h, self.filter_count)
        return f"{self.type}\t{output_size}\t\t{self.get_total_params()}"


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


def generate_random_uniform_matrices(n_filter, n_channel, size):
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
