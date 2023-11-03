import numpy as np
from src.layer.base import BaseLayer
from numpy.lib.stride_tricks import as_strided


class Pooling(BaseLayer):
    def __init__(self, size, stride, mode, input_shape):
        self.size = size
        self.stride = stride
        if mode == "max":
            self.pool_function = np.max
        elif mode == "avg":
            self.pool_function = np.average
        else:
            raise TypeError("pooling mode must be either 'max' or 'avg'.")
        self.type = f"{mode}_pool"
        self.input_shape = input_shape
        self.mode = mode

    @classmethod
    def load_from_file(cls, data):
        size = tuple(data["params"]["size"])
        input_shape = tuple(data["params"]["input_shape"])
        stride = data["params"]["stride"]
        mode = data["params"]["mode"]
        layer = cls(size, stride, mode, input_shape)
        return layer

    def run_pooling(self, inputs):
        res = []
        for i in inputs:
            strided_views = generate_strides(i, self.size, stride=self.stride)
            feature_map = np.array(
                [[self.pool_function(view) for view in row]
                 for row in strided_views]
            )
            res.append(feature_map)
        return np.array(res)

    def run(self, inputs: np.array):
        res = self.run_pooling(inputs)
        return res

    def to_object(self):
        return {
            "type": self.type,
            "params": {
                "mode": self.mode,
                "size": self.size,
                "stride": self.stride,
                "input_shape": self.input_shape
            },
        }

    def get_total_params(self):
        return 0

    def print_info(self):
        w = (self.input_shape[0] - self.size[0]) // self.stride
        h = (self.input_shape[1] - self.size[1]) // self.stride
        c = self.input_shape[2]
        output_shape = (w, h, c)

        return f"{self.type}\t{output_shape}\t{0}"


def generate_strides(mat: np.array, kernel_size, stride):
    view_shape = tuple(np.subtract(mat.shape, kernel_size) + 1) + kernel_size
    view_strides = mat.strides + mat.strides
    result = as_strided(mat, strides=view_strides, shape=view_shape)[
        ::stride, ::stride]
    return result
