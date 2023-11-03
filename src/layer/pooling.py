import numpy as np
from src.layer.base import BaseLayer
from numpy.lib.stride_tricks import as_strided


class Pooling(BaseLayer):
    def __init__(self, size, stride, mode):
        self.size = size
        self.stride = stride
        if mode == "max":
            self.pool_function = np.max
        elif mode == "avg":
            self.pool_function = np.average
        else:
            raise TypeError("pooling mode must be either 'max' or 'avg'.")
        self.type = f"{mode}_pool"
        self.mode = mode

    @classmethod
    def load_from_file(cls, data):
        size = tuple(data["params"]["size"])
        stride = data["params"]["stride"]
        mode = data["params"]["mode"]
        layer = cls(size, stride, mode)
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
                "stride": self.stride
            },
        }


def generate_strides(mat: np.array, kernel_size, stride):
    view_shape = tuple(np.subtract(mat.shape, kernel_size) + 1) + kernel_size
    view_strides = mat.strides + mat.strides
    result = as_strided(mat, strides=view_strides, shape=view_shape)[
        ::stride, ::stride]
    return result
