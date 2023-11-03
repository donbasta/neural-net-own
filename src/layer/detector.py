import numpy as np
from src.activations import relu, sigmoid
from src.layer.base import BaseLayer

DETECTOR_MODES = ["relu", "sigmoid"]


class Detector(BaseLayer):
    def __init__(self, activation):
        self.activation = activation
        self.type = "dtctr"

    @classmethod
    def load_from_file(cls, data):
        activation = data["params"]["activation"]
        layer = cls(activation)
        return layer

    def run(self, inputs: np.array):
        if self.activation == "relu":
            activation_func = relu
        elif self.activation == "sigmoid":
            activation_func = sigmoid
        res = np.vectorize(activation_func, otypes=[float])
        return res(inputs)

    def to_object(self):
        return {"type": self.type, "params": {"activation": self.activation}}
