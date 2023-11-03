import numpy as np
from src.activations import relu, sigmoid, linear
from src.layer.base import BaseLayer

ACTIVATION_MODES = ["relu", "sigmoid", "linear"]


class Dense(BaseLayer):
    def __init__(self, size, input_size, activation="linear"):
        self.type = "dense"
        self.size = size
        self.input_size = input_size
        self.weights = np.random.random((self.input_size + 1, size))
        self.activation = activation

    def run(self, inputs):
        biased_input = np.insert(inputs, 0, 1)
        biased_input = np.expand_dims(biased_input, axis=1)
        result = np.matmul(self.weights.T, biased_input).flatten()
        if self.activation == "sigmoid":
            activation_func = sigmoid
        elif self.activation == "relu":
            activation_func = relu
        elif self.activation == "linear":
            activation_func = linear
        return activation_func(result)

    def get_type(self):
        return "dense"

    def to_object(self):
        return {
            "type": self.type,
            "params": {
                "kernel": self.weights[: self.input_size, :].tolist(),
                "bias": self.weights[self.input_size: -1, :].tolist(),
            },
        }
