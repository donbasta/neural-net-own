import numpy as np
from src.activations import relu, sigmoid, linear
from src.layer.base import BaseLayer

ACTIVATION_MODES = ["relu", "sigmoid", "linear"]


def get_activation_function(activation_type: str):
    if activation_type == "sigmoid":
        return sigmoid
    elif activation_type == "relu":
        return relu
    elif activation_type == "linear":
        return linear


class Dense(BaseLayer):
    def __init__(self, size, input_size, activation="linear"):
        self.type = "dense"
        self.size = size
        self.input_size = input_size
        self.weights = np.random.random((self.input_size + 1, size))
        self.activation = activation

    @classmethod
    def load_from_file(cls, data):
        size = data["params"]["size"]
        input_size = data["params"]["input_size"]
        activation = data["params"]["activation"]
        layer = cls(size, input_size, activation)
        layer.weights = np.array(data["params"]["Wnb"])
        return layer

    def run(self, inputs):
        biased_input = np.insert(inputs, 0, 1)
        biased_input = np.expand_dims(biased_input, axis=1)
        result = np.matmul(self.weights.T, biased_input).flatten()
        return get_activation_function(self.activation)(result)

    def get_type(self):
        return "dense"

    def to_object(self):
        return {
            "type": self.type,
            "params": {
                "size": self.size,
                "input_size": self.input_size,
                "activation": self.activation,
                "Wnb": self.weights.tolist()
            },
        }

    def get_total_params(self):
        return self.weights.shape[0] * self.weights.shape[1]

    def print_info(self):
        return f"{self.type}\t{self.size}\t\t{self.get_total_params()}"
