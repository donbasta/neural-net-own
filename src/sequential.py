from src.layer.base import BaseLayer
import numpy as np
import json
from src.layer.convolutional import Convolutional
from src.layer.dense import Dense

from src.layer.detector import Detector
from src.layer.flatten import Flatten
from src.layer.pooling import Pooling


class Sequential:
    def __init__(self, layers=None):
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def add(self, layer: BaseLayer):
        self.layers.append(layer)

    def run(self, inputs):
        final_result = []

        for idx, i in enumerate(inputs):
            result = i
            for layer in self.layers:
                result = layer.run(result)
            final_result.append(result)
            print(f"Finished processing data {idx}")

        return np.array(final_result)

    def save_model(self, filepath):
        params = []
        for layer in self.layers:
            params.append(layer.to_object())
        with open(filepath, "w") as f:
            data = json.dumps(params)
            f.write(data)

    def backward(self, din, learning_rate):
        pass

    def load_model(self, filepath):
        layers_from_file = []
        with open(filepath, "r") as f:
            data = json.load(f)
        for layer in data:
            layer_obj = None
            if layer["type"] == "conv2d":
                layer_obj = Convolutional(
                    input_shape=(1, 28, 28),
                    padding=0,
                    filter_count=2,
                    kernel_shape=(2, 2),
                    stride=1,
                )
            elif layer["type"] == "dtctr":
                layer_obj = Detector(activation="relu")
            elif layer["type"] == "max_pool":
                layer_obj = Pooling(size=(2, 2), stride=1, mode="max")
            elif layer["type"] == "avg_pool":
                layer_obj = Pooling(size=(2, 2), stride=1, mode="max")
            elif layer["type"] == "flatten":
                layer_obj = Flatten()
            elif layer["type"] == "dense":
                layer_obj = Dense(size=10, input_size=10, activation="softmax")
            elif layer["type"] == "lstm":
                pass

            layers_from_file.append(layer_obj)

        self.layers = layers_from_file
