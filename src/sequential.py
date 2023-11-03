from src.layer.base import BaseLayer
import numpy as np
import json
from src.layer.convolutional import Convolutional
from src.layer.dense import Dense

from src.layer.detector import Detector
from src.layer.flatten import Flatten
from src.layer.lstm import LSTM
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
            if (idx + 1) % 500 == 0:
                print(f"Finished processing data {idx + 1}")
            final_result.append(result)

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
                layer_obj = Convolutional.load_from_file(layer)
            elif layer["type"] == "dtctr":
                layer_obj = Detector.load_from_file(layer)
            elif layer["type"] == "max_pool" or layer["type"] == "avg_pool":
                layer_obj = Pooling.load_from_file(layer)
            elif layer["type"] == "flatten":
                layer_obj = Flatten()
            elif layer["type"] == "dense":
                layer_obj = Dense.load_from_file(layer)
            elif layer["type"] == "lstm":
                layer_obj = LSTM.load_from_file(layer)

            layers_from_file.append(layer_obj)

        self.layers = layers_from_file
