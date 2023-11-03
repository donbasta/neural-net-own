import numpy as np


class BaseLayer:
    def __init__(self):
        self.type = "base"
        return

    def run(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def get_type(self):
        return self.type

    def get_X(self):
        return self.X

    def get_W(self):
        return self.W

    def get_shape(self, input_shape=None):
        pass

    def to_object(self):
        pass

    def print_info(self):
        pass

    def get_total_params(self):
        pass
