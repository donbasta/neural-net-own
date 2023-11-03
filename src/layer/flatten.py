from src.layer.base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, input_shape):
        self.output_shape = 1
        for c in input_shape:
            self.output_shape *= c
        self.type = "flatten"

    def run(self, inputs):
        return inputs.flatten()

    def to_object(self):
        return {
            "type": self.type,
            "params": {},
        }

    def get_total_params(self):
        return 0

    def print_info(self):
        return f"{self.type}\t{self.output_shape}\t{0}"
