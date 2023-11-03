from src.layer.base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        self.type = "flatten"

    def run(self, inputs):
        return inputs.flatten()

    def to_object(self):
        return {
            "type": self.type,
            "params": {},
        }
