from src.activations import sigmoid, tanh
from src.layer.base import BaseLayer
import numpy as np


class LSTM(BaseLayer):
    """
    Defines an LSTM layer consisting of inputs and kernels.
    """

    def __init__(
        self,
        input_shape,
        hidden_cell_dim,
        return_sequences=False
    ):
        self.input_shape = input_shape
        self.input_dim = input_shape[0]
        self.timestep = input_shape[1]
        self.hidden_cell_dim = hidden_cell_dim

        self.U_forget = np.random.random((self.input_dim, hidden_cell_dim))
        self.W_forget = np.random.random((hidden_cell_dim, hidden_cell_dim))
        self.b_forget = np.random.random((hidden_cell_dim, 1))

        self.U_input = np.random.random((self.input_dim, hidden_cell_dim))
        self.W_input = np.random.random((hidden_cell_dim, hidden_cell_dim))
        self.b_input = np.random.random((hidden_cell_dim, 1))

        self.U_c = np.random.random((self.input_dim, hidden_cell_dim))
        self.W_c = np.random.random((hidden_cell_dim, hidden_cell_dim))
        self.b_c = np.random.random((hidden_cell_dim, 1))

        self.U_output = np.random.random((self.input_dim, hidden_cell_dim))
        self.W_output = np.random.random((hidden_cell_dim, hidden_cell_dim))
        self.b_output = np.random.random((hidden_cell_dim, 1))

        self.return_sequences = return_sequences

    def run_convolution_stage(self, inputs: np.array):

        outputs = []
        cell_state = np.zeros((self.hidden_cell_dim, 1))
        hidden_state = np.zeros((self.hidden_cell_dim, 1))

        for t, x_t in enumerate(inputs):
            x_t = np.expand_dims(x_t, axis=1)
            f_t = sigmoid(np.matmul(self.U_forget.T, x_t) +
                          np.matmul(self.W_forget.T, hidden_state)
                          + self.b_forget)
            cell_state = np.multiply(cell_state, f_t)
            i_t = sigmoid(np.matmul(self.U_input.T, x_t) +
                          np.matmul(self.W_input.T, hidden_state)
                          + self.b_input)
            c_t = tanh(np.matmul(self.U_c.T, x_t) +
                       np.matmul(self.W_c.T, hidden_state)
                       + self.b_c)
            cell_state += np.multiply(i_t, c_t)
            o_t = sigmoid(np.matmul(self.U_output.T, x_t) +
                          np.matmul(self.W_output.T, hidden_state)
                          + self.b_output)
            tanh_cell_state = tanh(cell_state)
            hidden_state = np.multiply(o_t, tanh_cell_state)

            if t == self.timestep - 1:
                outputs.append(hidden_state.flatten())
            else:
                if self.return_sequences:
                    outputs.append(hidden_state.flatten())

        if not self.return_sequences:
            outputs = outputs[0]

        return np.array(outputs)

    def run(self, inputs: np.array):
        return self.run_convolution_stage(inputs)

    def get_type(self):
        return self.type

    def to_object(self):
        return {
            "type": self.get_type(),
            "params": {
                "hidden_dim": self.hidden_cell_dim,
                "forget_gate": {
                    "U": self.U_forget,
                    "W": self.W_forget,
                    "b": self.b_forget,
                },
                "input_gate": {
                    "U": self.U_input,
                    "W": self.W_input,
                    "b": self.b_input,
                },
                "c_gate": {
                    "U": self.U_c,
                    "W": self.W_c,
                    "b": self.b_c,
                },
                "output_gate": {
                    "U": self.U_output,
                    "W": self.W_output,
                    "b": self.b_output,
                },
            },
        }
