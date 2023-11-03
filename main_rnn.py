import numpy as np
import pandas as pd

from src import Sequential
from src.layer import Dense


def create_sequences(data, seq_length):
    """
    Transform a time-series data into a multi-columnar tabular data whose values are the previous dates
    Sumber: Spesifikasi tugas besar, with some changes
    """
    sequences = []
    targets = []
    data_len = len(data)
    for i in range(data_len - seq_length):
        seq_end = i + seq_length
        seq_x = data[i:seq_end]
        seq_y = data[seq_end]
        sequences.append(seq_x)
        targets.append(seq_y)
    sequences = np.array(sequences)
    targets = np.array(targets)
    targets = targets.reshape((targets.shape[0], 1))
    return sequences, targets


def get_data():
    train_raw = pd.read_csv("./datasets/Train_stock_market.csv")
    train_open_x, train_open_y = create_sequences(train_raw["Open"], 5)
    train_high_x, train_high_y = create_sequences(train_raw["High"], 5)
    train_volume_x, train_volume_y = create_sequences(train_raw["Volume"], 5)
    train_close_x, train_close_y = create_sequences(train_raw["Close"], 5)
    train_low_x, train_low_y = create_sequences(train_raw["Low"], 5)
    train_x = np.concatenate(
        (train_open_x, train_high_x, train_volume_x, train_close_x, train_low_x), axis=1)
    train_y = np.concatenate(
        (train_open_y, train_high_y, train_volume_y, train_close_y, train_low_y), axis=1)
    return train_x, train_y


def forward_prop_stock_market():
    train_data_x, train_data_y = get_data()

    model = Sequential()
    model.add(Dense(size=10, input_size=25, activation="relu"))
    model.add(Dense(size=5, input_size=10, activation="linear"))

    predictions = model.run(inputs=train_data_x)

    print(predictions)

    model.save_model("model-timeseries-dense.json")


if __name__ == "__main__":
    forward_prop_stock_market()
