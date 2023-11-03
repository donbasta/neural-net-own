import numpy as np
import pandas as pd

from src import Sequential
from src.layer import Dense, LSTM
from icecream import ic


def create_sequences(data, seq_length):
    """
    Transform a time-series data into a multi-columnar tabular data
    whose values are the previous dates
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
    sequences = np.array(sequences).astype("float")
    targets = np.array(targets).astype("float")
    targets = targets.reshape((targets.shape[0], 1))
    return sequences, targets


def get_data(window_size, num_row=-1):
    train_raw = pd.read_csv("./datasets/Train_stock_market.csv")
    train_open_x, train_open_y = create_sequences(
        train_raw["Open"], window_size)
    train_open_x /= 10.0
    train_open_y /= 10.0

    train_high_x, train_high_y = create_sequences(
        train_raw["High"], window_size)
    train_high_x /= 10.0
    train_high_y /= 10.0

    train_volume_x, train_volume_y = create_sequences(
        train_raw["Volume"], window_size)
    train_volume_x /= 10000.0
    train_volume_y /= 10000.0

    train_close_x, train_close_y = create_sequences(
        train_raw["Close"], window_size)
    train_close_x /= 10.0
    train_close_y /= 10.0

    train_low_x, train_low_y = create_sequences(train_raw["Low"], window_size)
    train_low_x /= 10.0
    train_low_y /= 10.0

    train_x = np.array([train_open_x, train_high_x,
                       train_volume_x, train_close_x, train_low_x])
    train_x = np.moveaxis(train_x, 0, -1)
    train_y = np.concatenate(
        (train_open_y, train_high_y, train_volume_y,
         train_close_y, train_low_y), axis=1)

    if num_row != -1:
        return train_x[:num_row], train_y[:num_row]

    return train_x, train_y


NROW = 3


def forward_prop_stock_market():
    train_data_x, train_data_y = get_data(5, NROW)

    model = Sequential()
    model.add(LSTM(input_shape=(5, 5),
                   hidden_cell_dim=3, return_sequences=False))
    model.add(Dense(size=10, input_size=3, activation="relu"))
    model.add(Dense(size=5, input_size=10, activation="linear"))

    predictions = model.run(inputs=train_data_x)

    ic(predictions)
    ic(predictions.shape)

    model.save_model("model-timeseries-dense.json")


def forward_prop_stock_market_load_model():
    train_data_x, train_data_y = get_data(5, NROW)

    model = Sequential()
    model.load_model("model-timeseries-dense.json")

    print(model.layers)

    predictions = model.run(inputs=train_data_x)
    ic(predictions)
    ic(predictions.shape)


def perform_experiment(window_size=5):
    train_data_x, _ = get_data(window_size)
    print(train_data_x.shape)

    model = Sequential()
    model.add(LSTM(input_shape=(window_size, 5),
              hidden_cell_dim=64, return_sequences=False))
    model.add(Dense(size=5, input_size=64, activation="linear"))
    model.summary()
    predictions = model.run(inputs=train_data_x)

    ic(predictions)
    ic(predictions.shape)

    model.save_model(f"model-tes-{window_size}.json")


if __name__ == "__main__":
    # get_data()
    # forward_prop_stock_market()
    # forward_prop_stock_market_load_model()

    perform_experiment(4)
