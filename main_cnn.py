import os

import numpy as np
from src import Sequential
from src.layer import Convolutional, Detector, Flatten, Dense, Pooling
from icecream import ic

from PIL import Image

PANDAS_BEARS = "./datasets/PandasBears"


def get_data():
    train = {"x": [], "y": []}

    for d in os.listdir(os.path.join(PANDAS_BEARS, "Train/Bears/")):
        try:
            # print(d)
            image = Image.open(os.path.join(PANDAS_BEARS, "Train/Bears/", d))
            train["x"].append(np.asarray(image))
            train["y"].append("bear")
        except Exception as e:
            print(e)
    for d in os.listdir(os.path.join(PANDAS_BEARS, "Train/Pandas/")):
        try:
            # print(d)
            image = Image.open(os.path.join(PANDAS_BEARS, "Train/Pandas/", d))
            train["x"].append(np.asarray(image))
            train["y"].append("panda")
        except Exception as e:
            print(e)

    return train


NROW = 10


def forward_prop_panda_bear():
    train_data = get_data()

    train_x, train_y = train_data["x"], train_data["y"]  # noqa: F841
    train_x = np.asarray(train_x)

    train_x = train_x.reshape((500, 3, 256, 256))

    train_x_cut = train_x[:NROW]
    train_y_cut = train_y[:NROW]  # noqa: F841

    train_x_cut = train_x_cut.astype("float")
    train_x_cut /= 255

    model = Sequential()
    model.add(
        Convolutional(
            input_shape=(3, 256, 256),
            padding=0,
            filter_count=2,
            kernel_shape=(3, 3),
            stride=1,
        )
    )
    model.add(Detector(activation="relu"))
    model.add(Pooling(size=(2, 2), stride=2,
              mode="max", input_shape=(254, 254, 2)))
    model.add(Pooling(size=(5, 5), stride=5,
              mode="avg", input_shape=(127, 127, 2)))
    model.add(Flatten(input_shape=(25, 25, 2)))
    model.add(Dense(size=10, input_size=1250, activation="relu"))
    model.add(Dense(size=1, input_size=10, activation="sigmoid"))

    model.summary()

    predictions = model.run(inputs=train_x_cut)

    model.save_model("model-image-convnet.json")

    ic(predictions)


def forward_prop_panda_bear_load_model():
    train_data = get_data()

    train_x, train_y = train_data["x"], train_data["y"]  # noqa: F841
    train_x = np.asarray(train_x)

    train_x = train_x.reshape((500, 3, 256, 256))

    train_x_cut = train_x[:NROW]
    train_y_cut = train_y[:NROW]  # noqa: F841

    train_x_cut = train_x_cut.astype("float")
    train_x_cut /= 255

    model = Sequential()
    model.load_model("model-image-convnet.json")

    predictions = model.run(inputs=train_x_cut)

    ic(predictions)


def test_load_model():
    model = Sequential()
    model.load_model("model.json")


if __name__ == "__main__":
    # test_load_model()
    forward_prop_panda_bear()
    # forward_prop_panda_bear_load_model()
