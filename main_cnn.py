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

    print(len(train))
    return train


def forward_prop_panda_bear():
    # print("Loading Panda Bear Dataset...")
    train_data = get_data()

    train_x, train_y = train_data["x"], train_data["y"]  # noqa: F841
    train_x = np.asarray(train_x)

    train_x = train_x.reshape((500, 3, 256, 256))
    # print(f"Training shape: {train_x.shape}")

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
    model.add(Pooling(size=(2, 2), stride=2, mode="max"))
    model.add(Flatten())
    model.add(Dense(size=10, input_size=32258, activation="sigmoid"))
    model.add(Dense(size=10, input_size=10, activation="sigmoid"))

    # print(train_x.shape)
    result = model.run(inputs=train_x)

    model.save_model()

    ic(result)
    ic(result.shape)


def test_load_model():
    model = Sequential()
    model.load_model("model.json")


if __name__ == "__main__":
    test_load_model()
    forward_prop_panda_bear()
