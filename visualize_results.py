"""
Copyright (C) Hoang Pham Duc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Hoang Pham Duc <phamduchoangeee@gmail.com>, May 2020
"""

import sys
import json
from pathlib import Path
import numpy as np
import keras
import utils
import matplotlib.pyplot as plt

model_path = './models/convnet_model.json'
weight_path = './models/convnet_weights.h5'
np.set_printoptions(threshold=sys.maxsize)


def load_model():
    if not Path(model_path).is_file():
        sys.stdout.write('Please train model using basic_model.py first')
        sys.stdout.flush()
        raise SystemExit

    with open(model_path) as file:
        model = keras.models.model_from_json(json.load(file))
        file.close()

    model.load_weights(weight_path)

    return model


def main():
    LABELS_NUM, X_train, y_train, X_test, y_test = utils.load_all_data()
    model = load_model()
    # print_accuracy(model, X_train, y_train, X_test, y_test)
    # print loss and val_loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()
