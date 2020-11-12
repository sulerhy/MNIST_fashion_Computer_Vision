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
from PIL import Image

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
    testing_examples(load_model(), X_test[0:10, ], y_test[0:10, ])


def testing_examples(model, inputs, grounds):
    results = model.predict(inputs)
    result_keys = np.argmax(results, 1)
    grounds = np.argmax(grounds, 1)
    for i in range(inputs.shape[0]):
        image_matrix = inputs[i, :, :]
        image_matrix = np.reshape(image_matrix, (28, 28))
        label_name = utils.get_label_name(result_keys[i])
        ground = utils.get_label_name(grounds[i])
        print("LABEL " + str(i) + ":------------" + label_name + "---------------")
        plt.suptitle("Predict:  " + label_name + "\nGround True: " + ground)
        plt.gray()
        plt.imshow(image_matrix)
        plt.show()
        input("Type anything to next")
        plt.close()
        continue


if __name__ == "__main__":
    # execute only if run as a script
    main()
