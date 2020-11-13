"""
Copyright (C) Hoang Pham Duc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Hoang Pham Duc <phamduchoangeee@gmail.com>, Oct 2020
"""

import sys
import json
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import utils
import matplotlib.pyplot as plt

import numpy as np

# global static variables
n_x = 128
n_y = 64
X_shape = (-1, 28, 28)

epochs_num = 15
batch_size = 32


def generate_optimizer():
    return keras.optimizers.Adam()


def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer=generate_optimizer(),
                  metrics=['accuracy'])


def generate_model(labels_number):
    sys.stdout.write('Loading new model\n\n')
    sys.stdout.flush()

    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(labels_number))
    model.add(Activation('softmax'))

    compile_model(model)

    with open('./models/convnet_model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
        outfile.close()

    return model


def train(model, X_train, y_train, X_test, y_test):
    sys.stdout.write('Training model\n\n')
    sys.stdout.flush()

    # train each iteration individually to back up current state
    # safety measure against potential crashes

    sys.stdout.flush()
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        epochs=epochs_num,
                        verbose=1)
    model.save_weights('./models/convnet_weights.h5')
    # print(history)
    # print loss and val_loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    sys.stdout.write("Saving Done, training complete!")
    return model


def main():
    sys.stdout.write(utils.starting_text() + '\n\n')
    sys.stdout.flush()
    LABELS_NUM, X_train, Y_train, X_test, Y_test = utils.load_all_data()
    print("Labels number:" + str(LABELS_NUM))
    print("X_train:" + str(np.shape(X_train)))
    print("Y_train:" + str(np.shape(Y_train)))
    print("X_test:" + str(np.shape(X_test)))
    print("Y_test:" + str(np.shape(Y_test)))

    model = generate_model(LABELS_NUM)
    model = train(model, X_train, Y_train, X_test, Y_test)




if __name__ == "__main__":
    # execute only if run as a script
    main()
