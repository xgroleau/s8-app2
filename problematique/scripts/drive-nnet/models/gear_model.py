from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import tensorflow as tf


def create(lr, l1):
    model = Sequential()
    model.add(Dense(units=l1, activation='sigmoid', input_shape=(9,)))
    model.add(Dense(units=8, activation='softmax'))
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')

    print(model.summary())

    return model


def create_trained(dataset, lr=0.0005, l1=12):
    log_dir = f"logs/gear-lr-{lr}-l1-{l1}" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    x_gear = np.column_stack((dataset.rpm, dataset.gear))
    y_gear = dataset.gearCmd

    x_train, x_test, y_train, y_test = train_test_split(x_gear, y_gear, shuffle=True, test_size=0.15)
    model = create(lr, l1)

    history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), shuffle=False, verbose=1,
                        callbacks=[
                            tf.keras.callbacks.TensorBoard(log_dir)
                        ])

    return model


def predict(model, observation, dataset):
    gear_val = dataset.gear_to_categorical(observation['gear'])
    rpm_val = dataset.normalize_rpm(observation['rpm'][0])

    gear_input = np.column_stack((rpm_val, gear_val))
    gear_action = np.array([dataset.gear_to_int(model.predict(gear_input).squeeze())])

    return {
        'gear': gear_action
    }