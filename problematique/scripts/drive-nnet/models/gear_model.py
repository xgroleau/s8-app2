from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np


def create():
    model = Sequential()
    model.add(Dense(units=12, activation='sigmoid', input_shape=(9,)))
    model.add(Dense(units=8, activation='softmax'))
    model.compile(optimizer=Adam(lr=0.005), loss='mse')

    print(model.summary())

    return model


def create_trained(dataset):
    x_gear = np.column_stack((dataset.rpm, dataset.gear))
    y_gear = dataset.gearCmd

    # TODO: check if we split
    #x_train_gear, x_test_gear, y_train_gear, y_test_gear = train_test_split(x_gear, y_gear, shuffle=True, test_size=0.15)
    model = create()

    model.fit(x_gear, y_gear, batch_size=300, epochs=5, shuffle=False, verbose=1)
    return model


def predict(model, observation, dataset):
    gear_val = dataset.gear_to_categorical(observation['gear'])
    rpm_val = dataset.normalize_rpm(observation['rpm'][0])

    gear_input = np.column_stack((rpm_val, gear_val))
    gear_action = np.array([dataset.gear_to_int(model.predict(gear_input).squeeze())])

    return {
        'gear': gear_action
    }