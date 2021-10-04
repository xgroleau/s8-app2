from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def create(lr):
    model = Sequential()
    model.add(Dense(units=16, activation='sigmoid', input_shape=(11,)))
    model.add(Dense(units=9, activation='softmax'))
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')

    print(model.summary())

    return model


def create_trained(dataset, lr=0.001):
    x_gear = np.column_stack((dataset.rpm, dataset.speed_x, dataset.gear))
    y_gear = dataset.gearCmd

    x_train, x_test, y_train, y_test = train_test_split(x_gear, y_gear, shuffle=True, test_size=0.15)
    model = create(lr)

    history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test), shuffle=False, verbose=1)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Gear model loss LR {lr}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(["train_loss", "val_loss"])
    # plt.savefig(f"figures/loss/gear-loss-{lr}.png")
    plt.show()
    return model


def predict(model, observation, dataset):
    gear_val = dataset.gear_to_categorical(observation['gear'])
    speed_x = dataset.normalize_rpm(observation['speed'][0])
    rpm_val = dataset.normalize_rpm(observation['rpm'][0])

    gear_input = np.column_stack((rpm_val, speed_x, gear_val))
    gear_action = np.array([dataset.gear_to_int(model.predict(gear_input).squeeze())])

    return {
        'gear': gear_action
    }