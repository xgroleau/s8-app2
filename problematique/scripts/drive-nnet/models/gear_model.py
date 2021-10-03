from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def create():
    model = Sequential()
    model.add(Dense(units=12, activation='sigmoid', input_shape=(9,)))
    model.add(Dense(units=8, activation='softmax'))
    model.compile(optimizer=Adam(lr=0.0005), loss='mean_squared_error')

    print(model.summary())

    return model


def create_trained(dataset):
    x_gear = np.column_stack((dataset.rpm, dataset.gear))
    y_gear = dataset.gearCmd

    x_train, x_test, y_train, y_test = train_test_split(x_gear, y_gear, shuffle=True, test_size=0.15)
    model = create()

    history = model.fit(x_train, y_train, batch_size=64, epochs=15, shuffle=False, verbose=1)
    loss = model.evaluate(x_test, y_test)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Gear model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("figures/loss/gear-loss-0005")
    plt.show()
    print(f"Gear model loss on test set: {loss}")
    return model


def predict(model, observation, dataset):
    gear_val = dataset.gear_to_categorical(observation['gear'])
    rpm_val = dataset.normalize_rpm(observation['rpm'][0])

    gear_input = np.column_stack((rpm_val, gear_val))
    gear_action = np.array([dataset.gear_to_int(model.predict(gear_input).squeeze())])

    return {
        'gear': gear_action
    }