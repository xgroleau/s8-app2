from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def create(lr):
    model = Sequential()
    model.add(Dense(units=18, activation='relu', input_shape=(9,)))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=2, activation='sigmoid'))
    model.compile(optimizer=Adam(lr), loss='mean_squared_error')

    print(model.summary())

    return model


def create_trained(dataset, lr=0.005):
    x_accel = np.dstack((dataset.speed_x, dataset.trackPos)).squeeze()
    x_accel = np.column_stack((x_accel, dataset.track[:, 8:11], dataset.wheelSpinVel))
    y_accel = np.dstack((dataset.accelCmd, dataset.brakeCmd)).squeeze()

    x_train, x_test, y_train, y_test = train_test_split(x_accel, y_accel, shuffle=True, test_size=0.15)

    model = create(lr)
    history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test), shuffle=False, verbose=1)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Acceleration model loss LR {lr}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(["train_loss", "val_loss"])
    plt.savefig(f"figures/loss/accel-loss-{lr}.png")
    plt.show()
    return model


def predict(model, observation, dataset):
    # Input
    speed_x_val = dataset.normalize_speed_x(observation['speed'][0])
    track_val = dataset.normalize_track(observation['track'])[8:11]
    trackPos_val = dataset.normalize_trackPos(observation['trackPos'][0])
    wheelSpinVel_val = dataset.normalize_wheelSpinVel(observation['wheelSpinVel'])

    accel_input = np.array([[speed_x_val, trackPos_val, *track_val, *wheelSpinVel_val]])

    # Prediction
    prediction_driving = model.predict(accel_input).squeeze()

    # Extract values from predictions
    accel_action = min(prediction_driving[0], 1)
    brake_action = min(prediction_driving[1], 1)

    return {
        'accel': np.array([accel_action], dtype=np.float32),
        'brake': np.array([brake_action], dtype=np.float32),
    }