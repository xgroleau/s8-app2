from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np


def create():
    model = Sequential()
    model.add(Dense(units=18, activation='relu', input_shape=(9,)))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=2, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=0.005), loss='mean_squared_error')

    print(model.summary())

    return model


def create_trained(dataset):
    # Driving
    x_accel = np.dstack((dataset.speed_x, dataset.trackPos)).squeeze()
    x_accel = np.column_stack((x_accel, dataset.track[:, 8:11], dataset.wheelSpinVel))
    y_accel = np.dstack((dataset.accelCmd, dataset.brakeCmd)).squeeze()

    # TODO: check if we split
    # x_train_drive, x_test_drive, y_train_drive, y_test_drive = train_test_split(x_driving, y_driving, shuffle=True, test_size=0.15)

    model = create()
    model.fit(x_accel, y_accel, batch_size=300, epochs=5, shuffle=False, verbose=1)
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