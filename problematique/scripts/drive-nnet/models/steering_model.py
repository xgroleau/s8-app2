from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np


def create():
    model = Sequential()
    model.add(Dense(units=8, activation='relu', input_shape=(3,)))
    model.add(Dense(units=1, activation='tanhS'))
    model.compile(optimizer=Adam(lr=0.005), loss='mean_squared_error')

    print(model.summary())

    return model


def create_trained(dataset):
    # Driving
    x_driving = np.dstack((dataset.angle, dataset.speed_x, dataset.trackPos)).squeeze()
    y_driving = np.dstack((dataset.steerCmd, )).squeeze()

    # TODO: check if we split
    # x_train_drive, x_test_drive, y_train_drive, y_test_drive = train_test_split(x_driving, y_driving, shuffle=True, test_size=0.15)

    model = create()
    model.fit(x_driving, y_driving, batch_size=300, epochs=5, shuffle=False, verbose=1)
    return model


def predict(model, observation, dataset):
    # Input
    angle_val = dataset.normalize_angle(observation['angle'][0])
    speed_x_val = dataset.normalize_speed_x(observation['speed'][0])
    trackPos_val = dataset.normalize_trackPos(observation['trackPos'][0])

    accel_input = np.array([[angle_val, speed_x_val, trackPos_val]])

    # Prediction
    prediction_driving = model.predict(accel_input)[0]

    # Extract values from predictions
    steer_action = min(prediction_driving[0], 1)

    return {
        'steer': np.array([steer_action], dtype=np.float32),
    }
