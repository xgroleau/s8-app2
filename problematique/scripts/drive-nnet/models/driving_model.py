from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def create(lr):
    model = Sequential()
    model.add(Dense(units=128, activation='sigmoid', input_shape=(8,)))
    model.add(Dense(units=32, activation='sigmoid'))
    model.add(Dense(units=8, activation='sigmoid'))
    model.add(Dense(units=2, activation='tanh'))
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')

    print(model.summary())

    return model


def create_trained(dataset, lr=0.001):
    # Driving
    x_driving = np.dstack((dataset.angle, dataset.speed_x, dataset.speed_y, dataset.trackPos,dataset.track[:, 8], dataset.track[:, 9], dataset.track[:, 10], dataset.curveVal)).squeeze()
    y_driving = np.dstack((dataset.accelBrakeCmd, dataset.steerCmd)).squeeze()

    x_train, x_test, y_train, y_test = train_test_split(x_driving, y_driving, shuffle=True, test_size=0.15)
    
    model = create(lr)
    history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), shuffle=False, verbose=1)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Drive model loss LR {lr}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(["train_loss", "val_loss"])
    # plt.savefig(f"figures/loss/drive-loss-{lr}.png")
    plt.show()
    return model


def predict(model, observation, dataset):
    # Input
    angle_val = dataset.normalize_angle(observation['angle'][0])
    speed_x_val = dataset.normalize_speed_x(observation['speed'][0])
    speed_y_val = dataset.normalize_speed_y(observation['speed'][1])
    trackPos_val = dataset.normalize_trackPos(observation['trackPos'][0])
    track_val = dataset.normalize_track(observation['track'])
    curve_val = dataset.normalize_curveVal(dataset.calculate_curve_val(observation['track'], observation['trackPos']))

    driving_input = np.array([[angle_val, speed_x_val, speed_y_val, trackPos_val, track_val[8],track_val[9],track_val[10], curve_val]])
    
    # Prediction
    prediction_driving = model.predict(driving_input).squeeze()

    # Extract values from predictions
    accel_action = min(prediction_driving[0], 1) if prediction_driving[0] > 0 else 0
    brake_action = min(abs(prediction_driving[0]), 1) if prediction_driving[0] < 0 else 0
    steer_action = min(prediction_driving[1], 1)

    return {
        'accel': np.array([accel_action], dtype=np.float32),
        'brake': np.array([brake_action], dtype=np.float32),
        'steer': np.array([steer_action], dtype=np.float32),
    }