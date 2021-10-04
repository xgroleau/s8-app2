from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def create(lr):
    model = Sequential()
    model.add(Dense(units=9, activation='relu', input_shape=(3,)))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')

    print(model.summary())

    return model


def create_trained(dataset, lr=0.0001):
    x_steering = np.dstack((dataset.angle, dataset.speed_x, dataset.trackPos)).squeeze()
    y_steering = np.dstack((dataset.steerCmd, )).squeeze()

    x_train, x_test, y_train, y_test = train_test_split(x_steering, y_steering, shuffle=True, test_size=0.15)

    model = create(lr)
    history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test), shuffle=False, verbose=1)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Steering model loss LR {lr}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(["train_loss", "val_loss"])
    plt.savefig(f"figures/loss/steering-loss-{lr}.png")
    plt.show()
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
