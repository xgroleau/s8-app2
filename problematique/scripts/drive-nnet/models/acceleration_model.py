# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA,
# OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# Author: Xavier Groleau <xavier.groleau@@usherbrooke.ca>
# Author: Charles Quesnel <charles.quesnel@@usherbrooke.ca>
# Author: Michael Samson <michael.samson@@usherbrooke.ca>
# Université de Sherbrooke, APP2 S8GIA, A2018


from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime

# Model for predicting the required acceleration of the car
def create(lr, l1, l2):
    """Creates the model, the learning rate and number or neurones can be passed as argument"""
    model = Sequential()
    model.add(Dense(units=l1, activation='sigmoid', input_shape=(9,)))
    model.add(Dense(units=l2, activation='sigmoid'))
    model.add(Dense(units=2, activation='sigmoid'))
    model.compile(optimizer=Adam(lr), loss='mean_squared_error')

    print(model.summary())

    return model


def create_trained(dataset, lr=0.001, l1=18, l2=6):
    """Creates a trained model and logs the data for tensorboard, returns the model"""
    log_dir = f"logs/accel-lr-{lr}-l1-{l1}-l2{l2}"

    x_accel = np.dstack((dataset.speed_x, dataset.trackPos)).squeeze()
    x_accel = np.column_stack((x_accel, dataset.track[:, 8:11], dataset.wheelSpinVel))
    y_accel = np.dstack((dataset.accelCmd, dataset.brakeCmd)).squeeze()

    x_train, x_test, y_train, y_test = train_test_split(x_accel, y_accel, shuffle=True, test_size=0.15)

    model = create(lr, l1, l2)
    history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), shuffle=False,
                verbose=1, callbacks=[
                tf.keras.callbacks.TensorBoard(log_dir)
                ])

    return model


def predict(model, observation, dataset):
    """Predicts using the model via the observation, the dataset is required to normalize the inputs
    returns a dictionary of the actions"""
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