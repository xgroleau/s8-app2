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


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split

# DEPRECATED DO NOT USE
# Predicts the acceleration and brakes of the car. Used with the DriveBot
def create():
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_shape=(31,)))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=2, activation='tanh'))
    model.compile(optimizer=Adam(lr=0.005), loss='mean_squared_error')

    print(model.summary())

    return model


def create_trained(dataset):
    # Driving
    x_driving = np.dstack((dataset.angle, dataset.speed_x, dataset.speed_y, dataset.trackPos)).squeeze()
    x_driving = np.column_stack((x_driving, dataset.track, dataset.gear))
    y_driving = np.dstack((dataset.accelBrakeCmd, dataset.steerCmd)).squeeze()

    # TODO: check if we split
    #x_train_drive, x_test_drive, y_train_drive, y_test_drive = train_test_split(x_driving, y_driving, shuffle=True, test_size=0.15)
    
    model = create()
    hist = model.fit(x_train_drive, y_train_drive, batch_size=300, epochs=5, shuffle=False, verbose=1)
    return model


def predict(model, observation, dataset):
    # Input
    angle_val = dataset.normalize_angle(observation['angle'][0])
    speed_x_val = dataset.normalize_speed_x(observation['speed'][0])
    speed_y_val = dataset.normalize_speed_y(observation['speed'][1])
    trackPos_val = dataset.normalize_trackPos(observation['trackPos'][0])
    track_val = dataset.normalize_track(observation['track'])
    gear_val = dataset.gear_to_categorical(observation['gear'])
    
    driving_input = np.array([angle_val, speed_x_val, speed_y_val, trackPos_val])
    driving_input = np.array([np.concatenate((driving_input, track_val, gear_val.squeeze()))])
    
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