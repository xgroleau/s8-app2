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

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

import os
from os import listdir
from os.path import isfile, join
import sys
import time
import logging


import numpy as np
from sklearn.model_selection import train_test_split

from models import driving_model, gear_model
from data.dataset import DataSet

sys.path.append('../..')
from torcs.control.core import TorcsControlEnv, TorcsException, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


################################
# Define helper functions here
################################
def main():
    
    recordingsPath = os.path.join(CDIR, 'recordings')
    if not os.path.exists(recordingsPath):
        os.makedirs(recordingsPath)

    # Training model
    files_dir = os.path.join(CDIR, 'data_set')

    dataset = DataSet(files_dir)

    # Normalisation
    dataset.normalize()

    # Driving
    x_driving = np.dstack((dataset.angle, dataset.speed_x, dataset.speed_y, dataset.trackPos)).squeeze()
    x_driving = np.column_stack((x_driving, dataset.track, dataset.gear))
    y_driving = np.dstack((dataset.accelBrakeCmd, dataset.steerCmd)).squeeze()

    x_train_drive, x_test_drive, y_train_drive, y_test_drive = train_test_split(x_driving, y_driving, shuffle=True, test_size=0.15)
    model_drive = driving_model.create_driving_model()

    model_drive.fit(x_train_drive, y_train_drive, batch_size=300, epochs=5, shuffle=False, verbose=1)

    # Gear model
    x_gear = np.column_stack((dataset.rpm, dataset.gear))
    y_gear = dataset.gearCmd

    x_train_gear, x_test_gear, y_train_gear, y_test_gear = train_test_split(x_gear, y_gear, shuffle=True, test_size=0.15)
    model_gear = gear_model.create_gear_model()

    model_gear.fit(x_train_gear, y_train_gear, batch_size=300, epochs=5, shuffle=False, verbose=1)
    
    try:
        with TorcsControlEnv(render=True) as env:

            nbTracks = len(TorcsControlEnv.availableTracks)
            nbSuccessfulEpisodes = 0
            for episode in range(nbTracks):
                logger.info('Episode no.%d (out of %d)' % (episode + 1, nbTracks))
                startTime = time.time()

                observation = env.reset()
                trackName = env.getTrackName()

                nbStepsShowStats = 1000
                curNbSteps = 0
                done = False
                with EpisodeRecorder(os.path.join(recordingsPath, 'track-%s.pklz' % (trackName))) as recorder:
                    while not done:
                        # Parsing data_set and normalize input
                        
                        angle_val = dataset.normalize_angle(observation['angle'][0])
                        speed_x_val = dataset.normalize_speed_x(observation['speed'][0])
                        speed_y_val = dataset.normalize_speed_y(observation['speed'][1])
                        trackPos_val = dataset.normalize_trackPos(observation['trackPos'][0])
                        track_val = dataset.normalize_track(observation['track'])
                        gear_val = dataset.gear_to_categorical(observation['gear'])
                        rpm_val = dataset.normalize_rpm(observation['rpm'][0])

                        # Predictions
                        driving_input = np.array([angle_val, speed_x_val, speed_y_val, trackPos_val])
                        driving_input = np.array([np.concatenate((driving_input, track_val, gear_val.squeeze()))])
                        prediction_driving = model_drive.predict(driving_input).squeeze()

                        gear_input = np.column_stack((rpm_val, gear_val))
                        prediction_gear = dataset.gear_to_int(model_gear.predict(gear_input).squeeze())

                        # Extract values from predictions
                        accel_action = min(prediction_driving[0], 1) if prediction_driving[0] > 0 else 0
                        brake_action = min(abs(prediction_driving[0]), 1) if prediction_driving[0] < 0 else 0
                        steer_action = min(prediction_driving[1], 1)
                        gear_action = prediction_gear

                        # TODO: Select the next action based on the observation
                        action = {
                                'accel': np.array([accel_action], dtype=np.float32),
                                'brake': np.array([brake_action], dtype=np.float32),
                                'steer': np.array([steer_action], dtype=np.float32),
                                'gear': np.array(([gear_action]), dtype=np.int32)
                        }
                        recorder.save(observation, action)
    
                        # Execute the action
                        observation, reward, done, _ = env.step(action)
                        curNbSteps += 1
    
                        if observation and curNbSteps % nbStepsShowStats == 0:
                            curLapTime = observation['curLapTime'][0]
                            distRaced = observation['distRaced'][0]
                            logger.info('Current lap time = %4.1f sec (distance raced = %0.1f m)' % (curLapTime, distRaced))
    
                        if done:
                            if reward > 0.0:
                                logger.info('Episode was successful.')
                                nbSuccessfulEpisodes += 1
                            else:
                                logger.info('Episode was a failure.')
    
                            elapsedTime = time.time() - startTime
                            logger.info('Episode completed in %0.1f sec (computation time).' % (elapsedTime))

            logger.info('-----------------------------------------------------------')
            logger.info('Total number of successful tracks: %d (out of %d)' % (nbSuccessfulEpisodes, nbTracks))
            logger.info('-----------------------------------------------------------')

    except TorcsException as e:
        logger.error('Error occured communicating with TORCS server: ' + str(e))

    except KeyboardInterrupt:
        pass

    logger.info('All done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
