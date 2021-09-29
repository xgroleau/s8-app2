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
import sys
import time
import numpy as np
import logging

sys.path.append('../..')
from torcs.control.core import TorcsControlEnv, TorcsException, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


################################
# Define helper functions here
################################

class SimpleController(object):

    def __init__(self, absEnabled=True):
        self.absEnabled = absEnabled

    # usage: GEAR = calculateGear(STATE)
    #
    # Calculate the gear of the transmission for the current car state.
    # Adapted from the code of the WCCI2008 example C++ client:
    # http://cig.ws.dei.polimi.it/wp-content/uploads/2008/04/client-cpp_v02.tgz
    #
    # Input:
    # - STATE, a structure describing the current state of the car (see function 'waitForState').
    #
    # Output:
    # - GEAR, the selected gear. -1 is reverse, 0 is neutral and the forward gear can range from 1 to 6.
    #
    def _calculateGear(self, state):

        # Gear Changing Constants
        GEAR_UP = [5000, 6000, 6000, 6500, 7000, 0]
        GEAR_DOWN = [0, 2500, 3000, 3000, 3500, 3500]

        curGear = state['gear'][0]
        curRpm = state['rpm'][0]

        # If gear is 0 (N) or -1 (R) just return 1
        if curGear < 1:
            nextGear = 1
        # Check if the RPM value of car is greater than the one suggested
        # to shift up the gear from the current one.
        elif curGear < 6 and curRpm >= GEAR_UP[curGear - 1]:
            nextGear = curGear + 1
        # Check if the RPM value of car is lower than the one suggested
        # to shift down the gear from the current one.
        elif curGear > 1 and curRpm <= GEAR_DOWN[curGear - 1]:
            nextGear = curGear - 1
        else:
            # Otherwise keep current gear
            nextGear = curGear

        return nextGear

    # usage: STEERING = calculateSteering(STATE)
    #
    # Calculate the steering value for the current car state.
    # Adapted from the code of the WCCI2008 example C++ client:
    # http://cig.ws.dei.polimi.it/wp-content/uploads/2008/04/client-cpp_v02.tgz
    #
    # Input:
    # - STATE, a structure describing the current state of the car (see function 'waitForState').
    #
    # Output:
    # - STEERING, the steering value. -1 and +1 means respectively full left and right, that corresponds to an angle of 0.785398 rad.
    #
    def _calculateSteering(self, state):
        # Steering constants
        steerLock = 0.785398
        steerSensitivityOffset = 80.0
        wheelSensitivityCoeff = 1.0

        curAngle = state['angle'][0]
        curTrackPos = state['trackPos'][0]
        curSpeedX = state['speed'][0]

        # Steering angle is computed by correcting the actual car angle w.r.t. to track
        # axis and to adjust car position w.r.t to middle of track
        targetAngle = curAngle - curTrackPos * 2.0

        # At high speed, reduce the steering command to avoid loosing control
        if curSpeedX > steerSensitivityOffset:
            steering = targetAngle / (steerLock * (curSpeedX - steerSensitivityOffset) * wheelSensitivityCoeff)
        else:
            steering = targetAngle / steerLock

        # Normalize steering
        steering = np.clip(steering, -1.0, 1.0)

        return steering

    # usage: ACCELERATION = calculateAcceleration(STATE)
    #
    # Calculate the accelerator (gas pedal) value for the current car state.
    # Adapted from the code of the WCCI2008 example C++ client:
    # http://cig.ws.dei.polimi.it/wp-content/uploads/2008/04/client-cpp_v02.tgz
    #
    # Input:
    # - STATE, a structure describing the current state of the car (see function 'waitForState').
    #
    # Output:
    # - ACCELERATION, the virtual gas pedal (0 means no gas, 1 full gas), in the range [0,1].
    #
    def _calculateAcceleration(self, state):

        # Accel and Brake Constants
        maxSpeedDist = 95.0
        maxSpeed = 100.0
        sin10 = 0.17365
        cos10 = 0.98481
        angleSensitivity = 2.0

        curSpeedX = state['speed'][0]
        curTrackPos = state['trackPos'][0]

        # checks if car is out of track
        if (curTrackPos < 1 and curTrackPos > -1):

            # Reading of sensor at +10 degree w.r.t. car axis
            rxSensor = state['track'][8]
            # Reading of sensor parallel to car axis
            cSensor = state['track'][9]
            # Reading of sensor at -5 degree w.r.t. car axis
            sxSensor = state['track'][10]

            # Track is straight and enough far from a turn so goes to max speed
            if cSensor > maxSpeedDist or (cSensor >= rxSensor and cSensor >= sxSensor):
                targetSpeed = maxSpeed
            else:
                # Approaching a turn on right
                if rxSensor > sxSensor:
                    # Computing approximately the "angle" of turn
                    h = cSensor * sin10
                    b = rxSensor - cSensor * cos10
                    angle = np.arcsin(b * b / (h * h + b * b))

                # Approaching a turn on left
                else:
                    # Computing approximately the "angle" of turn
                    h = cSensor * sin10
                    b = sxSensor - cSensor * cos10
                    angle = np.arcsin(b * b / (h * h + b * b))

                # Estimate the target speed depending on turn and on how close it is
                targetSpeed = maxSpeed * (cSensor * np.sin(angle) / maxSpeedDist) * angleSensitivity
                targetSpeed = np.clip(targetSpeed, 0.0, maxSpeed)

            # Accel/brake command is exponentially scaled w.r.t. the difference
            # between target speed and current one
            accel = (2.0 / (1.0 + np.exp(curSpeedX - targetSpeed)) - 1.0)

        else:
            # when out of track returns a moderate acceleration command
            accel = 0.3

        if accel > 0:
            accel = accel
            brake = 0.0
        else:
            brake = -accel
            accel = 0.0

            if self.absEnabled:
                # apply ABS to brake
                brake = self._filterABS(state, brake)

        brake = np.clip(brake, 0.0, 1.0)
        accel = np.clip(accel, 0.0, 1.0)

        return accel, brake

    def _filterABS(self, state, brake):

        wheelRadius = [0.3179, 0.3179, 0.3276, 0.3276]
        absSlip = 2.0
        absRange = 3.0
        absMinSpeed = 3.0

        curSpeedX = state['speed'][0]

        # convert speed to m/s
        speed = curSpeedX / 3.6

        # when speed lower than min speed for abs do nothing
        if speed >= absMinSpeed:
            # compute the speed of wheels in m/s
            slip = np.dot(state['wheelSpinVel'], wheelRadius)
            # slip is the difference between actual speed of car and average speed of wheels
            slip = speed - slip / 4.0
            # when slip too high apply ABS
            if slip > absSlip:
                brake = brake - (slip - absSlip) / absRange

            # check brake is not negative, otherwise set it to zero
            brake = np.clip(brake, 0.0, 1.0)

        return brake

    # usage: ACTION = drive(STATE)
    #
    # Calculate the accelerator, brake, gear and steering values based on the current car state.
    #
    # Input:
    # - STATE, a structure describing the current state of the car (see function 'waitForState').
    #
    # Output:
    # - ACTION, the structure describing the action to execute (see function 'applyAction').
    #
    def drive(self, state):
        accel, brake = self._calculateAcceleration(state)
        gear = self._calculateGear(state)
        steer = self._calculateSteering(state)

        action = {'accel': np.array([accel], dtype=np.float32),
                  'brake': np.array([brake], dtype=np.float32),
                  'gear': np.array([gear], dtype=np.int32),
                  'steer': np.array([steer], dtype=np.float32)}
        return action


def main():

    recordingsPath = os.path.join(CDIR, 'recordings')
    if not os.path.exists(recordingsPath):
        os.makedirs(recordingsPath)

    try:
        with TorcsControlEnv(render=False) as env:
            controller = SimpleController()

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
                        # Select the next action based on the observation
                        action = controller.drive(observation)
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
