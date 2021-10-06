# Author: Xavier Groleau <xavier.groleau@@usherbrooke.ca>
# Author: Charles Quesnel <charles.quesnel@@usherbrooke.ca>
# Author: Michael Samson <michael.samson@@usherbrooke.ca>
# Université de Sherbrooke, APP2 S8GIA, A2018

import numpy as np

# Taken from drive-simple
class SimpleSteeringController:
    def __init__(self):
        pass
    
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
    def calculateSteering(self, state):
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