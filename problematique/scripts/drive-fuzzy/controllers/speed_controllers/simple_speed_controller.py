import numpy as np

class SimpleSpeedController:
    def __init__(self):
        self.absEnabled = False
    
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
    def calculateAcceleration(self, state):

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
            # Reading of sensor at -10 degree w.r.t. car axis
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