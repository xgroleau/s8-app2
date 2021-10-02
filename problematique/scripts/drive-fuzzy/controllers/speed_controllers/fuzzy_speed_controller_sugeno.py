import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from skfuzzy import control as ctrl

class FuzzySpeedControllerSugeno:
    def __init__(self):
        self.sim = createFuzzyController()
        
    def calculateAcceleration(self, state):
        maxSpeedDist = 95.0
        maxSpeed = 150.0
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
                
        else:
            targetSpeed = 0
        
        
        self.sim.input['desiredSpeed'] = targetSpeed
        self.sim.input['currentSpeed'] = curSpeedX
        
        self.sim.compute()
        
        accel = self.sim.output['accelCmd']
    
        if accel > 0:
            accel = accel
            brake = 0.0
        else:
            brake = -accel
            accel = 0.0

        brake = np.clip(brake, 0.0, 1.0)
        accel = np.clip(accel, 0.0, 1.0)

        return accel, brake
    
def singletonmf(x, a):
    """
    Singleton membership function generator.
    Parameters
    ----------
    x : 1d array
        Independent variable.
    a : constant
    Returns
    -------
    y : 1d array
        Singleton membership function.
    """
    y = np.zeros(len(x))

    if a >= np.min(x) and a <= np.max(x):
        idx = (np.abs(x - a)).argmin()
        y[idx] = 1.0

    return y

def createFuzzyController():
    # Fuzzy variables (Universe of Discourse)
    # Inputs
    desiredSpeed = ctrl.Antecedent(np.linspace(0, 100, 1000), 'desiredSpeed')
    currentSpeed = ctrl.Antecedent(np.linspace(0, 100, 1000), 'currentSpeed')
    
    # Outputs
    accelCmd = ctrl.Consequent(np.linspace(-1, 1, 1000), 'accelCmd', defuzzify_method='centroid')
    
    # Accumulation methods
    accelCmd.accumulation_method = np.fmax
    
    # Membership Functions
    desiredSpeed['slow'] = fuzz.trapmf(desiredSpeed.universe, [0, 0, 15, 50])
    desiredSpeed['medium'] = fuzz.trapmf(desiredSpeed.universe, [40, 60, 85, 100])
    desiredSpeed['fast'] = fuzz.trapmf(desiredSpeed.universe, [90, 100, 100, 100])
    
    currentSpeed['slow'] = fuzz.trapmf(currentSpeed.universe, [0, 0, 15, 50])
    currentSpeed['medium'] = fuzz.trapmf(currentSpeed.universe, [40, 60, 85, 100])
    currentSpeed['fast'] = fuzz.trapmf(currentSpeed.universe, [90, 100, 100, 100])

    
    accelCmd['maxBrake'] = singletonmf(accelCmd.universe, -1)
    accelCmd['brake'] = singletonmf(accelCmd.universe, -0.4)
    accelCmd['cruise'] = singletonmf(accelCmd.universe, 0.5)
    accelCmd['accel'] = singletonmf(accelCmd.universe, 1)
    #accelCmd['maxAccel'] = singletonmf(accelCmd.universe, 1)
    
    # Rules
    rules = []
    rules.append(ctrl.Rule(antecedent=(desiredSpeed['slow'] & currentSpeed['slow']), consequent=accelCmd['cruise']))
    rules.append(ctrl.Rule(antecedent=(desiredSpeed['slow'] & currentSpeed['medium']), consequent=accelCmd['brake']))
    rules.append(ctrl.Rule(antecedent=(desiredSpeed['slow'] & currentSpeed['fast']), consequent=accelCmd['maxBrake']))
    
    rules.append(ctrl.Rule(antecedent=(desiredSpeed['medium'] & currentSpeed['slow']), consequent=accelCmd['accel']))
    rules.append(ctrl.Rule(antecedent=(desiredSpeed['medium'] & currentSpeed['medium']), consequent=accelCmd['cruise']))
    rules.append(ctrl.Rule(antecedent=(desiredSpeed['medium'] & currentSpeed['fast']), consequent=accelCmd['brake']))
    
    rules.append(ctrl.Rule(antecedent=(desiredSpeed['fast'] & currentSpeed['slow']), consequent=accelCmd['accel']))
    rules.append(ctrl.Rule(antecedent=(desiredSpeed['fast'] & currentSpeed['medium']), consequent=accelCmd['cruise']))
    rules.append(ctrl.Rule(antecedent=(desiredSpeed['fast'] & currentSpeed['fast']), consequent=accelCmd['brake']))
    
    for rule in rules:
        rule.and_func = np.multiply
        rule.or_func = np.fmax

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)

    for var in sim.ctrl.fuzzy_variables:
        var.view()
    plt.show()
    
    return sim

createFuzzyController()