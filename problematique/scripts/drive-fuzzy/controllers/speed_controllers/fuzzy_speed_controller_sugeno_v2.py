import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from skfuzzy import control as ctrl

class FuzzySpeedControllerSugenoV2:
    def __init__(self):
        self.sim = createFuzzyController()
        
    def calculateAcceleration(self, state):
        maxSpeedDist = 95.0
        sin10 = 0.17365
        cos10 = 0.98481

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
                angle = 0
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

            self.sim.input['roadCurve'] = angle
            self.sim.input['currentSpeed'] = curSpeedX
            self.sim.input['frontSensor'] = cSensor
        
            self.sim.compute()
        
            accel = self.sim.output['accelCmd']
            
        else:
            accel = 0.3        
    
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
    roadCurve = ctrl.Antecedent(np.linspace(0, np.pi/2, 1000), 'roadCurve')
    currentSpeed = ctrl.Antecedent(np.linspace(0, 120, 1000), 'currentSpeed')
    frontSensor = ctrl.Antecedent(np.linspace(0, 100, 1000), 'frontSensor')
    
    # Outputs
    accelCmd = ctrl.Consequent(np.linspace(-1, 1, 1000), 'accelCmd', defuzzify_method='centroid')
    
    # Accumulation methods
    accelCmd.accumulation_method = np.fmax
    
    # Membership Functions
    roadCurve['none'] = fuzz.trapmf(roadCurve.universe, [0, 0, 0.4, 0.8])
    roadCurve['small'] = fuzz.trapmf(roadCurve.universe, [0.6, 0.7, 0.9, 1])
    roadCurve['big'] = fuzz.trapmf(roadCurve.universe, [0.9, 1.2, np.pi/2, np.pi/2])
    
    currentSpeed['slow'] = fuzz.trapmf(currentSpeed.universe, [0, 0, 15, 40])
    currentSpeed['medium'] = fuzz.trapmf(currentSpeed.universe, [20, 50, 90, 120])
    currentSpeed['fast'] = fuzz.trapmf(currentSpeed.universe, [80, 100, 120, 120])
    
    frontSensor['close'] = fuzz.trapmf(frontSensor.universe, [0, 0, 20, 60])
    frontSensor['far'] = fuzz.trapmf(frontSensor.universe, [20, 60, 100, 100])

    
    accelCmd['maxBrake'] = singletonmf(accelCmd.universe, -1)
    accelCmd['noGas'] = singletonmf(accelCmd.universe, 0)
    accelCmd['cruise'] = singletonmf(accelCmd.universe, 0.5)
    accelCmd['accel'] = singletonmf(accelCmd.universe, 1)
    #accelCmd['maxAccel'] = singletonmf(accelCmd.universe, 1)
    
    # Rules
    rules = []
    rules.append(ctrl.Rule(antecedent=(frontSensor['far'] & roadCurve['none'] & currentSpeed['slow']), consequent=accelCmd['accel']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['far'] & roadCurve['none'] & currentSpeed['medium']), consequent=accelCmd['accel']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['far'] & roadCurve['none'] & currentSpeed['fast']), consequent=accelCmd['cruise']))
    
    rules.append(ctrl.Rule(antecedent=(frontSensor['far'] & roadCurve['small'] & currentSpeed['slow']), consequent=accelCmd['accel']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['far'] & roadCurve['small'] & currentSpeed['medium']), consequent=accelCmd['cruise']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['far'] & roadCurve['small'] & currentSpeed['fast']), consequent=accelCmd['cruise']))
    
    rules.append(ctrl.Rule(antecedent=(frontSensor['far'] & roadCurve['big'] & currentSpeed['slow']), consequent=accelCmd['accel']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['far'] & roadCurve['big'] & currentSpeed['medium']), consequent=accelCmd['noGas']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['far'] & roadCurve['big'] & currentSpeed['fast']), consequent=accelCmd['maxBrake']))
    
    
    rules.append(ctrl.Rule(antecedent=(frontSensor['close'] & roadCurve['none'] & currentSpeed['slow']), consequent=accelCmd['cruise']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['close'] & roadCurve['none'] & currentSpeed['medium']), consequent=accelCmd['maxBrake']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['close'] & roadCurve['none'] & currentSpeed['fast']), consequent=accelCmd['maxBrake']))
    
    rules.append(ctrl.Rule(antecedent=(frontSensor['close'] & roadCurve['small'] & currentSpeed['slow']), consequent=accelCmd['cruise']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['close'] & roadCurve['small'] & currentSpeed['medium']), consequent=accelCmd['noGas']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['close'] & roadCurve['small'] & currentSpeed['fast']), consequent=accelCmd['maxBrake']))
    
    rules.append(ctrl.Rule(antecedent=(frontSensor['close'] & roadCurve['big'] & currentSpeed['slow']), consequent=accelCmd['accel']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['close'] & roadCurve['big'] & currentSpeed['medium']), consequent=accelCmd['noGas']))
    rules.append(ctrl.Rule(antecedent=(frontSensor['close'] & roadCurve['big'] & currentSpeed['fast']), consequent=accelCmd['maxBrake']))
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