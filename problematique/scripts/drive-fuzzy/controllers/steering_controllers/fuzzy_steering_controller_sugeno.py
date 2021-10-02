import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from skfuzzy import control as ctrl

class FuzzySteeringControllerSugeno:
    def __init__(self):
        self.sim = createFuzzyController()
        
    def calculateSteering(self, state):
        self.sim.input['angle'] = state['angle'][0]
        self.sim.input['trackPos'] = state['trackPos'][0]
        
        self.sim.compute()
        
        return self.sim.output['steerCmd']
    
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
    angle = ctrl.Antecedent(np.linspace(-np.pi, np.pi, 1000), 'angle')
    trackPos = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'trackPos')
    
    # Outputs
    steerCmd = ctrl.Consequent(np.linspace(-1, 1, 1000), 'steerCmd', defuzzify_method='centroid')
    
    # Accumulation methods
    steerCmd.accumulation_method = np.fmax
    
    # Membership Functions
    angle['right'] = fuzz.trapmf(angle.universe, [-np.pi, -np.pi, -np.pi/8, 0])
    angle['straight'] = fuzz.trapmf(angle.universe, [-np.pi/32, -np.pi/64, np.pi/64, np.pi/32])
    angle['left'] = fuzz.trapmf(angle.universe, [0, np.pi/8, np.pi, np.pi])
    
    trackPos['left'] = fuzz.trapmf(trackPos.universe, [-1, -1, -0.5, -0.05])
    trackPos['center'] = fuzz.trapmf(trackPos.universe, [-0.2, -0.05, 0.05, 0.2])
    trackPos['right'] = fuzz.trapmf(trackPos.universe, [0.05, 0.5, 1, 1])
    
    steerCmd['hardLeft'] = singletonmf(steerCmd.universe, -1)
    steerCmd['left'] = singletonmf(steerCmd.universe, -0.4)
    steerCmd['straight'] = singletonmf(steerCmd.universe, 0)
    steerCmd['right'] = singletonmf(steerCmd.universe, 0.4)
    steerCmd['hardRight'] = singletonmf(steerCmd.universe, 1)
    
    # Rules
    rules = []
    rules.append(ctrl.Rule(antecedent=(angle['right'] & trackPos['left']), consequent=steerCmd['straight']))
    rules.append(ctrl.Rule(antecedent=(angle['right'] & trackPos['center']), consequent=steerCmd['left']))
    rules.append(ctrl.Rule(antecedent=(angle['right'] & trackPos['right']), consequent=steerCmd['hardLeft']))
    
    rules.append(ctrl.Rule(antecedent=(angle['straight'] & trackPos['left']), consequent=steerCmd['right']))
    rules.append(ctrl.Rule(antecedent=(angle['straight'] & trackPos['center']), consequent=steerCmd['straight']))
    rules.append(ctrl.Rule(antecedent=(angle['straight'] & trackPos['right']), consequent=steerCmd['left']))
    
    rules.append(ctrl.Rule(antecedent=(angle['left'] & trackPos['left']), consequent=steerCmd['hardRight']))
    rules.append(ctrl.Rule(antecedent=(angle['left'] & trackPos['center']), consequent=steerCmd['right']))
    rules.append(ctrl.Rule(antecedent=(angle['left'] & trackPos['right']), consequent=steerCmd['straight']))
    
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