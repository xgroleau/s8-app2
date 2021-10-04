import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from skfuzzy import control as ctrl

class FuzzyGearController:
    def __init__(self):
        self.sim = createFuzzyController()
    
    def calculateGear(self, state):
        curGear = state['gear'][0]
        curRpm = state['rpm'][0]
        
        self.sim.input['rpm'] = curRpm
        self.sim.compute()
        gearCmd = self.sim.output['gearCmd']
        
        if curGear < 1:
            nextGear = 1
            
        elif curGear < 6 and gearCmd >= 0.5:
            nextGear = curGear + 1
        
        elif curGear > 1 and gearCmd <= -0.5:
            nextGear = curGear - 1
            
        else:
            nextGear = curGear

        return nextGear
    
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
    rpm = ctrl.Antecedent(np.linspace(0, 10000, 1000), 'rpm')
    
    # Outputs
    gearCmd = ctrl.Consequent(np.linspace(-1, 1, 1000), 'gearCmd', defuzzify_method='centroid')
    
    # Accumulation methods
    gearCmd.accumulation_method = np.fmax
    
    # Membership Functions
    rpm['under'] = fuzz.trapmf(rpm.universe, [0, 0, 2500, 3500])
    rpm['peak'] = fuzz.trapmf(rpm.universe, [2500, 3500, 5000, 6500])
    rpm['over'] = fuzz.trapmf(rpm.universe, [5000, 6500, 10000, 10000])
    
    gearCmd['downShift'] = singletonmf(gearCmd.universe, -1)
    gearCmd['noChange'] = singletonmf(gearCmd.universe, 0)
    gearCmd['upShift'] = singletonmf(gearCmd.universe, 1)

    
    # Rules
    rules = []
    rules.append(ctrl.Rule(antecedent=(rpm['under']), consequent=gearCmd['downShift']))
    rules.append(ctrl.Rule(antecedent=(rpm['peak']), consequent=gearCmd['noChange']))
    rules.append(ctrl.Rule(antecedent=(rpm['over']), consequent=gearCmd['upShift']))
    
    for rule in rules:
        rule.and_func = np.fmin
        rule.or_func = np.fmax

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)

    for var in sim.ctrl.fuzzy_variables:
        var.view()
    plt.show()
    
    return sim

createFuzzyController()