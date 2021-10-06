# Author: Xavier Groleau <xavier.groleau@@usherbrooke.ca>
# Author: Charles Quesnel <charles.quesnel@@usherbrooke.ca>
# Author: Michael Samson <michael.samson@@usherbrooke.ca>
# Université de Sherbrooke, APP2 S8GIA, A2018

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from skfuzzy import control as ctrl

class FuzzySteeringControllerMamdani:
    def __init__(self):
        self.sim = createFuzzyController()
        
    def view(self):
        for var in self.sim.ctrl.fuzzy_variables:
            var.view()
            plt.show()
        
    def calculateSteering(self, state):
        # Set fuzzy variables
        self.sim.input['angle'] = state['angle'][0]
        self.sim.input['trackPos'] = state['trackPos'][0]
        
        # Compute output
        self.sim.compute()
        
        # Return command
        return self.sim.output['steerCmd']

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
    
    steerCmd['hardLeft'] = fuzz.trapmf(steerCmd.universe, [-1, -1, -0.75, -0.5])
    steerCmd['left'] = fuzz.trapmf(steerCmd.universe, [-0.75, -0.5, -0.25, -0.01])
    steerCmd['straight'] = fuzz.trapmf(steerCmd.universe, [-0.1, -0.05, 0.05, 0.1])
    steerCmd['right'] = fuzz.trapmf(steerCmd.universe, [0.01, 0.25, 0.5, 0.75])
    steerCmd['hardRight'] = fuzz.trapmf(steerCmd.universe, [0.5, 0.75, 1, 1])
    
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

    # Create control system
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    
    return sim