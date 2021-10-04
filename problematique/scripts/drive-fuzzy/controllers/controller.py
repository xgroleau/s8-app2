import numpy as np
from controllers.gear_controller.simple_gear_controller import SimpleGearController
from controllers.speed_controllers.simple_speed_controller import SimpleSpeedController
from controllers.steering_controllers.simple_steering_controller import SimpleSteeringController

# Class to wrap the three seperate controllers for each command
class Controller(object):
    # The constructor specifies which indivual controller we want to use. By default, it uses the one from drive-simple
    def __init__(self, 
                 gearController=SimpleGearController(), 
                 speedController=SimpleSpeedController(), 
                 steeringController=SimpleSteeringController()):
        self.gearController = gearController
        self.speedController = speedController
        self.steeringController = steeringController

    # Compute the action using each controller which does it's own computation
    def computeAction(self, state):
        accel, brake = self.speedController.calculateAcceleration(state)
        gear = self.gearController.calculateGear(state)
        steer = self.steeringController.calculateSteering(state)

        action = {'accel': np.array([accel], dtype=np.float32),
                  'brake': np.array([brake], dtype=np.float32),
                  'gear': np.array([gear], dtype=np.int32),
                  'steer': np.array([steer], dtype=np.float32)}
        return action

