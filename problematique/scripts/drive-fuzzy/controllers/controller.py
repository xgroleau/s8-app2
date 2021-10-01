import numpy as np
from controllers.gear_controller.simple_gear_controller import SimpleGearController
from controllers.speed_controllers.simple_speed_controller import SimpleSpeedController
from controllers.steering_controllers.simple_steering_controller import SimpleSteeringController

class Controller(object):

    def __init__(self, 
                 gearController=SimpleGearController(), 
                 speedController=SimpleSpeedController(), 
                 steeringController=SimpleSteeringController()):
        self.gearController = gearController
        self.speedController = speedController
        self.steeringController = steeringController

    def computeAction(self, state):
        accel, brake = self.speedController.calculateAcceleration(state)
        gear = self.gearController.calculateGear(state)
        steer = self.steeringController.calculateSteering(state)

        action = {'accel': np.array([accel], dtype=np.float32),
                  'brake': np.array([brake], dtype=np.float32),
                  'gear': np.array([gear], dtype=np.int32),
                  'steer': np.array([steer], dtype=np.float32)}
        return action

