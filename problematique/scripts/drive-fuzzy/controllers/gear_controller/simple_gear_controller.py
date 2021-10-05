# Author: Xavier Groleau <xavier.groleau@@usherbrooke.ca>
# Author: Charles Quesnel <charles.quesnel@@usherbrooke.ca>
# Author: Michael Samson <michael.samson@@usherbrooke.ca>
# Université de Sherbrooke, APP2 S8GIA, A2018

# Taken from drive-simple
class SimpleGearController:
    def __init__(self):
        pass
    
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
    def calculateGear(self, state):

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