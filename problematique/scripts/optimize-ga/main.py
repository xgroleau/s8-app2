# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA,
# OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

import os
import sys
import time
import numpy as np
import logging

sys.path.append('../..')
from torcs.optim.core import TorcsOptimizationEnv, TorcsException

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


################################
# Define helper functions here
################################

def main():

    try:
        maxEvaluationTime = 40.0  # sec
        with TorcsOptimizationEnv(maxEvaluationTime) as env:

            # Loop a few times for demonstration purpose
            for i in range(15):

                # Uncomment to use the default values in the TORCS simulator
                parameters = {'gear-2-ratio': np.array([2.5]),
                              'gear-3-ratio': np.array([1.5]),
                              'gear-4-ratio': np.array([1.5]),
                              'gear-5-ratio': np.array([1.5]),
                              'gear-6-ratio': np.array([1.0]),
                              'rear-differential-ratio': np.array([4.5]),
                              'rear-spoiler-angle': np.array([14.0]),
                              'front-spoiler-angle': np.array([6.0])}

                parameters =   {'gear-2-ratio': np.array([1.7]), 
                                'gear-3-ratio': np.array([1.6]), 
                                'gear-4-ratio': np.array([2.8]), 
                                'gear-5-ratio': np.array([0.1]), 
                                'gear-6-ratio': np.array([0.3]), 
                                'rear-differential-ratio': np.array([8.6]), 
                                'rear-spoiler-angle': np.array([50.1]), 
                                'front-spoiler-angle': np.array([1.5])}


                # Uncomment to generate random values in the proper range for each variable
                # parameters = env.action_space.sample()

                # Generate a random vector of parameters in the proper interval
                logger.info('Generated new parameter vector: ' + str(parameters))

                # Perform the evaluation with the simulator
                observation, _, _, _ = env.step(parameters)
 
                # Display simulation results
                logger.info('##################################################')
                logger.info('Results:')
                logger.info('Time elapsed (sec) =   %f', maxEvaluationTime)
                logger.info('Top speed (km/h)   =   %f', observation['topspeed'][0])
                logger.info('Distance raced (m) =   %f', observation['distRaced'][0])
                logger.info('Fuel used (l)      =   %f', observation['fuelUsed'][0])
                logger.info('##################################################')

    except TorcsException as e:
        logger.error('Error occured communicating with TORCS server: ' + str(e))

    except KeyboardInterrupt:
        pass

    logger.info('All done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
