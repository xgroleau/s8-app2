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
import logging

import numpy as np
from sklearn.model_selection import train_test_split

from models import basic_fully_connected

sys.path.append('../..')
from torcs.control.core import TorcsControlEnv, TorcsException, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


################################
# Define helper functions here
################################

def main():

    # Training model
    files = os.path.join(CDIR, 'data')
    episodes = [EpisodeRecorder.restore(f) for f in files]

    # Input
    angle = np.array([e.angle for e in episodes]).flatten()
    speed = np.array([e.speed for e in episodes]).flatten()
    trackPos = np.array([e.trackPos for e in episodes]).flatten()
    gear = np.array([e.gear for e in episodes]).flatten()
    rpm = np.array([e.rpm for e in episodes]).flatten()

    # Output
    accelCmd = np.array([e.accelCmd for e in episodes]).flatten()
    brakeCmd = np.array([e.brakeCmd for e in episodes]).flatten()
    gearCmd = np.array([e.gearCmd for e in episodes]).flatten()
    steerCmd = np.array([e.steerCmd for e in episodes]).flatten()

    x = np.vstack((angle, speed, trackPos, gear, rpm))
    y = np.vstack((accelCmd, brakeCmd, gearCmd, steerCmd))

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.15)
    model = basic_fully_connected.create_model()

    model.fit(x_train, y_train, batch_size=300, epochs=500, shuffle=False, verbose=1)
    exit(0)
    
    try:
        with TorcsControlEnv(render=False) as env:

            nbTracks = len(TorcsControlEnv.availableTracks)
            nbSuccessfulEpisodes = 0
            for episode in range(nbTracks):
                logger.info('Episode no.%d (out of %d)' % (episode + 1, nbTracks))
                startTime = time.time()

                observation = env.reset()
                trackName = env.getTrackName()

                nbStepsShowStats = 1000
                curNbSteps = 0
                done = False
                with EpisodeRecorder(os.path.join(recordingsPath, 'track-%s.pklz' % (trackName))) as recorder:
                    while not done:
                        # TODO: Select the next action based on the observation
                        action = env.action_space.sample()
                        recorder.save(observation, action)
    
                        # Execute the action
                        observation, reward, done, _ = env.step(action)
                        curNbSteps += 1
    
                        if observation and curNbSteps % nbStepsShowStats == 0:
                            curLapTime = observation['curLapTime'][0]
                            distRaced = observation['distRaced'][0]
                            logger.info('Current lap time = %4.1f sec (distance raced = %0.1f m)' % (curLapTime, distRaced))
    
                        if done:
                            if reward > 0.0:
                                logger.info('Episode was successful.')
                                nbSuccessfulEpisodes += 1
                            else:
                                logger.info('Episode was a failure.')
    
                            elapsedTime = time.time() - startTime
                            logger.info('Episode completed in %0.1f sec (computation time).' % (elapsedTime))

            logger.info('-----------------------------------------------------------')
            logger.info('Total number of successful tracks: %d (out of %d)' % (nbSuccessfulEpisodes, nbTracks))
            logger.info('-----------------------------------------------------------')

    except TorcsException as e:
        logger.error('Error occured communicating with TORCS server: ' + str(e))

    except KeyboardInterrupt:
        pass

    logger.info('All done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
