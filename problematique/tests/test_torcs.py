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
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

import sys
import os
import logging
import tempfile
import unittest
import matplotlib.pyplot as plt

sys.path.append('../.')
from torcs.control.core import TorcsControlEnv, Episode, EpisodeRecorder
from torcs.optim.core import TorcsOptimizationEnv

CDIR = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_DIR = os.path.join(CDIR, 'data')

logger = logging.getLogger(__name__)


class TestTorcsControlEnv(unittest.TestCase):

    def testStep(self):
        with TorcsControlEnv(render=False) as env:
            for _ in range(4):
                env.reset()

                action = env.action_space.sample()
                for _ in range(100):
                    # Execute the action
                    observation, _, done, _ = env.step(action)
                    assert isinstance(observation, dict)
                    assert not done

                    # Select the next action based on the observation
                    action = env.action_space.sample()

    def testStepSingleTrack(self):
        with TorcsControlEnv(render=False, track='g-track-3') as env:
            for _ in range(4):
                env.reset()

                action = env.action_space.sample()
                for _ in range(100):
                    # Execute the action
                    observation, _, done, _ = env.step(action)
                    assert isinstance(observation, dict)
                    assert not done

                    # Select the next action based on the observation
                    action = env.action_space.sample()


class TestEpisodeRecorder(unittest.TestCase):

    def testSave(self):
        with TorcsControlEnv(render=False) as env:
            observation = env.reset()

            nbTotalStates = 1024
            _, filename = tempfile.mkstemp()
            try:
                with EpisodeRecorder(filename, syncEvery=63) as recorder:
                    for _ in range(nbTotalStates):

                        # Select the next action based on the observation
                        action = env.action_space.sample()
                        recorder.save(observation, action)

                        # Execute the action
                        observation, _, done, _ = env.step(action)
                        if done:
                            break

                episode = EpisodeRecorder.restore(filename)
                self.assertTrue(len(episode.states) == nbTotalStates)
            finally:
                os.remove(filename)


class TestEpisode(unittest.TestCase):

    def testVisualize(self):
        recordingFilename = os.path.join(TEST_DATA_DIR, 'track.pklz')
        episode = EpisodeRecorder.restore(recordingFilename)
        episode.visualize(showObservations=True, showActions=True)
        plt.pause(1.0)
        plt.close('all')


class TestTorcsOptimizationEnv(unittest.TestCase):

    def testStep(self):
        maxEvaluationTime = 40.0  # sec
        with TorcsOptimizationEnv(maxEvaluationTime) as env:

            # Loop a few times for demonstration purpose
            for i in range(15):

                parameters = env.action_space.sample()

                # Perform the evaluation with the simulator
                observation, _, _, _ = env.step(parameters)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
