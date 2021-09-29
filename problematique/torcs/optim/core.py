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

import os
import gym
import numpy as np
import logging
import select
import socket
import time
import tempfile
import subprocess
import threading

from threading import Thread
from gym import spaces
from gym.utils import seeding


logger = logging.getLogger(__name__)


def argsToString(args):
    argStr = ""
    for i, arg in enumerate(args):
        if arg == "*":
            arg = "\'%s\'" % (arg)
        if i == 0:
            argStr = arg
        else:
            argStr += " " + arg
    return argStr


def str2observation(msg):
    """
    Parse a state message from string and convert it to a structure.
    Input:
    - MSG, the state message string to parse. The input format is as follow:
        result -1 79.1908 3168.86 0 0.0228271
    Output:
    - STATE, the state structure with defined variables and values.
    """
    observation = dict()
    elems = msg.split(' ')[1:]
    names = ['bestlap', 'topspeed', 'distRaced', 'damage', 'fuelUsed']
    for i, name in enumerate(names):
        observation[name] = np.fromstring(elems[i], dtype=np.float, sep=' ')
    return observation


class ProcessMonitor(Thread):

    def __init__(self, process, timeout=0.1):
        super(ProcessMonitor, self).__init__()
        self.process = process
        self.timeout = timeout
        self._stopEvent = threading.Event()

    def stop(self):
        self._stopEvent.set()

    def run(self):

        while self.process.poll() is None or not self._stopEvent.is_set():
            # Log the output of STDOUT
            while True:
                try:
                    if self.process.stdout:
                        ready = select.select(
                            [self.process.stdout], [], [], self.timeout)[0]
                        if ready:
                            output = self.process.stdout.readline().decode('utf-8').strip()
                            if output:
                                logger.log(logging.DEBUG, output)
                            else:
                                break
                except OSError as e:
                    # [Errno 9] Bad file descriptor
                    logger.log(logging.DEBUG,
                               'Error occured in process monitor: ' + str(e))
                except ValueError as e:
                    # I/O operation on closed file
                    logger.log(logging.DEBUG,
                               'Error occured in process monitor: ' + str(e))

                if self._stopEvent.is_set():
                    break

            # Log the output of STDERR
            while True:
                try:
                    if self.process.stderr:
                        ready = select.select(
                            [self.process.stderr], [], [], self.timeout)[0]
                        if ready:
                            output = self.process.stderr.readline().decode('utf-8').strip()
                            if output:
                                if output != 'No stack.':
                                    logger.log(logging.ERROR, output)
                            else:
                                break
                except OSError as e:
                    # [Errno 9] Bad file descriptor
                    logger.log(logging.DEBUG,
                               'Error occured in process monitor: ' + str(e))
                except ValueError as e:
                    # I/O operation on closed file
                    logger.log(
                        logging.DEBUG, 'I/O operation error occured in process monitor: ' + str(e))

                if self._stopEvent.is_set():
                    break


class TorcsException(Exception):
    pass


class TorcsOptimizationEnv(gym.Env):
    """
    Description:
        Optimize the parameters of a car in the TORCS 3D car racing game

    Observation:
        Type: Box(1)
        Num    Observation
        0    bestlap
        The time elapsed during current lap. [seconds]

        Type: Box(1)
        Num    Observation
        0    distRaced
        The distance covered by the car from the beginning of the race. [meters]

        Type: Box(1)
        Num    Observation
        0    fuel
        The volume of fuel comsumed. [liters]

        Type: Box(1)
        Num    Observation
        0    maximum speed of the car along the longitudinal axis (X-axis)
        The maximum reached speed of the car. [km/h]

    Actions:
        Type: Box(1)
        Num    Observation                 Min         Max
        0    gear ratio                   0.1          5.0
        The 2nd gear ratio.

        Type: Box(1)
        Num    Observation                 Min         Max
        0    gear ratio                   0.1          5.0
        The 3th gear ratio.

        Type: Box(1)
        Num    Observation                 Min         Max
        0    gear ratio                   0.1          5.0
        The 4th gear ratio.

        Type: Box(1)
        Num    Observation                 Min         Max
        0    gear ratio                   0.1          5.0
        The 5th gear ratio.

        Type: Box(1)
        Num    Observation                 Min         Max
        0    gear ratio                   0.1          5.0
        The 6th gear ratio.

        Type: Box(1)
        Num    Observation                 Min         Max
        0    gear ratio                   1.0          10.0
        The gear ratio of the rear differential.

        Type: Box(1)
        Num    Observation                 Min         Max
        0       angle                      0.0         90.0
        The angle of the rear spoiler. [deg]

        Type: Box(1)
        Num    Observation                 Min         Max
        0       angle                      0.0         90.0
        The angle of the front spoiler. [deg]

    Reward:
        None

    Starting State:
        None

    Episode Termination:
        None
    """

    host = 'localhost'
    port = 3001
    buffer_size = 10000
    recv_timeout = 5.0  # sec

    availableTracks = ['inf-circle']

    availableDisplayModes = ['results only']
    availableModules = ['optserver']

    nbParameters = 8
    timeStepSimulation = 0.04  # sec

    def __init__(self, maxEvaluationTime=40.0):
        self.__dict__.update(maxEvaluationTime=maxEvaluationTime)

        self.action_space = spaces.Dict({'gear-2-ratio': spaces.Box(low=0.1, high=5.0, shape=(1,), dtype=np.float),
                                         'gear-3-ratio': spaces.Box(low=0.1, high=5.0, shape=(1,), dtype=np.float),
                                         'gear-4-ratio': spaces.Box(low=0.1, high=5.0, shape=(1,), dtype=np.float),
                                         'gear-5-ratio': spaces.Box(low=0.1, high=5.0, shape=(1,), dtype=np.float),
                                         'gear-6-ratio': spaces.Box(low=0.1, high=5.0, shape=(1,), dtype=np.float),
                                         'rear-differential-ratio': spaces.Box(low=1.0, high=10.0, shape=(1,), dtype=np.float),
                                         'rear-spoiler-angle': spaces.Box(low=0.0, high=90.0, shape=(1,), dtype=np.float),
                                         'front-spoiler-angle': spaces.Box(low=0.0, high=90.0, shape=(1,), dtype=np.float)})

        self.observation_space = spaces.Dict({'topspeed': spaces.Box(np.array([0.0]), np.array([np.finfo(np.float).max]), dtype=np.float),
                                              'distRaced': spaces.Box(np.array([0.0]), np.array([np.finfo(np.float).max]), dtype=np.float),
                                              'fuelUsed': spaces.Box(np.array([0.0]), np.array([np.finfo(np.float).max]), dtype=np.float)})

        self.client = None
        self.torcsServerProcess = None
        self.torcsServerMonitor = None
        self.torcsServerStatus = None
        self.torcsServerConfig = None

        self.seed()
        self.reset()

    def _generateConfig(self, track='inf-circle', displayMode='results only', module='optserver'):

        if track not in self.availableTracks:
            raise Exception('Unsupported track: %s' % (track))

        if displayMode not in self.availableDisplayModes:
            raise Exception('Unsupported display mode: %s' % (displayMode))

        if module not in self.availableModules:
            raise Exception('Unsupported module: %s' % (module))

        config = """<?xml version="1.0" encoding="UTF-8"?>
                    <!DOCTYPE params SYSTEM "params.dtd">

                    <params name="Quick Race">
                      <section name="Header">
                        <attstr name="name" val="Quick Race"/>
                        <attstr name="description" val="Quick Race"/>
                        <attnum name="priority" val="10"/>
                        <attstr name="menu image" val="data/img/splash-qr.png"/>
                      </section>

                      <section name="Tracks">
                        <attnum name="maximum number" val="1"/>
                        <section name="1">
                          <attstr name="name" val="%s"/>
                          <attstr name="category" val="oval"/>
                        </section>

                      </section>

                      <section name="Races">
                        <section name="1">
                          <attstr name="name" val="Quick Race"/>
                        </section>

                      </section>

                      <section name="Quick Race">
                        <attnum name="distance" unit="km" val="0"/>
                        <attstr name="type" val="race"/>
                        <attstr name="starting order" val="drivers list"/>
                        <attstr name="restart" val="yes"/>
                        <attnum name="laps" val="999999999999"/>
                        <attstr name="display mode" val="%s"/>
                        <section name="Starting Grid">
                          <attnum name="rows" val="2"/>
                          <attnum name="distance to start" val="25"/>
                          <attnum name="distance between columns" val="20"/>
                          <attnum name="offset within a column" val="10"/>
                          <attnum name="initial speed" val="0"/>
                          <attnum name="initial height" val="1"/>
                        </section>

                      </section>

                      <section name="Drivers">
                        <attnum name="maximum number" val="20"/>
                        <attnum name="focused idx" val="0"/>
                        <attstr name="focused module" val="%s"/>

                        <section name="1">
                          <attnum name="idx" val="0"/>
                          <attstr name="module" val="%s"/>
                        </section>
                      </section>

                      <section name="Configuration">
                        <attnum name="current configuration" val="4"/>
                        <section name="1">
                          <attstr name="type" val="track select"/>
                        </section>

                        <section name="2">
                          <attstr name="type" val="drivers select"/>
                        </section>

                        <section name="3">
                          <attstr name="type" val="race config"/>
                          <attstr name="race" val="Quick Race"/>
                          <section name="Options">
                            <section name="1">
                              <attstr name="type" val="race length"/>
                            </section>

                            <section name="2">
                              <attstr name="type" val="display mode"/>
                            </section>
                          </section>
                        </section>
                      </section>
                    </params>
        """ % (track, displayMode, module, module)
        return config

    def _startTorcsSimulatorThread(self):

        # 3D graphic display disabled
        configStr = self._generateConfig(track='inf-circle', displayMode='results only', module='optserver')
        command = ['torcs', '-s', '-nofuel', '-nogui', '-nodamage', '-nolaptime']

        # Write configuration to temporary file
        fd, configPath = tempfile.mkstemp(suffix='.xml')
        with os.fdopen(fd, 'w') as f:
            f.write(configStr)
        command += ['-r', configPath]
        self.torcsServerConfig = configPath

        logger.debug('Launching TORCS server: ' + argsToString(command))
        self.torcsServerProcess = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.torcsServerMonitor = ProcessMonitor(self.torcsServerProcess)
        self.torcsServerMonitor.daemon = True
        self.torcsServerMonitor.start()

        self._connectToTorcsServer()

    def _checkTorcsServer(self):
        """
        Check if the TORCS simulator is running.
        """
        isRunning = False
        if self.torcsServerProcess is not None:
            if self.torcsServerProcess.poll() is None:
                isRunning = True
        return isRunning

    def _stopTorcsServer(self):
        """
        Stop the TORCS simulator if it is running. This will kill the the simulator subprocess.
        """
        logger.debug('Stopping TORCS server')

        if self.client is not None:
            self.client.close()
            self.client = None

        if self.torcsServerConfig is not None:
            if os.path.exists(self.torcsServerConfig):
                os.remove(self.torcsServerConfig)

        if self.torcsServerProcess is not None:
            self.torcsServerProcess.kill()
            self.torcsServerProcess.wait()
            if self.torcsServerProcess.stdout:
                self.torcsServerProcess.stdout.close()
            if self.torcsServerProcess.stderr:
                self.torcsServerProcess.stderr.close()

            # NOTE: may be overkill
            with open(os.devnull, 'wb') as devnull:
                subprocess.call(['killall', 'torcs-bin'], stdout=devnull, stderr=devnull)

            self.torcsServerProcess = None

        if self.torcsServerMonitor is not None:
            self.torcsServerMonitor.stop()
            self.torcsServerMonitor.join()
            self.torcsServerMonitor = None

    def _evaluateParameters(self, parameters):
        """
        Send a set of parameters to be evaluated by the TORCS simulator.
        """

        # Send evaluation parameters to optimisation server
        names = ['gear-2-ratio', 'gear-3-ratio', 'gear-4-ratio', 'gear-5-ratio', 'gear-6-ratio',
                 'rear-differential-ratio', 'rear-spoiler-angle', 'front-spoiler-angle']
        msg = 'eval ' + str(int(self.maxEvaluationTime / self.timeStepSimulation))
        for name in names:
            # NOTE: rescale values to the interval [0, 1]
            minValue = self.action_space.spaces[name].low
            maxValue = self.action_space.spaces[name].high
            normValue = (parameters[name] - minValue) / (maxValue - minValue)
            msg += ' ' + str(normValue[0])

        try:
            logger.debug('Sending parameters to server: ' + msg)
            self.client.sendall(msg.encode(encoding='ascii'))
        except OSError as e:
            raise TorcsException(
                'Parameters failed to be sent to server: ' + str(e))

    def _connectToTorcsServer(self):
        try:
            # Open an UDP socket for communication with server
            self.client = socket.socket(
                family=socket.AF_INET, type=socket.SOCK_DGRAM)
            self.client.connect((self.host, self.port))
            self.client.setblocking(0)
            logger.debug('Socket connected to TORCS server')

            # Loop while no connection has been established
            connected = False
            while not connected:
                if not self._checkTorcsServer():
                    raise TorcsException('Simulator closed unexpectedly')

                # Connect and send the identifier
                try:
                    logger.debug('Requesting information from server')
                    self.client.sendall('info?'.encode(encoding='ascii'))
                except OSError as e:
                    if 'Connection refused' not in str(e):
                        raise TorcsException(
                            'Identifier failed to be sent to server: ' + str(e))

                # Read message from server
                msg = None
                try:
                    ready = select.select(
                        [self.client], [], [], self.recv_timeout)
                    if ready[0]:
                        msg = self.client.recv(
                            self.buffer_size).decode('utf-8').strip()
                        if len(msg) == 0:
                            raise TorcsException(
                                'Socket connection broken with TORCS server.')
                        else:
                            logger.debug(
                                'Received message from TORCS server: ' + msg)

                except OSError as e:
                    if 'Connection refused' not in str(e):
                        raise TorcsException(
                            'Failed to receive message from server: ' + str(e))

                # Parse message
                if msg is None:
                    logger.info('Waiting for simulator to initialize.')
                    time.sleep(2.0)
                elif 'info' in msg:
                    logger.info('Connected to TORCS server.')
                    connected = True
                else:
                    raise TorcsException(
                        'Unknown message received from server: ' + msg)

        except Exception as e:
            logger.error('Failed to connect to TORCS server: ' + str(e))
            if self.client:
                self.client.close()
            self._stopTorcsServer()

    def _getRawObservation(self):

        logger.debug('Waiting for simulation results.')

        # Read state data from server
        nbSuccessiveFailures = 0
        maxNbSuccessiveFailures = 4
        recvSuccess = False
        while not recvSuccess:
            if not self._checkTorcsServer():
                raise TorcsException('Simulator closed unexpectedly')

            # Read message from server
            msg = None
            try:
                ready = select.select(
                    [self.client], [], [], self.recv_timeout)[0]
                if ready:
                    msg = self.client.recv(
                        self.buffer_size).decode('utf-8').strip()
                    if len(msg) == 0:
                        raise TorcsException(
                            'Socket connection broken with TORCS server.')
                    else:
                        logger.debug(
                            'Received message from TORCS server: ' + msg)
                        recvSuccess = True
                else:
                    logger.warn(
                        'Timeout waiting to receive message from server.')

            except OSError as e:
                nbSuccessiveFailures += 1
                if nbSuccessiveFailures >= maxNbSuccessiveFailures:
                    raise TorcsException(
                        'Failed to receive message from server: ' + str(e))

        # Parse message
        observation = None
        if msg is None:
            pass
        elif '***shutdown***' in msg:
            logger.debug('Client shutdown from server.')
            self.torcsServerStatus = 'shutdown'
        elif '***restart***' in msg:
            logger.debug('Client restart from server.')
            self.torcsServerStatus = 'restart'
        elif 'time-over' in msg:
            raise TorcsException('Time-over response received from server')
        else:
            # Parse state data from message
            logger.debug('Results received: ' + msg)

            observation = str2observation(msg)
            self.torcsServerStatus = 'running'

        return observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        self._evaluateParameters(action)

        reward = 0.0
        done = False
        rawObservation = self._getRawObservation()

        observation = None
        if rawObservation is not None:
            # Remove unnecessary attributes from the raw observation
            observation = dict()
            for name in list(self.observation_space.spaces.keys()):
                observation[name] = rawObservation[name]

        return observation, reward, done, {}

    def reset(self):
        # Restart TORCS server
        self._stopTorcsServer()
        self._startTorcsSimulatorThread()

        return None

    def close(self):
        self._stopTorcsServer()
        super(TorcsOptimizationEnv, self).close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self, *args):
        self.close()