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
import gzip
import pickle
import gym
import numpy as np
import logging
import select
import socket
import time
import six
import tempfile
import subprocess
import threading
import matplotlib.pyplot as plt

from threading import Thread
from gym import spaces
from gym.utils import seeding

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


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
        (angle -0.00396528)(curLapTime -0.962)(damage 0)(distFromStart 5759.1)(distRaced 0)(fuel 80)(gear 0)
    Output:
    - STATE, the state structure with defined variables and values.
    """
    observation = dict()
    for elem in msg.split('(')[1:]:
        subelems = elem.split(' ')
        name = subelems[0]
        values = ' '.join(subelems[1:])[:-1]
        observation[name] = np.fromstring(values, dtype=np.float, sep=' ')
    return observation


def action2str(action, meta=0):
    """
    Convert an action structure to a string message representation.

    Input:
    - ACTION, the action dictionary.
    - META, the meta-variable controlling the restart of the simulation. Set to 1 to restart the simulation, otherwise 0.

    Output:
    - MSG, the string message representation. The output format is as follow:
           (accel 0.1)(brake 0.962)(steer 0.2322)(gear 2)(meta 0)
    """
    output = StringIO()
    for name, value in six.iteritems(action):
        output.write('(%s %f)' % (name, value))
    output.write('(meta %d)' % (meta))
    msg = output.getvalue()
    output.close()
    return msg


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


class TorcsControlEnv(gym.Env):
    """
    Description:
        Control in the TORCS 3D car racing game

     - MODE, set to 'gui' to enable 3D graphic display (a window will appear),
       or to 'nogui' (default) for console mode. The console mode is significantly faster
       since no rendering is required.

    Observation:
        Type: Box(1)
        Num    Observation                 Min         Max
        0    angle                        -pi           pi
        The angle between the car direction and the direction of the track axis. [rad]

        Type: Box(1)
        Num    Observation
        0    curLapTime
        The time elapsed during current lap. [seconds]

        Type: Box(1)
        Num    Observation
        0    damage
        The current damage of the car (the higher is the value the higher is the damage). [points]

        Type: Box(1)
        Num    Observation
        0    distFromStart
        The distance of the car from the start line along the track line. [meters]

        Type: Box(1)
        Num    Observation
        0    distRaced
        The distance covered by the car from the beginning of the race. [meters]

        Type: Box(1)
        Num    Observation
        0    fuel
        The current fuel level. [liters]

        Type: Discrete(8)
        Num    Observation
        0      reverse gear
        1      neutral
        2      1st gear
        3      2nd gear
        4      3rd gear
        5      4th gear
        6      5th gear
        7      6th gear
        The gear value

        Type: Box(1)
        Num    Observation
        0    lastLapTime
        The time to complete the last lap. [seconds]

        Type: Box(1)
        Num    Observation                   Min         Max
        0    rpm                              0         10000
        The number of rotation per minute of the car engine. [rpm]

        Type: Box(2)
        Num    Observation
        0    speed of the car along the longitudinal axis (X-axis)
        1    speed of the car along the transverse axis (Y-axis)
        The speed of the car. [km/h]

        Type: Box(19)
        Num    Observation
        0-19   track
        A vector of 19 range finder sensors: each sensors represents the distance between the track
        edge and the car. Sensors are oriented every 10 degrees from -pi/2 and +pi/2 in front of the car.
        Distance are in meters within a range of 100 meters. When the car is outside of the
        track (i.e., pos is less than -1 or greater than 1), these values are not reliable! [meters]

        Type: Box(1)
        Num    Observation
        0    trackPos
        The distance between the car and the track axis. The value is normalized w.r.t to the track
        width: it is 0 when car is on the axis, -1 when the car is on the right edge of the track and +1
        when it is on the left edge of the car. Values greater than 1 or smaller than -1 means that the
        car is outside of the track.

        Type: Box(4)
        Num    Observation
        0-3    wheelSpinVel
        A vector of 4 sensors representing the rotation speed of wheels. [rad/s]

    Actions:
        Type: Box(1)
        Num    Action                     Min         Max
        0      accel                      0           1
        The virtual gas pedal (0 means no gas, 1 full gas).

        Type: Box(1)
        Num    Action                     Min         Max
        0      brake                      0           1
        The virtual brake pedal (0 means no brake, 1 full brake).

        Type: Discrete(8)
        Num    Action
        0      reverse gear
        1      neutral
        2      1st gear
        3      2nd gear
        4      3rd gear
        5      4th gear
        6      5th gear
        7      6th gear
        The gear value

        Type: Box(1)
        Num    Action                     Min         Max
        0      steer                      -1           1
        The steering value. -1 and +1 means respectively full left and right, that corresponds to an angle of 0.785398 rad.

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
    recv_timeout = 2.0  # sec

    availableTracks = ['aalborg', 'alpine-1', 'alpine-2', 'brondehach', 'corkscrew',
                       'e-track-1', 'e-track-2', 'e-track-3', 'e-track-4', 'e-track-6',
                       'eroad', 'forza', 'g-track-1', 'g-track-2', 'g-track-3', 'ole-road-1',
                       'ruudskogen', 'spring', 'street-1', 'wheel-1', 'wheel-2']

    availableDisplayModes = ['normal', 'results only']
    availableModules = ['wcci2008competition', 'wcci2008player', 'wcci2008bot']

    noop = {'accel': np.array([0.0], dtype=np.float),
            'brake': np.array([0.0], dtype=np.float),
            'gear': np.array([0], dtype=np.int),
            'steer': np.array([0.0], dtype=np.float)}

    maxTimeAllowedOutsideTrack = 10.0  # sec
    maxAccumulatedDamage = 100.0

    def __init__(self, render=False, track=None):

        if track is not None and track not in self.availableTracks:
            raise Exception('Unsupported track name: %s' % (track))

        self.__dict__.update(render=render, track=track)

        self.action_space = spaces.Dict({'accel': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float),
                                         'brake': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float),
                                         'gear': spaces.Box(low=-1, high=6, shape=(1,), dtype=np.int),
                                         'steer': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float)})

        self.observation_space = spaces.Dict({'angle': spaces.Box(np.array([-np.pi]), np.array([np.pi]), dtype=np.float),
                                              'curLapTime': spaces.Box(np.array([0.0]), np.array([np.finfo(np.float).max]), dtype=np.float),
                                              'damage': spaces.Box(np.array([0.0]), np.array([np.finfo(np.float).max]), dtype=np.float),
                                              'distFromStart': spaces.Box(np.array([0.0]), np.array([np.finfo(np.float).max]), dtype=np.float),
                                              'distRaced': spaces.Box(np.array([0.0]), np.array([np.finfo(np.float).max]), dtype=np.float),
                                              'fuel': spaces.Box(np.array([0.0]), np.array([np.finfo(np.float).max]), dtype=np.float),
                                              'gear': spaces.Box(low=-1, high=6, shape=(1,), dtype=np.int),
                                              'rpm': spaces.Box(np.array([0.0]), np.array([10000.0]), dtype=np.float),
                                              'speed': spaces.Box(np.array([0.0] * 2), np.array([np.finfo(np.float).max] * 2), dtype=np.float),
                                              'track': spaces.Box(np.array([0.0] * 19), np.array([100.0] * 19), dtype=np.float),
                                              'trackPos': spaces.Box(np.array([-1.0]), np.array([1.0]), dtype=np.float),
                                              'wheelSpinVel': spaces.Box(np.array([0.0] * 4), np.array([np.finfo(np.float).max] * 4), dtype=np.float)})

        self.client = None
        self.torcsServerProcess = None
        self.torcsServerMonitor = None
        self.torcsServerStatus = None
        self.torcsServerConfig = None
        self.curTrackIndex = None

        self.lastLapTimeOutsideTrack = 0.0

        self.seed()
        self.steps_beyond_done = None

    def _generateConfig(self, track, abort=False, displayMode='normal', module='wcci2008competition'):

        if abort:
            abortStr = 'yes'
        else:
            abortStr = 'no'

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
                          <attstr name="category" val="road"/>
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
                        <attstr name="restart" val="no"/>
                        <attstr name="abort" val="%s"/>
                        <attnum name="laps" val="2"/>
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
                        <attnum name="focused idx" val="6"/>
                        <attstr name="focused module" val=""/>
                        <section name="1">
                          <attnum name="idx" val="1"/>
                          <attstr name="module" val="%s"/>
                        </section>

                      </section>

                      <section name="Drivers Start List">
                        <section name="1">
                          <attstr name="module" val="%s"/>
                          <attnum name="idx" val="1"/>
                        </section>

                      </section>

                      <section name="Configuration">
                        <attnum name="current configuration" val="1"/>
                      </section>

                    </params>
        """ % (track, abortStr, displayMode, module, module)
        return config

    def _selectTrack(self):
        if self.track is not None:
            track = self.track
        else:
            if self.curTrackIndex is None:
                self.curTrackIndex = -1
            self.curTrackIndex += 1
            track = self.availableTracks[self.curTrackIndex]
        return track

    def getTrackName(self):
        if self.track is not None:
            track = self.track
        else:
            track = self.availableTracks[self.curTrackIndex]
        return track

    def _startTorcsSimulatorThread(self, track):

        if self.render:
            # 3D graphic display enabled
            configStr = self._generateConfig(
                track, abort=False, displayMode='normal', module='wcci2008competition')
            command = ['torcs', '-s', '-nodamage', '-nolaptime']
        else:
            # 3D graphic display disabled
            configStr = self._generateConfig(
                track, abort=True, displayMode='results only', module='wcci2008competition')
            command = ['torcs', '-s', '-nogui', '-nodamage', '-nolaptime']

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

    def _applyAction(self, action):
        """
        Send an action to be executed by the TORCS simulator.
        Call this function only after receiving the car state in blocking mode.
        """
        # Send action to server
        msg = action2str(action)
        try:
            logger.debug('Sending action to server: ' + msg)
            self.client.sendall(msg.encode(encoding='ascii'))
        except OSError as e:
            raise TorcsException(
                'Action failed to be sent to server: ' + str(e))

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
                    logger.debug('Sending identifier to server')
                    self.client.sendall('wcci2008'.encode(encoding='ascii'))
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
                elif '***identified***' in msg:
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

    def _getRawObservation(self, blocking=True):

        logger.debug('Waiting for car state data.')

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
        else:
            # Parse state data from message
            logger.debug('State received: ' + msg)

            rawObservation = str2observation(msg)
            observation = dict()
            for name in six.iterkeys(rawObservation):
                if 'gear' in name:
                    observation[name] = rawObservation[name].astype(np.int)
                elif name == 'speedX' or name == 'speedY':
                    observation['speed'] = np.abs(np.concatenate(
                        (rawObservation['speedX'], rawObservation['speedY'])).astype(np.float))
                elif name == 'distRaced':
                    observation[name] = np.clip(
                        rawObservation[name].astype(np.float), 0.0, np.inf)
                else:
                    observation[name] = rawObservation[name].astype(np.float)

            self.torcsServerStatus = 'running'

            # Send an acknowledge message to simulator if in non-blocking mode
            if not blocking:
                try:
                    logger.debug('Sending acknowledge to server')
                    self.client.sendall('ACK'.encode(encoding='ascii'))
                except OSError as e:
                    raise TorcsException(
                        'Acknowledge failed to be sent to server: ' + str(e))

        return observation

    def _getReward(self, rawObservation):
        reward = 0.0
        done = False
        if rawObservation is None:
            logger.info('TORCS server ended the simulation.')
            reward = 0.0
            done = True
        else:
            curTrackPos = rawObservation['trackPos'][0]
            curDamage = rawObservation['damage'][0]
            curLapTime = rawObservation['curLapTime'][0]
            lastLapTime = rawObservation['lastLapTime'][0]

            if lastLapTime != 0.0:
                # Success: lap completed
                logger.info('Lap completed.')
                reward = 1.0
                done = True
            elif curTrackPos < -1.0 or curTrackPos > 1.0:
                if self.lastLapTimeOutsideTrack == 0.0:
                    self.lastLapTimeOutsideTrack = curLapTime
                else:
                    elapsedLapTime = curLapTime - self.lastLapTimeOutsideTrack
                    if elapsedLapTime > self.maxTimeAllowedOutsideTrack:
                        logger.info('The car was outside the track for more than %d sec!' % (self.maxTimeAllowedOutsideTrack))
                        reward = 0.0
                        done = True
            elif curDamage > self.maxAccumulatedDamage:
                logger.info('The car has significant damage (%0.1f)!' % (curDamage))
                reward = 0.0
                done = True
            else:
                self.lastLapTimeOutsideTrack = 0.0

        return reward, done

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        self._applyAction(action)

        rawObservation = self._getRawObservation()
        reward, done = self._getReward(rawObservation)

        observation = None
        if rawObservation is not None:
            # Remove unnecessary attributes from the raw observation
            observation = dict()
            for name in list(self.observation_space.spaces.keys()):
                observation[name] = rawObservation[name]

        if not done:
            pass
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1

        return observation, reward, done, {}

    def reset(self):
        self.steps_beyond_done = None

        track = self._selectTrack()
        logger.info('Using track: %s' % (track))

        # Restart TORCS server
        self._stopTorcsServer()
        self._startTorcsSimulatorThread(track)

        # Let the simulation run until the time is positive
        curLapTime = -1.0
        eps = 1.0
        while not curLapTime >= eps:
            self._applyAction(self.noop)
            rawObservation = self._getRawObservation()
            if rawObservation is None:
                raise TorcsException('Unable to get initial observation from the TORCS server.')
            curLapTime = rawObservation['curLapTime'][0]

        # Remove unnecessary attributes from the raw observation
        observation = dict()
        for name in list(self.observation_space.spaces.keys()):
            observation[name] = rawObservation[name]

        return observation

    def close(self):
        self._stopTorcsServer()
        super(TorcsControlEnv, self).close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self, *args):
        self.close()


class TorcsHumanEnv(TorcsControlEnv):

    def __init__(self, track=None):
        # NOTE: force rendering
        super(TorcsHumanEnv, self).__init__(render=True, track=track)

    def _startTorcsSimulatorThread(self, track):

        # 3D graphic display enabled
        configStr = self._generateConfig(
            track, abort=False, displayMode='normal', module='wcci2008player')
        command = ['torcs', '-d', '-nodamage', '-nolaptime']

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

    def step(self):
        observation = None
        action = None
        rawObservation = self._getRawObservation(blocking=False)
        reward, done = self._getReward(rawObservation)

        if rawObservation is not None:
            # Get action-related attributes from the raw observation
            action = dict()
            for name in six.iterkeys(rawObservation):
                if name.endswith('Cmd'):
                    action[name.replace('Cmd', '')] = rawObservation[name]

            # Remove unnecessary attributes from the raw observation
            observation = dict()
            for name in list(self.observation_space.spaces.keys()):
                observation[name] = rawObservation[name]

        return observation, action, reward, done, {}


class TorcsBotEnv(TorcsControlEnv):

    def _startTorcsSimulatorThread(self, track):

        if self.render:
            # 3D graphic display enabled
            configStr = self._generateConfig(
                track, abort=False, displayMode='normal', module='wcci2008bot')
            command = ['torcs', '-d', '-nodamage', '-nolaptime']
        else:
            # 3D graphic display disabled
            configStr = self._generateConfig(
                track, abort=True, displayMode='results only', module='wcci2008bot')
            command = ['torcs', '-d', '-nogui',
                       '-nodamage', '-nolaptime']

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

    def step(self):
        observation = None
        action = None
        rawObservation = self._getRawObservation(blocking=False)
        reward, done = self._getReward(rawObservation)

        if rawObservation is not None:
            # Get action-related attributes from the raw observation
            action = dict()
            for name in six.iterkeys(rawObservation):
                if name.endswith('Cmd'):
                    action[name.replace('Cmd', '')] = rawObservation[name]

            # Remove unnecessary attributes from the raw observation
            observation = dict()
            for name in list(self.observation_space.spaces.keys()):
                observation[name] = rawObservation[name]

        return observation, action, reward, done, {}


class Episode(object):

    availableAttrs = ['angle', 'curLapTime', 'damage', 'distFromStart',
                      'distRaced', 'fuel', 'gear', 'lastLapTime', 'rpm',
                      'speed', 'track', 'trackPos', 'wheelSpinVel',
                      'accelCmd', 'brakeCmd', 'gearCmd', 'steerCmd']

    def __init__(self, states):
        self.states = states

    def getAbsoluteTime(self):
        curLapTime = self._getSequence('curLapTime')
        t = np.zeros(len(curLapTime))
        t[0] = curLapTime[0]
        for i in range(2, len(curLapTime)):
            dt = curLapTime[i] - curLapTime[i - 1]
            if dt < 0:
                # New lap
                dt = curLapTime[i]
            t[i] = t[i - 1] + dt
        return t

    def _getSequence(self, attr):
        values = []
        if len(self.states) > 0:
            values = [state[attr] for state in self.states]
            values = np.array(values)
        return values

    def __getattribute__(self, name):
        if name == 't':
            return self.getAbsoluteTime()
        elif name in Episode.availableAttrs:
            return self._getSequence(name)
        else:
            return object.__getattribute__(self, name)

    def getAttributes(self):
        attrs = []
        if len(self.states) > 0:
            attrs = list(six.iterkeys(self.states[0]))
        return attrs

    def visualize(self, showObservations=True, showActions=True):

        t = self.t

        if showActions:
            # Show car acceleration and brake commands
            plt.figure()
            plt.subplot(211)
            plt.plot(t, self.accelCmd * 100)
            plt.title('Car acceleration command')
            plt.xlabel('Time [sec]')
            plt.ylabel('Acceleration [%]')
            plt.axis('tight')
            plt.subplot(212)
            plt.plot(t, self.brakeCmd * 100)
            plt.title('Car brake command')
            plt.xlabel('Time [sec]')
            plt.ylabel('Brake [%]')
            plt.axis('tight')
            plt.tight_layout()

            # Show car steering and gear commands
            plt.figure()
            plt.subplot(211)
            plt.plot(t, self.steerCmd * 0.785398)
            plt.title('Car steering command')
            plt.xlabel('Time [sec]')
            plt.ylabel('Steering angle [rad]')
            plt.subplot(212)
            plt.plot(t, self.gearCmd)
            plt.title('Tranmission gear command')
            plt.xlabel('Time [sec]')
            plt.ylabel('Gear')
            plt.tight_layout()

        if showObservations:
            # Show car speed along longitudinal and transverse axes
            speed = self.speed
            plt.figure()
            plt.subplot(211)
            plt.plot(t, speed[:, 0])
            plt.title('Car speed (longitudinal axis, X)')
            plt.xlabel('Time [sec]')
            plt.ylabel('Speed [km/h]')
            plt.subplot(212)
            plt.plot(t, speed[:, 1])
            plt.title('Car speed (transverse axis, Y)')
            plt.xlabel('Time [sec]')
            plt.ylabel('Speed [km/h]')
            plt.tight_layout()

            # Show car speed along longitudinal axis, selected gear and engine rpm
            speed = self.speed
            plt.figure()
            plt.subplot(311)
            plt.plot(t, speed[:, 0])
            plt.title('Car speed (longitudinal axis, X)')
            plt.xlabel('Time [sec]')
            plt.ylabel('Speed [km/h]')
            plt.subplot(312)
            plt.plot(t, self.gear)
            plt.title('Tranmission gear')
            plt.xlabel('Time [sec]')
            plt.ylabel('Gear')
            plt.subplot(313)
            plt.plot(t, self.rpm)
            plt.title('Engine rpm')
            plt.xlabel('Time [sec]')
            plt.ylabel('RPM')
            plt.tight_layout()

            # Show wheel speed velocities
            plt.figure()
            plt.plot(t, self.wheelSpinVel)
            plt.title('Wheel speed velocity')
            plt.xlabel('Time [sec]')
            plt.ylabel('Angular speed [rad/s]')
            plt.legend(['front-right', 'front-left', 'rear-right', 'rear-left'])
            plt.tight_layout()

            # Show track position
            plt.figure()
            plt.plot(t, self.trackPos)
            plt.title('Track position')
            plt.xlabel('Time [sec]')
            plt.ylabel('Track position')
            plt.tight_layout()

            # Fuel consumption and distance raced
            # WARNING: fuel may have been disabled during the simulation!
            plt.figure()
            plt.subplot(211)
            plt.plot(t, self.distRaced)
            plt.title('Distance covered by the car')
            plt.xlabel('Time [sec]')
            plt.ylabel('Distance [m]')
            plt.subplot(212)
            plt.plot(t, np.max(self.fuel) - self.fuel)
            plt.title('Fuel consumed')
            plt.xlabel('Time [sec]')
            plt.ylabel('Fuel volume [l]')
            plt.xlim([np.min(t), np.max(t)])
            plt.tight_layout()

            # Car damage
            # WARNING: car damage may have been disabled during the simulation!
            plt.figure()
            plt.plot(t, self.damage)
            plt.title('Car damage')
            plt.xlabel('Time [sec]')
            plt.ylabel('Damage points')
            plt.xlim([np.min(t), np.max(t)])
            plt.tight_layout()

            # Track distance
            fig = plt.figure()
            for i in range(4):
                ax = fig.add_subplot(2, 2, i + 1, polar=True)

                # Sensor angles
                angles = np.array([-90, -80, -70, -60, -50, -40, -30, -20, -10,
                                   0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) * np.pi / 180

                # Pick a random state
                idx = np.random.randint(len(t))
                radii = self.track[idx, :]
                kw = dict(arrowstyle="->", color='k')
                [ax.annotate("", xy=(angle, radius), xytext=(0, 0),
                             arrowprops=kw) for
                 angle, radius in zip(angles, radii)]
                ax.set_ylim(0, np.max(radii))
                plt.title('Track t=%0.1f sec' % t[idx])
            plt.tight_layout()


class EpisodeRecorder(object):

    def __init__(self, filename, syncEvery=4096):
        self.filename = os.path.abspath(filename)
        self.syncEvery = syncEvery
        dirPath = os.path.dirname(self.filename)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        self.states = []
        self.dirty = False

    def _sync(self, force=False):
        if self.dirty or force:
            logger.debug('Syncing recorded states (%d total) to file: %s' % (len(self.states), self.filename))
            with gzip.open(self.filename, 'wb') as f:
                pickle.dump(self.states, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.dirty = False

    def save(self, observation, action):
        if observation is not None and action is not None:
            assert isinstance(observation, dict)
            assert isinstance(action, dict)

            state = dict()
            state.update(observation)
            for key, value in six.iteritems(action):
                state[key + 'Cmd'] = value

            self.states.append(state)
            self.dirty = True

            if len(self.states) % self.syncEvery == 0:
                self._sync()

    def close(self):
        self._sync()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self, *args):
        self.close()

    @staticmethod
    def restore(filename):
        with gzip.open(filename, 'rb') as f:
            states = pickle.load(f)
        return Episode(states)
