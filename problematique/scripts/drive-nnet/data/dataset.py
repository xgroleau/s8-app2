from os import listdir
from os.path import join, isfile
import sys

import numpy as np
from .utils.utils import normalize, normalize_val, denormalize
from keras.utils import to_categorical

sys.path.append('../..')
sys.path.append('../../..')
from torcs.control.core import EpisodeRecorder


class DataSet:

    # Input
    speed_x: np.array
    speed_y: np.array
    angle: np.array
    trackPos: np.array
    track: np.array
    wheelSpinVel: np.array
    gear: np.array
    rpm: np.array

    # Output
    accelCmd: np.array
    brakeCmd: np.array
    gearCmd: np.array
    steerCmd: np.array

    def __init__(self, files_dir):
        files = [join(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f))]
        episodes = [EpisodeRecorder.restore(f) for f in files]

        # Input
        speed = np.concatenate([e.speed for e in episodes])
        self.speed_x = speed[:, 0]
        self.speed_y = speed[:, 1]

        gear = np.concatenate([e.gear for e in episodes]).squeeze()
        self.gear = self.gear_to_categorical(gear)

        self.angle = np.concatenate([e.angle for e in episodes]).squeeze()
        self.trackPos = np.concatenate([e.trackPos for e in episodes]).squeeze()
        self.track = np.concatenate([e.track for e in episodes]).squeeze()
        self.wheelSpinVel = np.concatenate([e.wheelSpinVel for e in episodes]).squeeze()
        self.rpm = np.concatenate([e.rpm for e in episodes]).squeeze()

        # Output
        gearCmd = np.concatenate([e.gearCmd for e in episodes]).squeeze()
        self.gearCmd = self.gear_to_categorical(gearCmd)  # Makes all values positive, 8 classes from -1 to 6

        self.accelCmd = np.concatenate([e.accelCmd for e in episodes]).squeeze()
        self.brakeCmd = np.concatenate([e.brakeCmd for e in episodes]).squeeze()
        self.accelBrakeCmd = self.accelCmd - self.brakeCmd

        self.steerCmd = np.concatenate([e.steerCmd for e in episodes]).squeeze()

    def normalize(self):
        # Gear is not normalized since it's a discreate value
        self.speed_x, self.min_speed_x, self.max_speed_x = normalize(self.speed_x)
        self.speed_y, self.min_speed_y, self.max_speed_y = normalize(self.speed_y)
        self.angle, self.min_angle, self.max_angle = normalize(self.angle)
        self.trackPos, self.min_trackPos, self.max_trackPos = normalize(self.trackPos)
        self.track, self.min_track, self.max_track = normalize(self.track)
        self.wheelSpinVel, self.min_wheelSpinVel, self.max_wheelSpinVel = normalize(self.wheelSpinVel)
        self.rpm, self.min_rpm, self.max_rpm = normalize(self.rpm)

    def normalize_angle(self, angle):
        return normalize_val(angle, self.min_angle, self.max_angle)

    def normalize_speed_x(self, speed_x):
        return normalize_val(speed_x, self.min_speed_x, self.max_speed_x)

    def normalize_speed_y(self, speed_y):
        return normalize_val(speed_y, self.min_speed_y, self.max_speed_y)

    def normalize_trackPos(self, trackPos):
        return normalize_val(trackPos, self.min_trackPos, self.max_trackPos)

    def normalize_wheelSpinVel(self, wheelSpinVel):
        return normalize_val(wheelSpinVel, self.min_wheelSpinVel, self.max_wheelSpinVel)

    def normalize_rpm(self, rpm):
        return normalize_val(rpm, self.min_rpm, self.max_rpm)

    def normalize_track(self, track):
        return normalize_val(track, self.min_track, self.max_track)

    def gear_to_categorical(self, gear):
        return to_categorical(gear + 1, num_classes=8)

    def gear_to_int(self, categorical_gear):
        return np.argmax(categorical_gear) -1