import sys
import os
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

sys.path.append('../..')
from torcs.control.core import Episode, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))


recordingFilename = os.path.join(CDIR, '../drive-fuzzy/recordings', 'track-aalborg.pklz')
episode = EpisodeRecorder.restore(recordingFilename)

recordingFilename2 = os.path.join(CDIR, '../drive-simple/recordings', 'track-aalborg.pklz')
episode2 = EpisodeRecorder.restore(recordingFilename2)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(episode.angle, episode.trackPos, episode.steerCmd)
#ax.set_xlabel('Angle')
#ax.set_ylabel('Track Pos')
#ax.set_zlabel('Steer Cmd')

sin10 = 0.17365
cos10 = 0.98481

rxSensor = episode.track[:, 8]
cSensor = episode.track[:, 9]
sxSensor = episode.track[:, 10]

angle = np.zeros(rxSensor.shape)
targetSpeed = np.zeros(rxSensor.shape)

for i in range(len(angle)):
    if cSensor[i] > 95 or (cSensor[i] >= rxSensor[i] and cSensor[i] >= sxSensor[i]):
        targetSpeed[i] = 100
        continue
    
    if rxSensor[i] > sxSensor[i]:
        # Computing approximately the "angle" of turn
        h = cSensor[i] * sin10
        b = rxSensor[i] - cSensor[i] * cos10
        angle[i] = np.arcsin(b * b / (h * h + b * b))
    
                    # Approaching a turn on left
    else:
        # Computing approximately the "angle" of turn
        h = cSensor[i] * sin10
        b = sxSensor[i] - cSensor[i] * cos10
        angle[i] = np.arcsin(b * b / (h * h + b * b))
    
    targetSpeed[i] = 100 * (cSensor[i] * np.sin(angle[i]) / 95) * 2
targetSpeed = np.clip(targetSpeed, 0.0, 100)

plt.figure()
#plt.plot(np.gradient(episode.track[:, 8]), label='Track 8')
#plt.plot(np.gradient(episode.track[:, 9]), label='Track 9')
#plt.plot(np.gradient(episode.track[:, 10]), label='Track 10')
#plt.plot(targetSpeed, label='Target')
#plt.plot(episode.speed[:, 0], label='Actual')
#plt.plot(-episode.brakeCmd, label='Accel')
plt.plot(angle, label='Angle')
plt.legend()

#plt.figure()
#plt.hist(angle)

plt.figure()
plt.plot(episode.track[:, 8], label='Track 8')
plt.plot(episode.track[:, 9], label='Track 9')
plt.plot(episode.track[:, 10], label='Track 10')
plt.legend()

plt.figure()
plt.plot(targetSpeed, label='Target')
plt.plot(episode.speed[:, 0], label='Actual')
plt.plot(-episode.brakeCmd, label='Accel')
plt.legend()

#plt.figure()
#plt.plot(episode2.steerCmd, label='Steer Cmd')
#plt.plot(episode2.trackPos, label='Track Pos')
#plt.plot(episode2.angle, label='Angle')
#plt.legend()

#plt.figure()
#plt.hist(episode.angle)


plt.show()

episode.visualize(showObservations=True, showActions=True)
plt.show()