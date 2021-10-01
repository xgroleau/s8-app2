import sys
import os
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../..')
from torcs.control.core import Episode, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))


recordingFilename = os.path.join(CDIR, '../drive-fuzzy/recordings', 'track-aalborg.pklz')
episode = EpisodeRecorder.restore(recordingFilename)

recordingFilename2 = os.path.join(CDIR, '../drive-simple/recordings', 'track-aalborg.pklz')
episode2 = EpisodeRecorder.restore(recordingFilename2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(episode.angle, episode.trackPos, episode.steerCmd)
ax.set_xlabel('Angle')
ax.set_ylabel('Track Pos')
ax.set_zlabel('Steer Cmd')

plt.figure()
plt.plot(episode.steerCmd, label='Steer Cmd')
plt.plot(episode.trackPos, label='Track Pos')
plt.plot(episode.angle, label='Angle')
plt.legend()

plt.figure()
plt.plot(episode2.steerCmd, label='Steer Cmd')
plt.plot(episode2.trackPos, label='Track Pos')
plt.plot(episode2.angle, label='Angle')
plt.legend()

plt.figure()
plt.hist(episode.angle)


plt.show()

episode.visualize(showObservations=True, showActions=True)
plt.show()