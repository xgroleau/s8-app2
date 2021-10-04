import sys
import os
import matplotlib.pyplot as plt

sys.path.append('../..')
from torcs.control.core import Episode, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

files = []
labels = ['Sugeno - Mult', 'Sugeno - Min', 'Mandani', 'Simple Driver']
episodes = []

files.append(os.path.join(CDIR, '../drive-fuzzy/recordings', 'sugeno-multiply-0.pklz'))
files.append(os.path.join(CDIR, '../drive-fuzzy/recordings', 'sugeno-fmin-0.pklz'))
files.append(os.path.join(CDIR, '../drive-fuzzy/recordings', 'sugeno-mandani-0.pklz'))
files.append(os.path.join(CDIR, '../drive-fuzzy/recordings', 'sugeno-simple-0.pklz'))


for fileName in files:
    episodes.append(EpisodeRecorder.restore(fileName))

plt.figure()

for i, episode in enumerate(episodes):
    plt.plot(episode.trackPos[1000:2200], label=labels[i])

plt.title('Steer Cmd')
plt.xlabel('Sample')
plt.ylabel("Steer Cmd")
plt.legend()
plt.show()
