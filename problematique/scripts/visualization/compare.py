import sys
import os
import logging
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../..')
from torcs.control.core import Episode, EpisodeRecorder

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

trackName = 'forza'
ais = ['drive-bot', 'drive-simple', 'drive-fuzzy']

for ai in ais:
    recordingFilename = os.path.join(CDIR, f'../{ai}/recordings', f'track-{trackName}.pklz')
    episode = EpisodeRecorder.restore(recordingFilename)
    
    time = np.max(episode.curLapTime)
    meanSpeed = np.mean(episode.speed[:,0])
    maxSpeed = np.max(episode.speed[:,0])
    
    print(f'{ai} : {trackName}')
    print(f'Time elapsed: {time}')
    print(f'Mean Speed: {meanSpeed}')
    print(f'Max Speed: {maxSpeed}')
    print('**********\n')
