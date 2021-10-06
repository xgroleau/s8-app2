import numpy as np

def getNbits(key):
    # Return the # of bits wanted for each key
    if 'gear' in key: 
        return 7
    elif 'spoiler' in key:
        return 8
    else:
        return 6
    
def getKeys():
    return ['front-spoiler-angle', 'gear-2-ratio', 'gear-3-ratio', 
            'gear-4-ratio', 'gear-5-ratio', 'gear-6-ratio', 
            'rear-differential-ratio', 'rear-spoiler-angle']
    
# Return the key value for a certain position in list
def getKeyForIndex(i):
    if i == 0: 
        return 'front-spoiler-angle'
    elif i == 6:
        return 'rear-differential-ratio'
    elif i == 7:
        return 'rear-spoiler-angle'
    else: 
        return 'gear-' + str(i+1) + '-ratio'
    
# Gives a random value for a key respecting the ranges of the different parameters
def randomInRange(key):
    
    if 'gear' in key: 
        return np.random.uniform(low=0.1, high=5.0)
    elif 'spoiler' in key:
        return np.random.uniform(low=0.0, high=90.0)
    else:
        return np.random.uniform(low=1.0, high=10.0)
    
# Returns the minimum and the range for a certain key   
def getRange(key): 
    if 'gear' in key: 
        return (0.1, 5.0 - 0.1)
    elif 'spoiler' in key:
        return (0.0, 90.0 - 0.0)
    else:
        return (1.0, 10.0 - 1.0)
    
