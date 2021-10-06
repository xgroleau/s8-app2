import numpy as np
import matplotlib.pyplot as plt


def showFigure(numIteration, maxRecord, overallMaxRecord, avgRecord, valueShown, breedingRate, mutationRate, popSize):
    fig = plt.figure()
    n = np.arange(numIteration)
    ax = fig.add_subplot(111)
    ax.plot(n, maxRecord, '-r', label='Generation Max')
    ax.plot(n, overallMaxRecord, '-b', label='Overall Max')
    ax.plot(n, avgRecord, '--k', label='Generation Average')
    ax.set_title(valueShown + ' over generations (br: ' + 
                 str(breedingRate) + ', mr: ' + str(mutationRate) + 
                 ') popSize: ' + str(popSize))
    ax.set_xlabel('Generation')
    ax.set_ylabel(valueShown + ' value')
    ax.legend()
    fig.tight_layout()

    plt.show()

# Plots the different paramaters (gear ratios, spoiler angles) over generations            
def plotParams(paramBest):
    plt.figure()
    axGear = plt.subplot(211)
    axAngle = plt.subplot(212)
    
    
    for key in paramBest:
        if 'ratio' in key:
            axGear.plot(paramBest[key], label=key)
        else:
            axAngle.plot(paramBest[key], label=key)
            
    axAngle.set_xlabel('Generation')
    axGear.set_ylabel('Gear ratios')
    axAngle.set_ylabel('Angle')
    
    # Set label to the left so the data is not hidden
    axGear.legend(loc=2)
    axAngle.legend()
    
    plt.show()# -*- coding: utf-8 -*-

