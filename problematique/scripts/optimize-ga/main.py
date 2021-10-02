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
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA,
# OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

import os
import sys
import time
import numpy as np
import logging

sys.path.append('../..')
from torcs.optim.core import TorcsOptimizationEnv, TorcsException

import matplotlib.pyplot as plt

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


################################
# Define helper functions here
################################

def initPopulation(env, popSize):
    population = []
    for i in range(popSize):
        population.append(env.action_space.sample())
    print(population[0])
    return population

def breedNewGeneration(population, fitness, popSize, crossoverProb, mutationProb):
    newPopulation = []
    numPairs = int(popSize / 2)
    pairs = doSelection(population, fitness, numPairs)
    
    for genitor1, genitor2 in pairs:
        # Perform a cross-over and place individuals in the new population
        child1, child2 = doCrossover(genitor1, genitor2, crossoverProb)
        newPopulation.extend([child1, child2])
    newPopulation = np.array(newPopulation)

    # Apply mutation to all individuals in the population
    return doMutation(newPopulation, mutationProb)
    #return newPopulation
    
def doSelection(population, fitness, numPairs):
    # Compute selection probability distribution
    eps = 1e-16
    fitness = fitness - np.min(fitness) + eps

    selectProb = np.cumsum(fitness) / np.sum(fitness)
    # Perform a roulette-wheel selection
    pairs = []
    for _ in range(numPairs):
        idx1 = np.argwhere(selectProb > np.random.uniform())[0][0]
        idx2 = np.argwhere(selectProb > np.random.uniform())[0][0]
        pairs.append((population[idx1], population[idx2]))

    return pairs

def doCrossover(genitor1, genitor2, crossoverProb):
    child1 = genitor1
    child2 = genitor2
    if crossoverProb > np.random.uniform():
        dataGen1 = list(genitor1.items())
        dataGen2 = list(genitor2.items())
        idx = np.random.randint(low=0, high=len(dataGen1))
        arrayChild1 = dataGen1[:idx] + dataGen2[idx:]
        arrayChild2 = dataGen2[:idx] + dataGen1[idx:]
        child1 = dict(arrayChild1)
        child2 = dict(arrayChild2)
        
    return child1, child1

def doBinaryMutation(element, mutationProb):
    element = encodeElement(element)
    mutatedBits = np.random.uniform(size=element.shape) < mutationProb
    nelement = np.logical_xor(element.astype(np.bool), mutatedBits).astype(element.dtype)
    newElement = decodeElement(nelement)
    return newElement

def encodeElement(element):
    bvalues = []
    for key in element:
        minimum, rangeOfValue = getRange(key)
        binValue = ufloat2bin((element[key] - minimum) / rangeOfValue , getNbits(key))
        bvalues.extend(binValue[0])
    return np.array(bvalues)

def decodeElement(element):
    newValue = {}
    for i in range(8): 
        key = getKeyForI(i)
        nbits = getNbits(key)
        value = element[:nbits]
        element = element[nbits:]
        minimum, rangeOfValue = getRange(key)
        newValue[key] = (bin2ufloat(value, nbits) * rangeOfValue) + minimum
        
    return newValue

def ufloat2bin(cvalue, nbits):
    if nbits > 64:
        raise Exception('Maximum number of bits limited to 64')
    ivalue = np.round(cvalue * (2**nbits - 1)).astype(np.uint64)
    bvalue = np.zeros((len(cvalue), nbits))

    # Overflow
    bvalue[ivalue > 2**nbits - 1] = np.ones((nbits,))

    # Underflow
    bvalue[ivalue < 0] = np.zeros((nbits,))

    bitmask = (2**np.arange(nbits)).astype(np.uint64)
    bvalue[np.logical_and(ivalue >= 0, ivalue <= 2**nbits - 1)] = (np.bitwise_and(np.tile(ivalue[:, np.newaxis], (1, nbits)), np.tile(bitmask[np.newaxis, :], (len(cvalue), 1))) != 0)
    return bvalue


def bin2ufloat(bvalue, nbits):
    if nbits > 64:
        raise Exception('Maximum number of bits limited to 64')
    ivalue = np.sum(bvalue * (2**np.arange(nbits)[np.newaxis, :]), axis=-1)
    cvalue = ivalue / (2**nbits - 1)
    return cvalue


def doMutation(population, mutationProb):
    for i in range(len(population)):
        population[i] = doBinaryMutation(population[i], mutationProb)
#        for key in population[i]:    
#            if mutationProb > np.random.uniform():
#                population[i][key] = np.array([randomInRange(key)])
    #mutatedBits = np.random.uniform(size=population.shape) < mutationProb
    #npopulation = np.logical_xor(population.astype(np.bool), mutatedBits).astype(population.dtype)
    return population

def getNbits(key):
    if 'gear' in key: 
        return 7
    elif 'spoiler' in key:
        return 8
    else:
        return 6
    
def getKeyForI(i):
    if i == 0: 
        return 'front-spoiler-angle'
    elif i == 6:
        return 'rear-differential-ratio'
    elif i == 7:
        return 'rear-spoiler-angle'
    else: 
        return 'gear-' + str(i+1) + '-ratio'
        
def randomInRange(key):
    if 'gear' in key: 
        return np.random.uniform(low=0.1, high=5.0)
    elif 'spoiler' in key:
        return np.random.uniform(low=0.0, high=90.0)
    else:
        return np.random.uniform(low=1.0, high=10.0)
    
def getRange(key):
    if 'gear' in key: 
        return (0.1, 5.0 - 0.1)
    elif 'spoiler' in key:
        return (0.0, 90.0 - 0.0)
    else:
        return (1.0, 10.0 - 1.0)
    
def getAverage(observations, key):
    numElement = len(observations)
    average = 0.0
    for observation in observations:
        average += observation[key][0] / numElement
    return average

def showFigure(numIteration, maxRecord, overallMaxRecord, avgRecord, valueShown):
    fig = plt.figure()
    n = np.arange(numIteration)
    ax = fig.add_subplot(111)
    ax.plot(n, maxRecord, '-r', label='For Generation Max')
    ax.plot(n, overallMaxRecord, '-b', label='For Overall Max')
    ax.plot(n, avgRecord, '--k', label='Generation Average')
    ax.set_title(valueShown + ' value over generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel(valueShown + ' value')
    ax.legend()
    fig.tight_layout()

    plt.show()
            
def main(maxEvaluationTime=40, optimizeFor='distRaced', breedingRate=0.8, mutationRate=0.01, popSize=50, numIteration=150):
    try:
        with TorcsOptimizationEnv(maxEvaluationTime) as env:
            population = initPopulation(env, popSize)
            bestIndividual = []
            bestIndividualFitness = -1e10
            bestIndividualObservations = 0
            maxFitnessRecord = np.zeros((numIteration,))
            overallMaxFitnessRecord = np.zeros((numIteration,))
            avgMaxFitnessRecord = np.zeros((numIteration,))
            maxFuelRecord = np.zeros((numIteration,))
            overallMaxFuelRecord = np.zeros((numIteration,))
            avgMaxFuelRecord = np.zeros((numIteration,))
            maxDistanceRecord = np.zeros((numIteration,))
            overallMaxDistanceRecord = np.zeros((numIteration,))
            avgMaxDistanceRecord = np.zeros((numIteration,))
            maxSpeedRecord = np.zeros((numIteration,))
            overallMaxSpeedRecord = np.zeros((numIteration,))
            avgMaxSpeedRecord = np.zeros((numIteration,))
            # Loop a few times for demonstration purpose
            for i in range(numIteration):
                fitness = []
                observations = []
                for parameters in population:
                    # Generate a random vector of parameters in the proper interval
                    # logger.info('Generated new parameter vector: ' + str(parameters))
    
                    # Perform the evaluation with the simulator
                    observation, _, _, _ = env.step(parameters)
                    
                    #Save values for plotting and log max speed
                    if optimizeFor == 'fuelUsed':
                        fitness.append((observation['distRaced'][0] / observation['fuelUsed'][0]) / 1000)
                    else:
                        fitness.append(observation[optimizeFor][0])
                    observations.append(observation)


                maxIndex = fitness.index(max(fitness))
                if fitness[maxIndex] > bestIndividualFitness:
                    bestIndividual = population[maxIndex]
                    bestIndividualFitness = fitness[maxIndex]
                    bestIndividualObservations = observations[maxIndex]
                
                maxFitnessRecord[i] = np.max(fitness)
                overallMaxFitnessRecord[i] = bestIndividualFitness
                avgMaxFitnessRecord[i] = np.mean(fitness)
                maxFuelRecord[i] = observations[maxIndex]['fuelUsed'][0]
                overallMaxFuelRecord[i] = bestIndividualObservations['fuelUsed'][0]
                avgMaxFuelRecord[i] = getAverage(observations, 'fuelUsed')
                maxDistanceRecord[i] = observations[maxIndex]['distRaced'][0]
                overallMaxDistanceRecord[i] = bestIndividualObservations['distRaced'][0]
                avgMaxDistanceRecord[i] = getAverage(observations, 'distRaced')
                maxSpeedRecord[i] = observations[maxIndex]['topspeed'][0]
                overallMaxSpeedRecord[i] = bestIndividualObservations['topspeed'][0]
                avgMaxSpeedRecord[i] = getAverage(observations, 'topspeed')
                logger.info('Best %s   =   %f', optimizeFor, fitness[maxIndex])
                
                # Get new Gen
                population = breedNewGeneration(population, fitness, popSize, breedingRate, mutationRate)
                
            showFigure(numIteration, maxFitnessRecord, overallMaxFitnessRecord, avgMaxFitnessRecord, 'Fitness')
            showFigure(numIteration, maxFuelRecord, overallMaxFuelRecord, avgMaxFuelRecord, 'Fuel')
            showFigure(numIteration, maxDistanceRecord, overallMaxDistanceRecord, avgMaxDistanceRecord, 'Distance')
            showFigure(numIteration, maxSpeedRecord, overallMaxSpeedRecord, avgMaxSpeedRecord, 'Top speed')

           
#                 Display simulation results
            logger.info('##################################################')
            logger.info('Results:')
            logger.info('Time elapsed (sec) =   %f', maxEvaluationTime)
            logger.info('Top fitness        =   %f', overallMaxFitnessRecord[-1])
            logger.info('Top speed (km/h)   =   %f', overallMaxSpeedRecord[-1])
            logger.info('Distance raced (m) =   %f', overallMaxDistanceRecord[-1])
            logger.info('Fuel used (l)      =   %f', overallMaxFuelRecord[-1])
            logger.info(bestIndividual)
            logger.info('##################################################')

    except TorcsException as e:
        logger.error('Error occured communicating with TORCS server: ' + str(e))

    except KeyboardInterrupt:
        pass

    logger.info('All done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
