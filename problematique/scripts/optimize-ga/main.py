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

# Author: Xavier Groleau <xavier.groleau@@usherbrooke.ca>
# Author: Charles Quesnel <charles.quesnel@@usherbrooke.ca>
# Author: Michael Samson <michael.samson@@usherbrooke.ca>
# Université de Sherbrooke, APP2 S8GIA, A2018

import os
import sys
import numpy as np
import logging

import graph_utils
import ga

sys.path.append('../..')
from torcs.optim.core import TorcsOptimizationEnv, TorcsException

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)
 
# Appends the parameters of the best individual of a generation to a list of best parameters over generations    
def addParamsToList(paramsBest, bestIndividual):
    for key in paramsBest:
        paramsBest[key].append(bestIndividual[key][0])
    
def main(maxEvaluationTime=60, optimizeFor='fuelUsed', breedingRate=0.2, mutationRate=0.01, popSize=50, numIteration=100):
    try:
        with TorcsOptimizationEnv(maxEvaluationTime) as env:
            population = ga.initPopulation(env, popSize)
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
            paramsBest =  {'front-spoiler-angle': [], 'gear-2-ratio': [], 
                           'gear-3-ratio': [], 'gear-4-ratio': [], 
                           'gear-5-ratio': [], 'gear-6-ratio': [], 
                           'rear-differential-ratio': [], 'rear-spoiler-angle': []}

            
            # Loop a few times for demonstration purpose
            for i in range(numIteration):
                fitness = []
                observations = []
                for parameters in population:
                    # Generate a random vector of parameters in the proper interval
#                    logger.info('Generated new parameter vector: ' + str(parameters))
    
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
                avgMaxFuelRecord[i] = ga.getAverage(observations, 'fuelUsed')
                maxDistanceRecord[i] = observations[maxIndex]['distRaced'][0]
                overallMaxDistanceRecord[i] = bestIndividualObservations['distRaced'][0]
                avgMaxDistanceRecord[i] = ga.getAverage(observations, 'distRaced')
                maxSpeedRecord[i] = observations[maxIndex]['topspeed'][0]
                overallMaxSpeedRecord[i] = bestIndividualObservations['topspeed'][0]
                avgMaxSpeedRecord[i] = ga.getAverage(observations, 'topspeed')
                addParamsToList(paramsBest, population[maxIndex])
                logger.info('Best %s   =   %f', optimizeFor, fitness[maxIndex])
                logger.info('Generated new parameter vector: ' + str(population[maxIndex]))

                # Get new Gen
                population = ga.breedNewGeneration(population, fitness, popSize, breedingRate, mutationRate)
                
            graph_utils.showFigure(numIteration, maxFitnessRecord, overallMaxFitnessRecord, avgMaxFitnessRecord, 'Fitness', breedingRate, mutationRate, popSize)
            graph_utils.plotParams(paramsBest)
            
            # Uncomment to plot evolution of values over generations 
            
            #showFigure(numIteration, maxFuelRecord, overallMaxFuelRecord, avgMaxFuelRecord, 'Fuel',  breedingRate, mutationRate, popSize)
            #showFigure(numIteration, maxDistanceRecord, overallMaxDistanceRecord, avgMaxDistanceRecord, 'Distance',  breedingRate, mutationRate, popSize)
            #showFigure(numIteration, maxSpeedRecord, overallMaxSpeedRecord, avgMaxSpeedRecord, 'Top speed',  breedingRate, mutationRate, popSize)   
            
            # Display simulation results
            logger.info('##################################################')
            logger.info('Results:')
            logger.info('Time elapsed (sec) =   %f', maxEvaluationTime)
            logger.info('Parameters (br, mr, pop, #gen)=   %f %f %d %d', breedingRate, mutationRate, popSize, numIteration)
            logger.info('Top fitness        =   %f', overallMaxFitnessRecord[-1])
            logger.info('Top speed (km/h)   =   %f', overallMaxSpeedRecord[-1])
            logger.info('Distance raced (m) =   %f', overallMaxDistanceRecord[-1])
            logger.info('Fuel used (l)      =   %f', overallMaxFuelRecord[-1])
            logger.info(bestIndividual)
            logger.info('##################################################')
            return maxFitnessRecord
            
    except TorcsException as e:
        logger.error('Error occured communicating with TORCS server: ' + str(e))

    except KeyboardInterrupt:
        pass

    logger.info('All done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Set parameters for simulation here
    # Available options for `optimizeFor` are fuelUsed and topspeed
    main(maxEvaluationTime=60, optimizeFor='fuelUsed', breedingRate=0.2, mutationRate=0.01, popSize=75, numIteration=75)      
    
