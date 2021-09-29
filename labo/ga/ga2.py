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

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

########################
# Define helper functions here
########################


# usage: FITNESS = evaluateFitness(X, Y)
#
# Evaluate the 2-dimensional 'peak' function
#
# Input:
# - X, the x coordinate
# - Y, the y coordinate
#
# Output:
# - FITNESS, the value of the 'peak' function at coordinates (X,Y)
#
def evaluateFitness(x, y):
    # The 2-dimensional function to optimize
    fitness = (1 - x)**2 * np.exp(-x**2 - (y + 1)**2) - \
        (x - x**3 - y**5) * np.exp(-x**2 - y**2)
    return fitness


# usage: POPULATION = initializePopulation(NUMPARAMS, POPSIZE, NBITS)
#
# Initialize the population as a matrix, where each individual is a binary string.
#
# Input:
# - NUMPARAMS, the number of parameters to optimize.
# - POPSIZE, the population size.
# - NBITS, the number of bits per indivual used for encoding.
#
# Output:
# - POPULATION, a binary matrix whose rows correspond to encoded individuals.
#
def initializePopulation(numparams, popsize, nbits=8):
    # Parameters should be in the interval [0,1], so
    # initialize values randomly in the interval [0,1]

    # TODO: initialize the population
    population = np.zeros((popsize, numparams * nbits))
    return population


# usage: BVALUES = encodeIndividual(CVALUES, NBITS)
#
# Encode an individual from a vector of continuous values to a binary string.
#
# Input:
# - CVALUES, a vector of continuous values representing the parameters.
# - NBITS, the number of bits per indivual used for encoding.
#
# Output:
# - BVALUES, a binary vector encoding the individual.
#
def encodeIndividual(cvalues, nbits):
    numparams = len(cvalues)

    # TODO: encode individuals into binary vectors
    bvalues = np.zeros((numparams * nbits,))
    return bvalues


# usage: CVALUES = decodeIndividual(BVALUES, NUMPARAMS)
#
# Decode an individual from a binary string to a vector of continuous values.
#
# Input:
# - BVALUES, a binary vector encoding the individual.
# - NUMPARAMS, the number of parameters for an individual.
#
# Output:
# - CVALUES, a vector of continuous values representing the parameters.
#
def decodeIndividual(ind, numparams, minValue, maxValue):
    # TODO: decode individuals from binary vectors
    cvalues = np.zeros((numparams,))
    return cvalues

# usage: PAIRS = doSelection(POPULATION, FITNESS, NUMPAIRS)
#
# Select pairs of individuals from the population.
#
# Input:
# - POPULATION, the binary matrix representing the population. Each row is an individual.
# - FITNESS, a vector of fitness values for the population.
# - NUMPAIRS, the number of pairs of individual to generate.
#
# Output:
# - PAIRS, a cell array with a matrix [IND1 IND2] for each pair.
#


def doSelection(population, fitness, numPairs):

    # TODO: select pairs of individual in the population
    pairs = []
    for _ in range(numPairs):
        idx1 = np.random.randint(0, len(population))
        idx2 = np.random.randint(0, len(population))
        pairs.append((population[idx1], population[idx2]))

    return pairs


# usage: [NIND1,NIND2] = doCrossover(IND1, IND2, CROSSOVERPROB, CUTPOINTMOD)
#
# Perform a crossover operation between two individuals, with a given probability
# and constraint on the cutting point.
#
# Input:
# - IND1, a binary vector encoding the first individual.
# - IND2, a binary vector encoding the second individual.
# - CROSSOVERPROB, the crossover probability.
# - CUTPOINTMOD, a modulo-constraint on the cutting point. For example, to only allow cutting
#   every 4 bits, set value to 4.
#
# Output:
# - NIND1, a binary vector encoding the first new individual.
# - NIND2, a binary vector encoding the second new individual.
#
def doCrossover(ind1, ind2, crossoverProb, cutPointMod=1):
    # TODO: Perform a crossover between two individuals
    nind1 = ind1
    nind2 = ind2
    return nind1, nind2


# usage: [NPOPULATION] = doMutation(POPULATION, MUTATIONPROB)
#
# Perform a mutation operation over the entire population.
#
# Input:
# - POPULATION, the binary matrix representing the population. Each row is an individual.
# - MUTATIONPROB, the mutation probability.
#
# Output:
# - NPOPULATION, the new population.
#
def doMutation(population, mutationProb):
    # TODO: Apply mutation to the population
    npopulation = population
    return npopulation


# usage: [BVALUE] = ufloat2bin(CVALUE, NBITS)
#
# Convert floating point values into a binary vector
#
# Input:
# - CVALUE, a scalar or vector of continuous values representing the parameters.
#   The values must be a real non-negative float in the interval [0,1]!
# - NBITS, the number of bits used for encoding.
#
# Output:
# - BVALUE, the binary representation of the continuous value. If CVALUES was a vector,
#   the output is a matrix whose rows correspond to the elements of CVALUES.
#
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


# usage: [CVALUE] = bin2ufloat(BVALUE, NBITS)
#
# Convert a binary vector into floating point values
#
# Input:
# - BVALUE, the binary representation of the continuous values. Can be a single vector or a matrix whose
#   rows represent independent encoded values.
#   The values must be a real non-negative float in the interval [0,1]!
# - NBITS, the number of bits used for encoding.
#
# Output:
# - CVALUE, a scalar or vector of continuous values representing the parameters.
#   the output is a matrix whose rows correspond to the elements of CVALUES.
#
def bin2ufloat(bvalue, nbits):
    if nbits > 64:
        raise Exception('Maximum number of bits limited to 64')
    ivalue = np.sum(bvalue * (2**np.arange(nbits)[np.newaxis, :]), axis=-1)
    cvalue = ivalue / (2**nbits - 1)
    return cvalue


########################
# Define code logic here
########################


def main():

    # Fix random number generator seed for reproducible results
    np.random.seed(0)

    # Set to False to disable realtime plotting of landscape and population.
    # This is much faster!
    SHOW_LANDSCAPE = True

    # The parameters for encoding the population
    numparams = 2

    # TODO : adjust population size and encoding precision
    popsize = 40
    nbits = 16
    population = initializePopulation(numparams, popsize, nbits)

    if SHOW_LANDSCAPE:
        # Plot function to optimize
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Function landscape')
        xymin = -3.0
        xymax = 3.0
        x, y = np.meshgrid(np.linspace(xymin, xymax, 100),
                           np.linspace(xymin, xymax, 100))
        z = evaluateFitness(x, y)

        ax.plot_surface(x, y, z, cmap=plt.get_cmap('coolwarm'),
                        linewidth=0, antialiased=False)

        e = np.zeros((popsize,))
        sp, = ax.plot(e, e, e, markersize=10, color='k', marker='.', linewidth=0, zorder=10)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-1, 4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    # TODO : Adjust optimization meta-parameters
    numGenerations = 15
    mutationProb = 0.01
    crossoverProb = 0.8
    bestIndividual = []
    bestIndividualFitness = -1e10
    maxFitnessRecord = np.zeros((numGenerations,))
    overallMaxFitnessRecord = np.zeros((numGenerations,))
    avgMaxFitnessRecord = np.zeros((numGenerations,))

    for i in range(numGenerations):
        if SHOW_LANDSCAPE:
            # Plot landscape
            x, y, z = [], [], []
            for p in range(popsize):
                cvalues = decodeIndividual(population[p], numparams, xymin, xymax)
                fitness = evaluateFitness(cvalues[0], cvalues[1])
                x.append(cvalues[0])
                y.append(cvalues[1])
                z.append(fitness)

            sp.set_data(x, y)
            sp.set_3d_properties(z)
            fig.canvas.draw()
            plt.pause(0.02)

        # Evaluate fitness function for all individuals in the population
        fitness = np.zeros((popsize,))
        for p in range(popsize):
            # Convert population to float values
            cvalues = decodeIndividual(population[p, :], numparams, xymin, xymax)

            # Calculate fitness
            fitness[p] = evaluateFitness(cvalues[0], cvalues[1])

        # Save best individual across all generations
        bestFitness = np.max(fitness)
        if bestFitness > bestIndividualFitness:
            bestIndividual = population[fitness == np.max(fitness)][0]
            bestIndividualFitness = bestFitness

        # Record progress information
        maxFitnessRecord[i] = np.max(fitness)
        overallMaxFitnessRecord[i] = bestIndividualFitness
        avgMaxFitnessRecord[i] = np.mean(fitness)

        # Display progress information
        print('Generation no.%d: best fitness is %f, average is %f' %
              (i, maxFitnessRecord[i], avgMaxFitnessRecord[i]))
        print('Overall best fitness is %f' % bestIndividualFitness)

        newPopulation = []
        numPairs = int(popsize / 2)
        pairs = doSelection(population, fitness, numPairs)
        for ind1, ind2 in pairs:
            # Perform a cross-over and place individuals in the new population
            nind1, nind2 = doCrossover(ind1, ind2, crossoverProb, cutPointMod=nbits)
            newPopulation.extend([nind1, nind2])
        newPopulation = np.array(newPopulation)

        # Apply mutation to all individuals in the population
        newPopulation = doMutation(newPopulation, mutationProb)

        # Replace current population with the new one
        population = newPopulation

    # Display best individual
    print('#########################')
    print('Best individual (decoded values):')
    print(decodeIndividual(bestIndividual, numparams, xymin, xymax))
    print('#########################')

    # Display plot of fitness over generations
    fig = plt.figure()
    n = np.arange(numGenerations)
    ax = fig.add_subplot(111)
    ax.plot(n, maxFitnessRecord, '-r', label='Generation Max')
    ax.plot(n, overallMaxFitnessRecord, '-b', label='Overall Max')
    ax.plot(n, avgMaxFitnessRecord, '--k', label='Generation Average')
    ax.set_title('Fitness value over generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness value')
    ax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
