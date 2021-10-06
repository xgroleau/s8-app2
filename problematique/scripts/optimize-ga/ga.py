import numpy as np
import parameters

def initPopulation(env, popSize):
    population = []
    for i in range(popSize):
        population.append(env.action_space.sample())
    return reorderGears(population)

# Breeds the new generation of a population
def breedNewGeneration(population, fitness, popSize, crossoverProb, mutationProb):
    newPopulation = []
    
    # Remember two best elements to add back to next generation
    bestElements = np.array(population)[np.argpartition(fitness, -4)[-2:]]
    
    # Select a pair of parents to breed based on fitness
    numPairs = int(popSize / 2) - 1
    pairs = doSelection(population, fitness, numPairs)
    
    # Perform a cross-over and place individuals in the new population
    for genitor1, genitor2 in pairs:
        child1, child2 = doCrossover(genitor1, genitor2, crossoverProb)
        newPopulation.extend([child1, child2])
        
    newPopulation = np.array(newPopulation)
    
    # Apply mutation to all individuals in the population
    newPopulation = doMutation(newPopulation, mutationProb)
    
    # Reorders gears
    newPopulation = list(reorderGears(newPopulation))
    newPopulation.extend(bestElements)
    
    return newPopulation
    
# Selects which parents to use for a crossover based on a probability wheel
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

# Does a crossover (according to the probabilty) between both parents
def doCrossover(genitor1, genitor2, crossoverProb):
    child1 = genitor1
    child2 = genitor2
    
    if crossoverProb > np.random.uniform():
        dataGen1 = list(genitor1.items())
        dataGen2 = list(genitor2.items())
        
        # Select a random crossover point and create 2 children
        idx = np.random.randint(low=0, high=len(dataGen1))
        arrayChild1 = dataGen1[:idx] + dataGen2[idx:]
        arrayChild2 = dataGen2[:idx] + dataGen1[idx:]
        
        child1 = dict(arrayChild1)
        child2 = dict(arrayChild2)
        
    return child1, child2

# Mutates a certain percentage of bits in the chromosome
def doBinaryMutation(gene, mutationProb):
    # Encode gene in binary format
    geneBinary = encodeGene(gene)
    
    # Generate which bits are to be mutated according to the probabilty
    mutatedBits = np.random.uniform(size=geneBinary.shape) < mutationProb
    
    # Do mutation
    newGeneBinary = np.logical_xor(geneBinary.astype(np.bool), mutatedBits).astype(geneBinary.dtype)
    
    # Convert back to float values
    newGene = decodeGene(newGeneBinary)
    
    return newGene

# Encodes a gene by converting every chromosme (parameter) to its binary representation and appending it to a binary string
def encodeGene(element):
    gene = []
    
    for key in parameters.getKeys():
        # Convert parameter to binary
        minimum, rangeOfValue = parameters.getRange(key)
        chromosome = ufloat2bin((element[key] - minimum) / rangeOfValue , parameters.getNbits(key))
        
        # Append to the gene
        gene.extend(chromosome[0])
        
    return np.array(gene)

# Recreate the dict representing the parameters from the binary reprensentation of a gene
def decodeGene(gene):
    newValue = {}
    
    for i in range(8): 
        key = parameters.getKeyForIndex(i)
        nbits = parameters.getNbits(key)
        value = gene[:nbits]
        gene = gene[nbits:]
        minimum, rangeOfValue = parameters.getRange(key)
        newValue[key] = (bin2ufloat(value, nbits) * rangeOfValue) + minimum
        
    return newValue

# Convert float value between 0 and 1 to a binary value of nbits bits
def ufloat2bin(cvalue, nbits):
    # Convert value to integer using the resolution possible with bits available
    ivalue = np.round(cvalue * (2**nbits - 1)).astype(np.uint64)
    
    # Initialize bit array
    bvalue = np.zeros((len(cvalue), nbits))

    # Overflow
    bvalue[ivalue > 2**nbits - 1] = np.ones((nbits,))

    # Underflow
    bvalue[ivalue < 0] = np.zeros((nbits,))

    bitmask = (2**np.arange(nbits)).astype(np.uint64)
    bvalue[np.logical_and(ivalue >= 0, ivalue <= 2**nbits - 1)] = (np.bitwise_and(np.tile(ivalue[:, np.newaxis], (1, nbits)), np.tile(bitmask[np.newaxis, :], (len(cvalue), 1))) != 0)
    return bvalue

# Convert binary value back to float using a resolution of nbits
def bin2ufloat(bvalue, nbits):
    ivalue = np.sum(bvalue * (2**np.arange(nbits)[np.newaxis, :]), axis=-1)
    cvalue = ivalue / (2**nbits - 1)
    return cvalue


def doMutation(population, mutationProb):
    for i in range(len(population)):
        population[i] = doBinaryMutation(population[i], mutationProb)
                
    return population

# Returns the average value of the key observations 
def getAverage(observations, key):
    numElement = len(observations)
    average = 0.0
    for observation in observations:
        average += observation[key][0] / numElement
    return average

# Reorders gears so they are in descending order
def reorderGears(newPopulation):
    for i in range(len(newPopulation)):
        param, data = list(zip(*list(newPopulation[i].items())))
        indexEnd= param.index('gear-6-ratio') + 1
        indexStart = param.index('gear-2-ratio')
        sortedGears = list(data)[:indexStart] + sorted(list(data)[indexStart:indexEnd], key=lambda value: value[0], reverse=True) + list(data)[indexEnd:]
        newPopulation[i] = dict(zip(param, sortedGears))
        
    return newPopulation