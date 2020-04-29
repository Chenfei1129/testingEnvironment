import numpy as np
import random


def maxFromDistribution(distribution):
    hypotheses = list(distribution.keys())
    probs = list(distribution.values())
    maxIndices = np.argwhere(probs == np.max(probs)).flatten()
    selectedIndex = np.random.choice(maxIndices)
    selectedHypothesis = hypotheses[selectedIndex]
    return selectedHypothesis 


def sampleFromDistribution(distribution):
    hypotheses = list(distribution.keys())
    probs = list(distribution.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedHypothesis = hypotheses[selectedIndex]
    return selectedHypothesis 

class SoftDistribution:
    def __init__(self, softParameter):
        self.softParameter = softParameter

    def __call__(self, dist):
        hypotheses = list(dist.keys())
        softenUnnormalizedProbabilities = np.array([np.power(probability, self.softParameter) for probability in list(actionDist.values())])
        softenNormalizedProbabilities = list(softenUnnormalizedProbabilities / np.sum(softenUnnormalizedProbabilities))
        softenDist = dict(zip(hypotheses, softenNormalizedProbabilities))
        return softenDist

