


import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys 
import random
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Local import
from src.MDPChasing.transitionFunction3 import TransitInSwampWorld, IsTerminal, StayInBoundaryByReflectVelocity, CheckBoundary, TransitionWithNoise, IsInSwamp, IsTerminal, IsTerminalAll

@ddt
class TestEnvNoPhysics(unittest.TestCase):
    def setUp(self):
        self.numOfAgent = 2
        self.sheepId = 0
        self.wolfId = 1
        self.minDistance = 50
        self.terminalPosition = [50, 50]
        self.xBoundary = [0, 640]
        self.yBoundary = [0, 480]
        self.isTerminal = IsTerminal(self.minDistance, self.terminalPosition, self.sheepId)
        self.stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(
            self.xBoundary, self.yBoundary)

    @data(([0, 0], [0, 0], [0, 0], [0, 0]), 
          ([1, 1], [0, 0], [0, 0], [1, 1]),
          ([1, 1], [1, 2], [1, 2], [1, 1]))         
    @unpack
    def testTransitionWithNoiseMean(self, standardDeviation, mu, groundTruthSampleMean, groundTruthsampleStandardDeviation):
        transitionWithNoise = TransitionWithNoise(standardDeviation)
        nextStates = [transitionWithNoise(mu) for _ in range(1000)]
        samplemean = [sum(nextstate[0] for nextstate in nextStates)/len(nextStates), sum(nextstate[1] for nextstate in nextStates)/len(nextStates)]
        standardDeviationResult = [np.std(nextStates[0]), np.std(nextStates[1])]
        truthValue = abs(np.array(samplemean) - np.array(groundTruthSampleMean))< 0.1
        self.assertTrue(truthValue.all())

    @data(([0, 0], [0, 0], [0, 0], [0, 0]), 
          ([2, 2], [0, 0], [0, 0], [2, 2]),
          ([1, 1], [1, 2], [1, 2], [1, 1]))         
    @unpack
    def testTransitionWithNoiseStandard(self, standardDeviation, mu, groundTruthSampleMean, groundTruthsampleStandardDeviation):
        transitionWithNoise = TransitionWithNoise(standardDeviation)
        nextStates = [transitionWithNoise(mu) for _ in range(1000)]
        samplemean = [sum(nextstate[0] for nextstate in nextStates)/len(nextStates), sum(nextstate[1] for nextstate in nextStates)/len(nextStates)]
        xNextState = [nextstate[0] for nextstate in nextStates]
        yNextState = [nextstate[1] for nextstate in nextStates]
        standardDeviationResult = [np.std(xNextState), np.std(yNextState)]       
        truthValue = abs(np.array(standardDeviationResult) - np.array(groundTruthsampleStandardDeviation))< 1
        self.assertTrue(truthValue.all())

    @data(([0, 0], [0, 0], [0, 0]), ([1, -2], [1, -3], [1, 2]), ([1, 3], [2, 2], [1, 3]))
    @unpack
    def testCheckBoundaryAndAdjust(self, state, action, groundTruthNextState):
        checkState, checkAction = self.stayInBoundaryByReflectVelocity(state, action)
        truthValue = checkState == groundTruthNextState
        self.assertTrue(truthValue)

    @data(([1, 1], True), ([1, -2], False), ([650, 120], False))
    @unpack
    def testCheckBoundary(self, position, groundTruth):
        self.checkBoundary = CheckBoundary(self.xBoundary, self.yBoundary)
        returnedValue = self.checkBoundary(position)
        truthValue = returnedValue == groundTruth
        self.assertTrue(truthValue)
    
    @data(([[0, 0], [0, 0]], 0, False),([[50, 50], [0, 0]], 1, False),([[150, 200], [450, 10]], 1, True))
    @unpack
    def testInSwamp(self, state, agentID, expectedResult):
        swamp = [[[100, 200], [150, 250]],[[400, 450], [0, 10]]]
        isInSwamp = IsInSwamp(swamp, agentID)
        checkInSwamp = isInSwamp(state)
        truthValue = checkInSwamp == expectedResult
        self.assertTrue(truthValue)

    @data(([[0, 50],[0, 0]], 0, True), ([[25, 25],[48, 50]], 1, True), ([[100, 2], [37, 30]], 0, False), ([[0, 0], [300, 300]], 1, False))
    @unpack
    def testTerminal(self, state, agentID, groundTruth):
        inTerminal = self.isTerminal(state)
        truthValue = inTerminal == groundTruth
        self.assertTrue(truthValue)

    @data(([[0, 50],[0, 0]], True), ([[25, 25],[48, 50]],  True), ([[100, 2], [37, 30]],  False), ([[0, 0], [300, 300]],  False))
    @unpack
    def testTerminalAll(self, state, groundTruth):
        isTerminalAll = IsTerminalAll(self.minDistance, self.terminalPosition, self.isTerminal)
        inTerminalAll = isTerminalAll(state)
        truthValue = inTerminalAll == groundTruth
        self.assertTrue(truthValue)
        
    @data(([0, 0], [[0, 0], [0, 0]],[[0, 0], [0, 0]], [[0, 0],[0, 0]]), 
           ([1, 1], [[1, 2], [3, 4]], [[1, 0], [0, 1]], [[2, 2],[3, 5]]),
           ([0, 0], [[640, 2], [3, 4]],[[1, 0], [0, 1]],[[639, 2],[3, 5]]),
           ([1, 0], [[640, 2], [0, 4]], [[1, -3], [-1, 1]], [[639, 1],[1, 5]]))
    @unpack

    def testTransition(self, standardDeviation, state, action, groundTruthReturnedNextStateMean):   
        transitionWithNoise = TransitionWithNoise (standardDeviation)
        transition = TransitInSwampWorld(self.stayInBoundaryByReflectVelocity, transitionWithNoise)
        nextStates = [transition(state, action) for _ in range(1000)]
        sampleMeanX = [sum(nextstate[0][0] for nextstate in nextStates)/len(nextStates), sum(nextstate[0][1] for nextstate in nextStates)/len(nextStates)] 
        sampleMeanY = [sum(nextstate[1][0] for nextstate in nextStates)/len(nextStates), sum(nextstate[1][1] for nextstate in nextStates)/len(nextStates)]
        sampleMean = [sampleMeanX, sampleMeanY]
        truthValue = abs(np.array(sampleMean) - np.array(groundTruthReturnedNextStateMean))<0.1
        self.assertTrue(truthValue.all()) 



if __name__ == '__main__':
    unittest.main()
