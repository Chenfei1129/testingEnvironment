
import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import random
import os
import statistics 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Local import
from src.MDPChasing.transitionFunction import TransitForNoPhysics, IsTerminal, StayInBoundaryByReflectVelocity, CheckBoundary, TransitionWithNoise, IsInObstacle

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
        self.stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(
            self.xBoundary, self.yBoundary)
        self.isTerminal = IsTerminal(
            self.minDistance, self.terminalPosition)

    @data(([[0, 50],[0, 0]], [True, False]), ([[25, 25],[48, 50]], [True, True]), ([[100, 2], [37, 30]],[False, True]), ([[0, 0], [300, 300]],[False, False]))
    @unpack
    def testTerminal(self, allAgentStates, groundTruth):
        inTerminal = self.isTerminal(allAgentStates)
        truthValue = inTerminal == groundTruth
        self.assertTrue(truthValue)
        
        
    @data((np.array([0, 0]), np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), np.array([[0, 0],[0, 0]])), 
    	  (np.array([1, 1]), np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 1]]), np.array([[2, 2],[3, 5]])),
          (np.array([0, 0]), np.array([[640, 2], [3, 4]]), np.array([[1, 0], [0, 1]]), np.array([[639, 2],[3, 5]])),
          (np.array([1, 0]), np.array([[640, 2], [0, 4]]), np.array([[1, -3], [-1, 1]]), np.array([[639, 1],[1, 5]])))
    @unpack	 
    def testTransition(self, standardDeviation, state, action, groundTruthReturnedNextStateMean):	
        transitionWithNoise = TransitionWithNoise (standardDeviation)
        transition = TransitForNoPhysics(self.stayInBoundaryByReflectVelocity, transitionWithNoise)
        nextStates = [transition(state, action) for _ in range(1000)]
        sampleMean = [sum(nextstate[0] for nextstate in nextStates)/len(nextStates), sum(nextstate[1] for nextstate in nextStates)/len(nextStates)]       
        truthValue = abs(sampleMean - groundTruthReturnedNextStateMean)<0.1
        self.assertTrue(truthValue.all())


    @data(([0, 0], [0, 0], np.array([0, 0]), np.array([0, 0])), 
          ([1, 1], [0, 0], np.array([0, 0]), np.array([1, 1])),
          ([1, 1], [1, 2], np.array([1, 2]), np.array([1, 1])))         
    @unpack
    def testTransitionWithNoiseMean(self, standardDeviation, mu, groundTruthSampleMean, groundTruthsampleStandardDeviation):
        transitionWithNoise = TransitionWithNoise(standardDeviation)
        nextStates = [transitionWithNoise(mu) for _ in range(1000)]
        samplemean = [sum(nextstate[0] for nextstate in nextStates)/len(nextStates), sum(nextstate[1] for nextstate in nextStates)/len(nextStates)]
        standardDeviationResult = [statistics.stdev(nextStates[0]), statistics.stdev(nextStates[1])]       
        truthValue = abs(samplemean - groundTruthSampleMean)< 0.1
        self.assertTrue(truthValue.all())

    @data(([0, 0], [0, 0], np.array([0, 0]), np.array([0, 0])), 
          ([2, 2], [0, 0], np.array([0, 0]), np.array([2, 2])),
          ([1, 1], [1, 2], np.array([1, 2]), np.array([1, 1])))         
    @unpack
    def testTransitionWithNoiseStandard(self, standardDeviation, mu, groundTruthSampleMean, groundTruthsampleStandardDeviation):
        transitionWithNoise = TransitionWithNoise(standardDeviation)
        nextStates = [transitionWithNoise(mu) for _ in range(1000)]
        samplemean = [sum(nextstate[0] for nextstate in nextStates)/len(nextStates), sum(nextstate[1] for nextstate in nextStates)/len(nextStates)]
        xNextState = [nextstate[0] for nextstate in nextStates]
        yNextState = [nextstate[1] for nextstate in nextStates]
        standardDeviationResult = [statistics.stdev(xNextState), statistics.stdev(yNextState)]       
        truthValue = abs(standardDeviationResult - groundTruthsampleStandardDeviation)< 1
        self.assertTrue(truthValue.all())

    @data(([0, 0], [0, 0], [0, 0]), ([1, -2], [1, -3], [1, 2]), ([1, 3], [2, 2], [1, 3]))
    @unpack
    def testCheckBoundaryAndAdjust(self, state, action, groundTruthNextState):
        checkState, checkAction = self.stayInBoundaryByReflectVelocity(state, action)
        truthValue = checkState == groundTruthNextState
        self.assertTrue(truthValue.all())

    @data(([1, 1], True), ([1, -2], False), ([650, 120], False))
    @unpack
    def testCheckBoundary(self, position, groundTruth):
        self.checkBoundary = CheckBoundary(self.xBoundary, self.yBoundary)
        returnedValue = self.checkBoundary(position)
        truthValue = returnedValue == groundTruth
        self.assertTrue(truthValue)
    
    @data(([0,0],[False,False]),([50,50],[False,False]),([150,200],[True,False]),([450,10],[False,True]))
    @unpack
    def testInObstacle(self, state, expectedResult):
        Obstacle = [[[100,200],[150,250]],[[400,450],[0,10]]]
        isInObstacle = IsInObstacle(Obstacle)
        checkInObstacle = isInObstacle(state)
        truthValue = checkInObstacle == expectedResult
        self.assertTrue(truthValue)
        
      

if __name__ == '__main__':
    unittest.main()