import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import random
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Local import
from src.MDPChasing.envNoPhysics import TransitForNoPhysics, IsTerminal, StayInBoundaryByReflectVelocity, CheckBoundary

@ddt
class TestEnvNoPhysics(unittest.TestCase):
    def setUp(self):
        
        self.xBoundary = [0, 640]
        self.yBoundary = [0, 480]
        self.stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(
            self.xBoundary, self.yBoundary)
        
        
    @data((np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), np.array([[0,0],[0,0]])), 
    	(np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[3,2],[3,6]])))
    @unpack	 
    def testTransition(self, state, action, sigma, groundTruthReturnedNextState):
    	transition = TransitForNoPhysics(self.stayInBoundaryByReflectVelocity)
    	nextState = transition(state, action, sigma)
    	truthValue = nextState == groundTruthReturnedNextState
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
        
        
      

if __name__ == '__main__':
    unittest.main()