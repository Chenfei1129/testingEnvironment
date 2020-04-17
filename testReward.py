import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# Local import
from src.MDPChasing.rewardFunction import RewardFunctionCompete,IsInObstacle

@ddt
class TestReward(unittest.TestCase):
	def setUp(self):
		self.xBoundary = [0,640]
		self.yBoundary = [0,480]
		self.xObstacle = [300,400]
		self.yObstacle = [300,400]
		self.actionCost = -1
		self.obstaclePenalty = -5   
		self.isInObstacle = IsInObstacle(self.xObstacle, self.yObstacle)
		self.isTerminal = lambda state: state == [5,5]
		self.terminalReward = 3
	@data(
		([100,450],[1,1],-1),
		([200,300],[21,21],-1),
		([350,350],[10,0],-6),
		([5,5],[0,10],2)
	) 
	@unpack
	def testRewardFunctionCompete(self,state,action,result):
		findReward = RewardFunctionCompete(self.actionCost, self.isTerminal, self.obstaclePenalty, self.isInObstacle, self.terminalReward)
		checkReward = findReward(state,action)
		
		self.assertEqual(checkReward, result)



	@data(([350,350],True),([300,300],True),([0,300],False),([500,350],False))
	@unpack
	def testInObstacle(self,state,groundTruth):
		calculated=self.isInObstacle(state)
		self.assertEqual(calculated,groundTruth)


if __name__ == '__main__':
    unittest.main()
