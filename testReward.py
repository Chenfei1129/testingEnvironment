import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# Local import
from src.MDPChasing.rewardFunction import RewardFunctionCompete
from src.MDPChasing.envNoPhysics import IsInObstacle2, IsTerminal2

@ddt
class TestReward(unittest.TestCase):
	def setUp(self):
		self.xBoundary = [0,640]
		self.yBoundary = [0,480]
		self.Obstacle = [[[300,400], [300, 400]]]
		self.actionCost = -1
		self.obstaclePenalty = -5   
		self.isInObstacle = IsInObstacle2(self.Obstacle)
		self.minDistance = 10
		self.TerminalPosition = [500, 500]
		self.isTerminal = IsTerminal2(self.minDistance, self.TerminalPosition)
		self.terminalReward = 2

	@data(
		([100, 450], [0, 0], -1),
		([350, 310], [1, 0], -6),
		([490, 500], [1, 0], 1)

	) 
	@unpack
	def testRewardFunctionCompete(self, state, action, result):
		findReward = RewardFunctionCompete(self.actionCost, self.isTerminal, self.obstaclePenalty, self.isInObstacle, self.terminalReward)
		checkReward = findReward(state,action)
		self.assertEqual(checkReward, result)



if __name__ == '__main__':
    unittest.main()

