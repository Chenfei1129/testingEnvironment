
import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# Local import
from src.MDPChasing.rewardFunction import RewardFunctionCompete
from src.MDPChasing.transitionFunction import IsInSwamp, IsTerminal

@ddt
class TestReward(unittest.TestCase):
	def setUp(self):
		self.xBoundary = [0, 640]
		self.yBoundary = [0, 480]
		self.swamp = [[[300,400], [300, 400]], [[0, 1], [0, 10]]]
		self.actionCost = -1
		self.swampPenalty = -5   
		self.isInSwamp = IsInSwamp(self.swamp)
		self.minDistance = 10
		self.TerminalPosition = [500, 500]
		self.isTerminal = IsTerminal(self.minDistance, self.TerminalPosition)
		self.terminalReward = 2

	@data(
		([[100, 450], [0, 10]],  [[0, 0],[1, 0]],  0, -1),
		([[100, 450], [0, 10]],  [[1, 2],[3, 4]], 1, -6),
		([[350, 310], [0, 5]], [[1, 2],[2, 3]], 0, -6),
		([[490, 500], [1, 2]], [[0, 0],[1, 2]], 0, 1)

	) 
	@unpack
	def testRewardFunctionCompete(self, state, action, agentID, result):
		findReward = RewardFunctionCompete(self.actionCost, self.isTerminal, self.swampPenalty, self.isInSwamp, self.terminalReward)
		checkReward = findReward(state, action, agentID)
		self.assertEqual(checkReward, result)



if __name__ == '__main__':
    unittest.main()


