

import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# Local import
from src.MDPChasing.rewardFunction import RewardFunctionSingleAgent
from src.MDPChasing.transitionFunction import IsInSwamp, IsTerminalSingleAgent

@ddt
class TestReward(unittest.TestCase):
	def setUp(self):
		self.xBoundary = [0,640]
		self.yBoundary = [0,480]
		self.swamp = [[[300,400], [300, 400]], [[0, 1], [0, 10]]]
		self.actionCost = -1
		self.swampPenalty = -5   
		self.isInSwamp = IsInSwamp(self.swamp)
		self.minDistance = 10
		self.TerminalPosition = [500, 500]
		self.isTerminal = IsTerminalSingleAgent(self.minDistance, self.TerminalPosition)
		self.terminalReward = 2

	@data(
		([100, 450],  [0, 1], [100, 451],  -1),
		([0, 1],  [0, 2], [0, 3], -6),
		([350, 310], [0, 5], [350, 315] , -6),
		([490, 500], [1, 2], [491, 502],  1)

	) 
	@unpack
	def testRewardFunctionCompete(self, state, action, newState, result):
		findReward = RewardFunctionSingleAgent(self.actionCost, self.terminalReward, self.swampPenalty, self.isTerminal, self.isInSwamp)
		checkReward = findReward(state, action, newState)
		self.assertEqual(checkReward, result)



if __name__ == '__main__':
    unittest.main()
