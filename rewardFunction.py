import numpy as np

class IsInObstacle():
	def __init__(self, xObstacle, yObstacle):
		self.xObstacle = xObstacle
		self.yObstacle = yObstacle

	def __call__(self, state):
		return (state[0] >= self.xObstacle[0] and state[0] <= self.xObstacle[1] and state[1] >= self.yObstacle[0] and state[1] <= self.yObstacle[1])
		
class RewardFunctionCompete():
    def __init__(self, actionCost, isTerminal, obstaclePenalty, isInObstacle, terminalReward):
        self.actionCost = actionCost## one second how much 
        self.isInObstacle = isInObstacle
        self.isTerminal = isTerminal
        self.obstaclePenalty = obstaclePenalty
        self.terminalReward = terminalReward

    def __call__(self, state, action):
        reward = self.actionCost
        if self.isInObstacle(state):
            reward+=self.obstaclePenalty
        if self.isTerminal(state):
            reward += self.terminalReward
        return reward


