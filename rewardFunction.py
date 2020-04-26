import numpy as np
		
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


