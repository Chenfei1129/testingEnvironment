import numpy as np
'''
def isTerminal(state,terminalState):
	if state==terminalState:
		return True
	else:
		return False
'''
class IsInObstacle():
	def __init__(self, xObstacle, yObstacle):
		self.xObstacle=xObstacle
		self.yObstacle=yObstacle

	def __call__(self, state):
		if state[0] >= self.xObstacle[0] and state[0] <= self.xObstacle[1] and state[1] >= self.yObstacle[0] and state[1] <= self.yObstacle[1]:
		    return True
		else:
			return False
		
'''				   	        
def isInObstacle(state):
	if state[0] >= xObstacle[0] and state[0] <= xObstacle[1] and state[1] >= yObstacle[0] and state[1] <= yObstacle[1]:
		    return True
	else:
			return False	
'''
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


