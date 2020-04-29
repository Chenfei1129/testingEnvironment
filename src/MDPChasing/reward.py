
import numpy as np
		
class RewardFunctionCompete():
    def __init__(self, timeCost, terminalReward, swampPenalty, isTerminal, isInSwamp):
        self.timeCost = timeCost
        self.terminalReward = terminalReward 
        self.isInSwamp = isInSwamp
        self.isTerminal = isTerminal
        self.swampPenalty = swampPenalty


    def __call__(self, state, action, agentID):
        reward = self.timeCost
        if self.isInSwamp(state, agentID):
            reward+=self.swampPenalty
        if self.isTerminal(state, agentID):
            reward += self.terminalReward
        return reward


