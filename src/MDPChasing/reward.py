
import numpy as np
		
class RewardFunctionCompete():
    def __init__(self, timeCost, terminalReward, swampPenalty, isTerminal, isInSwamp):
        self.timeCost = timeCost
        self.terminalReward = terminalReward 
        self.swampPenalty = swampPenalty
        self.isInSwamp = isInSwamp
        self.isTerminal = isTerminal


    def __call__(self, state, action):
        reward = self.timeCost
        if self.isInSwamp(state):
            reward+=self.swampPenalty
        if self.isTerminal(state):
            reward += self.terminalReward
        return reward


