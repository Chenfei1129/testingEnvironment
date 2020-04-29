
import numpy as np
		
class RewardFunctionCompete():
    def __init__(self, actionCost, isTerminal, obstaclePenalty, isInSwamp, terminalReward):
        self.actionCost = actionCost ## how much cost one step
        self.isInSwamp = isInSwamp
        self.isTerminal = isTerminal
        self.swampPenalty = obstaclePenalty
        self.terminalReward = terminalReward

    def __call__(self, state, action, agentID):
        reward = self.actionCost
        if self.isInSwamp(state, agentID):
            reward+=self.swampPenalty
        if self.isTerminal(state, agentID):
            reward += self.terminalReward
        return reward


