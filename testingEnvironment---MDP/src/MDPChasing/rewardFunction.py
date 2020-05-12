
import numpy as np
		
class RewardFunctionSingleAgent():
    def __init__(self, actionCost, terminalReward, swampPenalty, isTerminal, isInSwamp):
        self.actionCost = actionCost
        self.swampPenalty = swampPenalty
        self.terminalReward = terminalReward
        self.isInSwamp = isInSwamp
        self.isTerminal = isTerminal


    def __call__(self, state, action, newstate):
        reward = self.actionCost
        if self.isInSwamp(state)==True: 
            reward+=self.swampPenalty
        if self.isTerminal(state)==True:
            reward += self.terminalReward
        return reward

class RewardFunctionAllAgent():
    def __init__(self, allRewardFunctions):
        self.allRewardFunctions = allRewardFunctions

    def __call__(self, allStates, allActions, allNewStates):
        allReward = [self.allRewardFunctions[i](allStates[i], allActions[i], allNewStates[i]) for i in range(len(self.allRewardFunctions))]
        return allReward

