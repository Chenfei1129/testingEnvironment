import numpy as np
import random


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, resetState, chooseAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.chooseAction = chooseAction

    def __call__(self, policy):
            
        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [choose(actionDist) for choose, actionDist in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


