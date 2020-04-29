import numpy as np
import itertools as it

class Reset():
    def __init__(self, xBoundary, yBoundary, numOfAgent, isLegal = lambda state: True):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.numOfAgnet = numOfAgent
        self.isLegal = isLegal

    def __call__(self):
        xMin, xMax = self.xBoundary
        yMin, yMax = self.yBoundary
        initState = [[np.random.uniform(xMin, xMax),
                      np.random.uniform(yMin, yMax)]
                     for _ in range(self.numOfAgnet)]
        while np.all([self.isLegal(state) for state in initState]) is False:
            initState = [[np.random.uniform(xMin, xMax),
                          np.random.uniform(yMin, yMax)]
                         for _ in range(self.numOfAgnet)] 
        return np.array(initState)

class TransitMultiAgent():
    def __init__(self, numFramesToInterpolate, stayInBoundaryByReflectVelocity, isTerminal, addAgentStateNoise):
        self.numFramesToInterpolate = numFramesToInterpolate
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity
        self.isTerminal = isTerminal
        self.addAgentStateNoise = addAgentStateNoise

    def __call__(self, state, action):
        actionForInterpolation = np.array(action) / (self.numFramesToInterpolate + 1)
        for frameIndex in range(self.numFramesToInterpolate + 1):
            newState = np.array(state) + np.array(actionForInterpolation)
            checkedNewStateAndAction = [self.stayInBoundaryByReflectVelocity(position, velocity) 
                    for position, velocity in zip(newState, actionForInterpolation)]
            nextState, nextActionForInterpolation = list(zip(*checkedNewStateAndAction))
            if self.isTerminal(nextState):
                break
            state = nextState
            actionForInterpolation = nextActionForInterpolation
        
        if not self.isTerminal(nextState):
            noisyNextState = np.array([self.addAgentStateNoise(nextAgentState) for nextAgentState in nextState])
            checkedNewStateAndAction = [self.stayInBoundaryByReflectVelocity(position, velocity) 
                    for position, velocity in zip(noisyNextState, actionForInterpolation)]
            nextState, nextActionForInterpolation = list(zip(*checkedNewStateAndAction))
        return nextState

class AddAgentStateGaussianNoise():
    def __init__(self, standardDeviation):          
        self.standardDeviation = standardDeviation

    def __call__(self, agentState):
        x = np.random.normal(agentState[0], self.standardDeviation[0])
        y = np.random.normal(agentState[1], self.standardDeviation[1])
        noisyAgentState = np.array([x, y])
        return noisyAgentState

class StayInBoundaryByReflectVelocity():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position, velocity):
        adjustedX, adjustedY = position
        adjustedVelX, adjustedVelY = velocity
        if position[0] >= self.xMax:
            adjustedX = 2 * self.xMax - position[0]
            adjustedVelX = -velocity[0]
        if position[0] <= self.xMin:
            adjustedX = 2 * self.xMin - position[0]
            adjustedVelX = -velocity[0]
        if position[1] >= self.yMax:
            adjustedY = 2 * self.yMax - position[1]
            adjustedVelY = -velocity[1]
        if position[1] <= self.yMin:
            adjustedY = 2 * self.yMin - position[1]
            adjustedVelY = -velocity[1]
        checkedPosition = np.array([adjustedX, adjustedY])
        checkedVelocity = np.array([adjustedVelX, adjustedVelY])
        return checkedPosition, checkedVelocity

class IsTerminal():
    def __init__(self, minDistance, getPreyPos, getPredatorPos):
        self.minDistance = minDistance
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos

    def __call__(self, state):
        preyPosition = self.getPreyPos(state)
        predatorPosition = self.getPredatorPos(state)
        L2Normdistance = np.linalg.norm(np.array(preyPosition) - np.array(predatorPosition), ord=2) 
        terminal = (L2Normdistance <= self.minDistance)
        return terminal

class IsInSwamp():
    def __init__(self, swamp, agentId):
        self.swamp = swamp
        self.agentId = agentId

    def __call__(self, state):
        agentState = state[self.agentId]
        inSwampsFlags = [(agentState[0] >= xEachSwamp[0] and agentState[0] <= xEachSwamp[1] and agentState[1] >= yEachSwamp[0] and agentState[1] <= yEachSwamp[1])
             for xEachSwamp, yEachSwamp in self.swamp]
        isInAnyOne = True in inSwampsFlags
        return isInAnyOne



