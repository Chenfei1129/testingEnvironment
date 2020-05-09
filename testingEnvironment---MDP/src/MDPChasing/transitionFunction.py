import numpy as np 
import itertools as it

class Reset():
    def __init__(self, xBoundary, yBoundary, numOfAgent):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.numOfAgnet = numOfAgent

    def __call__(self):
        xMin, xMax = self.xBoundary
        yMin, yMax = self.yBoundary
        initState = [[np.random.uniform(xMin, xMax),
                      np.random.uniform(yMin, yMax)]
                     for _ in range(self.numOfAgnet)]
        return initState

class FixedReset():
    def __init__(self, initPositionList):
        self.initPositionList = initPositionList

    def __call__(self, trialIndex):
        initState = self.initPositionList[trialIndex]
        return initState

class TransitInSwampWorld():# generic pass a list of transition functions.
    def __init__(self, stayInBoundaryByReflectVelocity, transitionWithNoise):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity
        self.transitionWithNoise = transitionWithNoise

    def __call__(self, state, action):
        newState = np.array(state) + np.array(action)        
        checkedNewStateAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newState, action)]
        newState, newAction = list(zip(*checkedNewStateAndVelocities))
        finalNewState = [self.transitionWithNoise(singleState) for singleState in newState]
        return finalNewState

class TransitionWithNoise():
    def __init__(self, standardDeviation):          
        self.standardDeviation = standardDeviation

    def __call__(self, mu):
        x = np.random.normal(mu[0], self.standardDeviation[0])
        y = np.random.normal(mu[1], self.standardDeviation[1])
        result = [x, y]
        return result

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
        checkedPosition = [adjustedX, adjustedY]
        checkedVelocity = [adjustedVelX, adjustedVelY]
        return checkedPosition, checkedVelocity


class CheckBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        xPos, yPos = position
        if xPos >= self.xMax or xPos <= self.xMin:
            return False
        elif yPos >= self.yMax or yPos <= self.yMin:
            return False
        return True


class IsInSwamp():
    def __init__(self, swamp, agentID):
        self.swamp = swamp
        self.agentID = agentID

    def __call__(self, state):
        inOrNot = [ (state[self.agentID][0] >= xEachSwamp[0] and state[self.agentID][0] <= xEachSwamp[1] and state[self.agentID][1] >= yEachSwamp[0] and state[self.agentID][1] <= yEachSwamp[1])
             for xEachSwamp, yEachSwamp in self.swamp]
        if True in inOrNot:
            return True
        else:
            return False

class IsTerminal():
    def __init__(self, minDistance, terminalPosition, agentID):
        self.minDistance = minDistance
        self.terminalPosition = terminalPosition
        self.agentID = agentID

    def __call__(self, state):       
        L2Normdistance = np.array([np.linalg.norm(np.array(self.terminalPosition) - np.array(state[self.agentID]), ord=2)] )          
        return (L2Normdistance <= self.minDistance)

class IsTerminalAll():
    def __init__(self, minDistance, terminalPosition, isTerminal):
        self.minDistance = minDistance
        self.terminalPosition = terminalPosition
        self.isTerminal = isTerminal

    def __call__(self, state):     
        
        isTerminalOrNot = [self.isTerminal(state) for agentID in range(len(state)-1)]
        return (True in isTerminalOrNot)

