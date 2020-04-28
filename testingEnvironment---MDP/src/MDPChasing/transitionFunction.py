
import numpy as np
import statistics 
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
        return np.array(initState)

class FixedReset():
    def __init__(self, initPositionList):
        self.initPositionList = initPositionList

    def __call__(self, trialIndex):
        initState = self.initPositionList[trialIndex]
        return np.array(initState)

class TransitionWithNoise():
    def __init__(self, standardDeviation):          
        self.standardDeviation = standardDeviation

    def __call__(self, mu):
        x = np.random.normal(mu[0], self.standardDeviation[0])
        y = np.random.normal(mu[1], self.standardDeviation[1])
        result = [x, y]
        return np.array(result)

class TransitForNoPhysics():
    def __init__(self, stayInBoundaryByReflectVelocity, transitionWithNoise):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity
        self.transitionWithNoise = transitionWithNoise

    def __call__(self, state, action):
        newState = np.array(state) + np.array(action)        
        checkedNewStateAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newState, action)]
        newState, newAction = list(zip(*checkedNewStateAndVelocities))
        finalNewState = [self.transitionWithNoise(singleState) for singleState in newState]
        return np.array(finalNewState)

class IsTerminal():
    def __init__(self, minDistance, terminalPosition):
        self.minDistance = minDistance
        self.terminalPosition = terminalPosition

    def __call__(self, allAgentState):       
        L2Normdistance = np.array([np.linalg.norm(np.array(self.terminalPosition) - np.array(agentPosition), ord=2) 
            for agentPosition in allAgentState])
        isTerminalOrNot = [Distance <= self.minDistance for Distance in L2Normdistance]
        return isTerminalOrNot


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


class IsInObstacle():
    def __init__(self, Obstacle):
        self.Obstacle = Obstacle

    def __call__(self, state):
        inOrNot = [ (state[0] >= xEachObstacle[0] and state[0] <= xEachObstacle[1] and state[1] >= yEachObstacle[0] and state[1] <= yEachObstacle[1])
             for xEachObstacle, yEachObstacle in self.Obstacle]
        return inOrNot


class IsInObstacleSingleState():
    def __init__(self, Obstacle):
        self.Obstacle = Obstacle

    def __call__(self, state):
        inOrNot = [ (state[0] >= xEachObstacle[0] and state[0] <= xEachObstacle[1] and state[1] >= yEachObstacle[0] and state[1] <= yEachObstacle[1])
             for xEachObstacle, yEachObstacle in self.Obstacle]
        return sum(np.array(inOrNot))
        
class IsTerminalSingleState():
    def __init__(self, minDistance, terminalPosition):
        self.minDistance = minDistance
        self.terminalPosition = terminalPosition

    def __call__(self, state):       
        L2Normdistance = np.array([np.linalg.norm(np.array(self.terminalPosition) - np.array(state), ord=2)] )          
        isTerminalOrNot = [ L2Normdistance <= self.minDistance]
        return sum(np.array(isTerminalOrNot))

