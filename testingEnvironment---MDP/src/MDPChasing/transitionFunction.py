
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
 

class MultiAgentTransitionInGeneral():
    def __init__(self, allTransitions):
        self.allTransitions = allTransitions

    def __call__(self, allStates, allActions):
        allNewStates = [self.allTransitions[i](allStates, allActions[i]) for i in range(len(allTransitions))]


class MultiAgentTransitionInSwampWorld():
    def __init__(self, multiAgentTransitionInGeneral, terminalPosition):
        self.multiAgentTransitionInGeneral = MultiAgentTransitionInGeneral
        self.terminalPosition = terminalPosition

    def __call__(self, state, action):
        allStates = [state, terminalPosition]
        allActions = [action, [0,0]]
        return self.multiAgentTransitionInGeneral(allStates, allActions)


class SingleAgentTransitionInSwampWorld():
    def __init__(self, transitionWithNoise, stayInBoundaryByReflectVelocity, isTerminal):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity
        self.transitionWithNoise = transitionWithNoise
        self.isTerminal = isTerminal

    def __call__(self, state, action):
        if self.isTerminal(state)==True:
            newState = state
        else:
            newState = np.array(state) + np.array(action)
            newStateCheckBoundary, newActionCheckBoundary = self.stayInBoundaryByReflectVelocity(newState, action)
            newState = newStateCheckBoundary
            newaction = newActionCheckBoundary 
            finalNewState = self.transitionWithNoise(newState)
            return finalNewState


class TransitionWithNoise():
    def __init__(self, state):          
        self.state = state

    def __call__(self, noise):
        x = np.random.normal(state[0], self.noise[0])
        y = np.random.normal(state[1], self.noise[1])
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
    def __init__(self, swamp):
        self.swamp = swamp

    def __call__(self, state):
        inOrNot = [ (state[0] >= xEachSwamp[0] and state[0] <= xEachSwamp[1] and state[1] >= yEachSwamp[0] and state[1] <= yEachSwamp[1])
             for xEachSwamp, yEachSwamp in self.swamp]
        if True in inOrNot:
            return True
        else:
            return False
        
 class IsTerminal():
    def __init__(self, minDistance, terminalPosition):
        self.minDistance = minDistance
        self.terminalPosition = terminalPosition

    def __call__(self, state):       
        distanceToTerminal = np.array([np.linalg.norm(np.array(self.terminalPosition) - np.array(state), ord=2)] )          
        return (distanceToTerminal<= self.minDistance)


