import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it

import pygame as pg
from pygame.color import THECOLORS

from src.MDPChasing.policies import RandomPolicy
from src.visualization.drawDemo import DrawBackground, DrawCircleOutside, DrawState, ChaseTrialWithTraj, InterpolateState
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.MDPChasing.transitionFunction import TransitForNoPhysics, StayInBoundaryByReflectVelocity, Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal, TransitionWithNoise, \
    IsTerminalSingleState, IsInObstacle, IsInObstacleSingleState
from src.trajectory import SampleTrajectory

def main():

    # MDP Env
    xBoundary = [0, 600]
    yBoundary = [0, 600]
    xObstacle=[300, 400]
    yObstacle=[300, 400]
    noise = [1, 1]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transitionWithNoise = TransitionWithNoise(noise)

    transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity,transitionWithNoise )
    numOfAgent = 2
    xBoundaryReset = [000, 300]
    yBoundaryReset = [200, 200]
    resetState = Reset(xBoundaryReset, yBoundaryReset, numOfAgent)

    actionSpace = np.array([ [10, 0],[-10, 0],[-10, -10],[10, 10],[0, 10],[0, -10],[-10, 10],[10, -10]])
    randomPolicy = RandomPolicy(actionSpace)
    policy = lambda state: [randomPolicy(state), randomPolicy(state)]
    
    minDistance = 50
    target = [200, 200]
    isTerminal= IsTerminalSingleState(minDistance, target)
    maxRunningSteps = 100
   
    
    chooseAction= [maxFromDistribution, sampleFromDistribution] # or maFromDistribution variable

    sampleTrajecoty = SampleTrajectory(maxRunningSteps, transit, isTerminal, resetState, chooseAction)
   
    trajectories = [sampleTrajecoty(policy) for _ in range(10)]

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateObstacle',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    

    # generate demo image
    screenWidth = 600
    screenHeight = 600
    screen = pg.display.set_mode((screenWidth, screenHeight))
    screenColor = THECOLORS['black']
    xBoundary = [0, 600]
    yBoundary = [0, 600]
    lineColor = THECOLORS['white']
    lineWidth = 4
    xObstacle=[300,400]
    yObstacle=[300,400]
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth,xObstacle,yObstacle)

    fps=40
    circleColorSpace = np.array([[0, 0, 255], [0, 255, 0]])
    circleSize = 10
    positionIndex = [0, 1]
    
    saveImage = True
    imageSavePath = os.path.join(trajectoryDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    trajectoryParameters = 'obstacle'
    imageFolderName = str(trajectoryParameters)

    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    agentIdsToDraw = list(range(2))
    drawState = DrawState(fps, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground)
    
    numFramesToInterpolate = 3
    interpolateState = InterpolateState(numFramesToInterpolate, transit)

    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep)
   
    print(len(trajectories))
    [chaseTrial(trajectory) for trajectory in trajectories]
    pg.quit()

    
if __name__ == '__main__':
    main()

