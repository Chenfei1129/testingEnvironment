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
import math
import pygame as pg
from pygame.color import THECOLORS

from src.MDPChasing.state import GetAgentPosFromState 
from src.MDPChasing.envNoPhysics import Reset, TransitMultiAgent, AddAgentStateGaussianNoise, StayInBoundaryByReflectVelocity, IsTerminal, IsInSwamp
from src.MDPChasing.reward import RewardFunctionCompete
from src.analyticGeometryFunctions import computeAngleBetweenVectors
from src.MDPChasing.policies import stationaryAgentPolicy, RandomPolicy, HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy
from src.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution
from src.trajectory import SampleTrajectory
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.visualization.drawDemo import DrawBackground, DrawCircleOutside, DrawState, ChaseTrialWithTraj, InterpolateStateInVisualization

def main():

    # transit for sheep
    sheepId = 0
    wolfId = 1
    getSheepPos = GetAgentPosFromState(sheepId)
    getWolfPos = GetAgentPosFromState(wolfId)
    killzoneRadius = 50
    isTerminal = IsTerminal(killzoneRadius, getSheepPos, getWolfPos) 
    
    xBoundary = [0, 600]
    yBoundary = [0, 600]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    
    stdStateNoise = [1, 1]
    addAgentStateGaussianNoise = AddAgentStateGaussianNoise(stdStateNoise)
    
    numFramesToInterpolate = 2
    transitMultiAgent = TransitMultiAgent(numFramesToInterpolate, stayInBoundaryByReflectVelocity, isTerminal, addAgentStateGaussianNoise)
    
    numActionDirections = 8
    actionDirections = [(np.cos(directionId * 2 * math.pi / numActionDirections), np.sin(directionId * 2 * math.pi / numActionDirections)) for directionId in range(numActionDirections)]
    wolfSpeedRatio = 60 
    wolfActionSpace = list(map(tuple, np.array(actionDirections) * wolfSpeedRatio))
    wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(wolfActionSpace, getWolfPos, getSheepPos, computeAngleBetweenVectors)
    
    transitSheep = lambda state, sheepAction: transitMultiAgent(state, [sheepAction, maxFromDistribution(wolfPolicy(state))]) 
    
    # reward for sheep
    maxRunningSteps = 50
    timeRewardSheep = 1 / maxRunningSteps
    terminalPenaltySheep = -1
    swampPenalty = -100

    xSwamp = [300, 400]
    ySwamp = [300, 400]
    swamp = [xSwamp, ySwamp]
    isInSwampSheep = IsInSwamp(swamp, sheepId)

    rewardSheep = RewardFunctionCompete(timeRewardSheep, terminalPenaltySheep, swampPenalty, isTerminal, isInSwampSheep) 

    # sample trajectory
    numOfAgent = 2
    reset = Reset(xBoundary, yBoundary, numOfAgent)
 
    chooseAction= [maxFromDistribution, sampleFromDistribution] # or maFromDistribution variable

    sampleTrajecoty = SampleTrajectory(maxRunningSteps, transitMultiAgent, isTerminal, reset, chooseAction)

    # policy for sample trajectory
    # sheep policy from algorithm, e.g., QLearning, dqn, ddpg
    # here randomPolicy just for demo
    sheepSpeedRatio = 90
    sheepActionSpace = list(map(tuple, np.array(actionDirections) * sheepSpeedRatio))
    sheepPolicy = RandomPolicy(sheepActionSpace)
    #sheepPolicy = QLearning
    #sheepPolicy = ActorNetWorkAfterTrainFromActor-Critic

    policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]
    trajectories = [sampleTrajecoty(policy) for _ in range(10)]

    # save trajectory
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'example',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    
    trajectorySaveExtension = '.pickle'
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'killzoneRadius': killzoneRadius}
    generateTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectorySaveExtension, fixedParameters)
    parametersForTrajectoryPath = {'algorithmAgentId': sheepId}
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    
    saveToPickle(trajectories, trajectorySavePath)

    # visualize trajectory
    screenWidth = 600
    screenHeight = 600
    screen = pg.display.set_mode((screenWidth, screenHeight))
    screenColor = THECOLORS['black']
    lineColor = THECOLORS['white']
    lineWidth = 4
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth, xSwamp, ySwamp)

    fps = 15
    circleColorSpace = np.array([[0, 255, 0], [255, 0, 0]])
    circleSize = 10
    positionIndex = [0, 1]
    
    saveImage = True
    imageSavePath = os.path.join(trajectoryDirectory, 'pic')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = 'sheep'
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)

    agentIdsToDraw = list(range(numOfAgent))
    drawState = DrawState(fps, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground)
    
    interpolateStateInVisualization = InterpolateStateInVisualization(numFramesToInterpolate, stayInBoundaryByReflectVelocity)

    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1 
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateStateInVisualization, actionIndexInTimeStep)
   
    [chaseTrial(trajectory) for trajectory in trajectories]
    pg.quit()

    
if __name__ == '__main__':
    main()

