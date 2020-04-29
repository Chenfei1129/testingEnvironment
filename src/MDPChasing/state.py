import numpy as np


class GetAgentPosFromState:
    def __init__(self, agentId):
        self.agentId = agentId
 
    def __call__(self, state):
        state = np.asarray(state)
        agentPos = state[self.agentId]
        return agentPos

