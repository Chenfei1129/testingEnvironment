import numpy as np
import random
import copy 


class RandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self,state):
        actionDist = {tuple(action): 1 / len(self.actionSpace) for action in self.actionSpace}
        print(actionDist)
        return actionDist




def main():
    actionSpace = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]])
    actionDist = {tuple(action): 1 / len(actionSpace) for action in actionSpace}
    print (actionDist)

   

if __name__ == '__main__':
    main()
