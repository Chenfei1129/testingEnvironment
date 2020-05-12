import matplotlib.pyplot as plt
import numpy as np

class FindCenterPointState():
	def __init__(self, background):
		self.background = background

	def __call__(self, grid):
		x = [i*self.background[0]/(grid[0]) for i in range(0,grid[0]+1)]
		y = [i*self.background[1]/(grid[1]) for i in range(0,grid[1]+1)]
		xCenter = [(x[i] + x[i+1])/2 for i in range(0,grid[0])]
		yCenter = [(y[i] + y[i+1])/2 for i in range(0,grid[1])]
		return x, y, xCenter, yCenter

class FindCenterPointValue():
	def __init__(self, valueFunction):
		self.valueFunction = valueFunction

	def __call__(self, xCenter, yCenter):
		centerPointValue = []
		for y in yCenter:
		     centerPointValue.append([self.valueFunction([x,y]) for x in xCenter])
		return centerPointValue
		

class DrawValueMap():
	def __init__(self, findCenterPointState, findCenterPointValue, background):
		self.findCenterPointValue = findCenterPointValue
		self.findCenterPointState = findCenterPointState
		self.background = background#delete

	def __call__(self, grid):
		x, y, xCenter, yCenter = self.findCenterPointState(grid)
		centerPointValue = self.findCenterPointValue(xCenter, yCenter)

		x, y = np.meshgrid(x, y)
		value = np.array(centerPointValue)
		plt.pcolormesh(x, y, value)
		plt.colorbar()
		plt.show()
