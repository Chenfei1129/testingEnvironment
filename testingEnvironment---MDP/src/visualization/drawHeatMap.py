class FindCenterPointValue():
	def __init__(self, valueFunction, background):
		self.background = background
		self.valueFunction = valueFunction

	def __call__(self, grid):
		x = [i*self.background[0]/(self.grid[0])for i in range(self.grid[0])]
		y = [i*self.background[1]/(self.grid[1]) for i in range(self.grid[1])]
		xCenter = [(x[i] + x[i+1])/2 for i in range(self.grid[0])]
		yCenter = [(y[i] + y[i+1])/2 for i in range(self.grid[1])]
		for x in xCenter:
		    centerPointValue = [valueFunction([x,y]) for y in yCenter]
		return centerPointValue


		
class DrawValueMap():
	def __init__(self, centerPointValue, background, grid):
		self.centerPointValue = centerPointValue
		self.background = background
		self.grid = grid

	def __call__(self):
		x = [i*self.background[0]/(self.grid[0])for i in range(self.grid[0])]
		y = [i*self.background[1]/(self.grid[1]) for i in range(self.grid[1])]
		x, y = np.meshgrid(x, y)
		value = np.array(self.centerPointValue)
		plt.pcolormesh(x, y, value)
		plt.colorbar()
		plt.show()
