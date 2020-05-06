class FindCenterPointValue():
	def __init__(self, valueFunction, background):## whole statespace
		self.background = background
		self.valueFunction = valueFunction

	def __call__(self, grid):
		x = [i*self.background[0]/(grid[0])for i in range(grid[0]+1)]
		y = [i*self.background[1]/(grid[1]) for i in range(grid[1]+1)]
		xCenter = [(x[i] + x[i+1])/2 for i in range(grid[0])]
		yCenter = [(y[i] + y[i+1])/2 for i in range(grid[1])]
		centerPointValue = []
		for y in yCenter:
		     centerPointValue.append([valueFunction([x,y]) for x in xCenter])
		return centerPointValue


		

class DrawValueMap():
	def __init__(self, centerPointValue, background, grid):
		self.centerPointValue = centerPointValue
		self.background = background
		self.grid = grid

	def __call__(self):
		x = [i*self.background[0]/(self.grid[0])for i in range(self.grid[0]+1)]
		y = [i*self.background[1]/(self.grid[1]) for i in range(self.grid[1]+1)]
		print(x,y)
		x, y = np.meshgrid(x, y)
		value = np.array(self.centerPointValue)
		plt.pcolormesh(x, y, value)
		plt.colorbar()
		plt.show()
