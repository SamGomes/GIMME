import math
import copy
import random
import traceback


class InteractionsProfile(object):

	def __init__(self, dimensions = None):
		# if(dimensions != None):
		# 	if(type(dimensions)== dict):
		# 		self.dimensions = dimensions
		# 	elif(type(dimensions)== list):
		# 		self.unflatten(dimensions)
		# else:
		# 	self.dimensions = {}

		self.dimensions = {} if dimensions == None else dimensions
		self.dimensionality = len(self.dimensions)
		# self.normalize()

	def reset(self):
		for key in self.dimensions:
			self.dimensions[key] = 0
		return self

	def init(self):
		return self.reset()#.normalize()

	def generateCopy(self):
		keys = list(self.dimensions.keys())
		newVar = type(self)(copy.copy(self.dimensions))
		for key in keys:
			newVar.dimensions[key] = self.dimensions[key]
		return newVar

	def normalize(self):
		return self.normalization(self)

	def normalized(self):
		clone = self.generateCopy() 
		return self.normalization(clone)
	
	def normalization(self, profile):
		if(len(profile.dimensions)>1):
			total = 0
			for key in profile.dimensions:
				total += profile.dimensions[key]
			if(total==0):
				for key in profile.dimensions:
					profile.dimensions[key] = 1/len(profile.dimensions)
			else:
				for key in profile.dimensions:
					profile.dimensions[key] = profile.dimensions[key]/total
		return profile



	def randomize(self):
		return self.randomization(self)

	def randomized(self):
		clone = self.generateCopy() 
		return self.randomization(clone)

	def randomization(self, profile):
		profile.reset()
		for key in profile.dimensions:
			profile.dimensions[key] = random.uniform(0.0, 1.0)
		# profile.normalize()
		return profile


	def sqrDistanceBetween(self, profileToTest):
		cost = self.generateCopy()
		cost.reset()
		if(len(cost.dimensions) != len(profileToTest.dimensions)):
			traceback.print_stack()
			print(cost.dimensions)
			print(profileToTest.dimensions)
			raise Exception("[ERROR] Could not compute distance between profiles in different sized spaces. Execution aborted.")

		for key in cost.dimensions:
			cost.dimensions[key] = abs(self.dimensions[key] - profileToTest.dimensions[key])

		total = 0
		for key in cost.dimensions:
			cost.dimensions[key] = pow(cost.dimensions[key], 2)
			total += cost.dimensions[key]

		return total


	def distanceBetween(self, profileToTest):
		numDims = len(profileToTest.dimensions)
		return self.sqrDistanceBetween(profileToTest)**(1/float(numDims)) 


	def flattened(self):
		return [dim for dim in self.dimensions.values()]



	def unflattenFunc(self, profile, array):
		i = 0
		for key in profile.dimensions.keys():
			profile.dimensions[key] = array[i]
			i += 1
		return profile


	def unflatten(self, array):
		return self.unflattenFunc(self, array)

	def unflattened(self, array):
		clone = self.generateCopy() 
		return self.unflattenFunc(clone, array)