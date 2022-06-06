import copy
import json
import numpy
import pandas as pd

from abc import ABC, abstractmethod

from GIMMECore.ModelBridge.TaskModelBridge import TaskModelBridge
from ..PlayerStructs import *

from sklearn import linear_model, neighbors

class RegressionAlg(ABC):

	def __init__(self, playerModelBridge):
		self.playerModelBridge = playerModelBridge

		self.completionPerc = 0.0

	@abstractmethod
	def predict(self, profile, playerId):
		pass

	def isTabular(self):
		return False

	# instrumentation
	def getCompPercentage(self):
		return self.completionPerc


# ---------------------- KNNRegression ---------------------------
class KNNRegression(RegressionAlg):

	def __init__(self, playerModelBridge, numberOfNNs):
		super().__init__(playerModelBridge)
		self.numberOfNNs = numberOfNNs

	def distSort(self, elem):
		return elem.dist

	def creationTimeSort(self, elem):
		return elem.creationTime

	def predict(self, profile, playerId):
		# import time
		# startTime = time.time()

		pastModelIncs = self.playerModelBridge.getPlayerStatesDataFrame(playerId).getAllStates().copy()
		pastModelIncsSize = len(pastModelIncs)

		predictedState = PlayerState(profile = profile, characteristics = PlayerCharacteristics())
	
		for modelInc in pastModelIncs:
			modelInc.dist = profile.sqrDistanceBetween(modelInc.profile)

		pastModelIncs = sorted(pastModelIncs, key=self.distSort)

		numberOfIterations = min(self.numberOfNNs, len(pastModelIncs))
		pastModelIncs = pastModelIncs[:numberOfIterations]

		triangularNumberOfIt = sum(range(numberOfIterations + 1))
		for i in range(numberOfIterations):

			self.completionPerc = i/ numberOfIterations

			currState = pastModelIncs[i]
			pastCharacteristics = currState.characteristics
			ratio = (numberOfIterations - i)/triangularNumberOfIt

			predictedState.characteristics.ability += pastCharacteristics.ability * ratio
			predictedState.characteristics.engagement += pastCharacteristics.engagement * ratio

		# executionTime = (time.time() - startTime)
		# print('Execution time in seconds: ' + str(executionTime))
		
		return predictedState

# ---------------------- KNNRegressionSKLearn ---------------------------
class KNNRegressionSKLearn(RegressionAlg):

	def __init__(self, playerModelBridge, numberOfNNs):
		super().__init__(playerModelBridge)
		self.numberOfNNs = numberOfNNs


	def predict(self, profile, playerId):
		# import time
		# startTime = time.time()

		pastModelIncs = self.playerModelBridge.getPlayerStatesDataFrame(playerId).getAllStatesFlatten()

		lenPMI = len(pastModelIncs['profiles'])
		
		numberOfNNs = self.numberOfNNs
		if(lenPMI < self.numberOfNNs):
			if(lenPMI==0):
				return PlayerState(profile = profile, characteristics = PlayerCharacteristics(ability = 0.5, engagement = 0.5))
			numberOfNNs = lenPMI

		profData = profile.flattened()
		prevProfs = pastModelIncs['profiles']
		

		self.regrAb = neighbors.KNeighborsRegressor(numberOfNNs, weights="distance")
		self.regrAb.fit(prevProfs, pastModelIncs['abilities'])
		predAbilityInc = self.regrAb.predict([profData])[0]

		self.regrEng = neighbors.KNeighborsRegressor(numberOfNNs, weights="distance")
		self.regrEng.fit(prevProfs, pastModelIncs['engagements'])
		predEngagement = self.regrEng.predict([profData])[0]

		predState = PlayerState(profile = profile, characteristics = PlayerCharacteristics(ability = predAbilityInc, engagement = predEngagement))
		

		self.completionPerc = 1.0

		# executionTime = (time.time() - startTime)
		# print('Execution time in seconds: ' + str(executionTime))

		return predState

# ---------------------- LinearRegressionSKLearn ---------------------------
class LinearRegressionSKLearn(RegressionAlg):

	def __init__(self, playerModelBridge):
		super().__init__(playerModelBridge)

	def predict(self, profile, playerId):
		
		pastModelIncs = self.playerModelBridge.getPlayerStatesDataFrame(playerId).getAllStatesFlatten()

		if(len(pastModelIncs['profiles'])==0):
			return PlayerState(profile = profile, characteristics = PlayerCharacteristics(ability = 0.5, engagement = 0.5))

		profData = profile.flattened()

		prevProfs = pastModelIncs['profiles']

		regr = linear_model.LinearRegression() 
		regr.fit(prevProfs, pastModelIncs['abilities'])
		predAbilityInc = regr.predict([profData])[0]

		regr.fit(prevProfs, pastModelIncs['engagements'])
		predEngagement = regr.predict([profData])[0]

		predState = PlayerState(profile = profile, characteristics = PlayerCharacteristics(ability = predAbilityInc, engagement = predEngagement))
		
		self.completionPerc = 1.0

		return predState

# ---------------------- SVMRegressionSKLearn ---------------------------
class SVMRegressionSKLearn(RegressionAlg):

	def __init__(self, playerModelBridge):
		super().__init__(playerModelBridge)

	def predict(self, profile, playerId):
		
		pastModelIncs = self.playerModelBridge.getPlayerStatesDataFrame(playerId).getAllStatesFlatten()

		if(len(pastModelIncs['profiles'])==0):
			return PlayerState(profile = profile, characteristics = PlayerCharacteristics(ability = 0.5, engagement = 0.5))

		profData = profile.flattened()

		prevProfs = pastModelIncs['profiles']

		regr = svm.SVR()
		regr.fit(prevProfs, pastModelIncs['abilities'])
		predAbility = regr.predict([profData])[0]

		regr.fit(prevProfs, pastModelIncs['engagements'])
		predEngagement = regr.predict([profData])[0]

		predState = PlayerState(profile = profile, characteristics = PlayerCharacteristics(ability = predAbility, engagement = predEngagement))
		
		self.completionPerc = 1.0

		return predState

# ---------------------- DecisionTreesRegression ---------------------------
class DecisionTreesRegression(RegressionAlg):

	def __init__(self, playerModelBridge):
		super().__init__(playerModelBridge)

	def predict(self, profile, playerId):
		pass


# ---------------------- NeuralNetworkRegression ---------------------------
class NeuralNetworkRegression(RegressionAlg):

	def __init__(self, playerModelBridge):
		super().__init__(playerModelBridge)

	def predict(self, profile, playerId):
		pass


# ---------------------- Tabular Agent Synergy Method -------------------------------------
class TabularAgentSynergies(RegressionAlg):

	def __init__(self, playerModelBridge, taskModelBridge):
		super().__init__(playerModelBridge)
		
		self.taskModelBridge = taskModelBridge
		tempTable = pd.read_csv('synergyTable.txt', sep=",", dtype={'agent_1': object, 'agent_2': object}) 
		synergyTable = tempTable.pivot_table(values='synergy', index='agent_1', columns='agent_2') 
		
		self.synergyMatrix = synergyTable.to_numpy()
		self.synergyMatrix[numpy.isnan(self.synergyMatrix)] = 0
		self.symmetrize(self.synergyMatrix)

		# tempTable = pd.read_csv('taskTable.txt', sep=',', dtype={'task': object, 'agent': object})
		# taskTable = tempTable.pivot_table(values='synergy', index='task', columns='agent')

		# self.taskMatrix = taskTable.to_numpy()
		# self.taskMatrix[numpy.isnan(self.taskMatrix)] = 0

	def symmetrize(self, table):
		return table + table.T - numpy.diag(table.diagonal())


	def isTabular(self):
		return True

	def predict(self, profile, playerId):
		firstPlayerPreferencesInBinary = ''
		for dim in profile.dimensions:
			firstPlayerPreferencesInBinary += str(round(profile.dimensions[dim]))

		secondPlayerPreferences = self.playerModelBridge.getPlayerPreferencesEst(playerId)
		secondPlayerPreferenceInBinary = ''
		for dim in secondPlayerPreferences.dimensions:
			secondPlayerPreferenceInBinary += str(round(secondPlayerPreferences.dimensions[dim]))

		firstPlayerPreferencesIndex = int(firstPlayerPreferencesInBinary, 2)
		secondPlayerPreferencesIndex = int(secondPlayerPreferenceInBinary, 2)

		return self.synergyMatrix[firstPlayerPreferencesIndex][secondPlayerPreferencesIndex]

	# either this, or find here the best task
	def predictTasks(self, taskId, playerId):
		playerPreferences = self.playerModelBridge.getPlayerPreferencesEst(playerId)
		playerPreferenceInBinary = ''
		for dim in playerPreferences.dimensions:
			playerPreferenceInBinary += str(round(playerPreferences.dimensions[dim]))

		taskProfile = self.taskModelBridge.getTaskInteractionsProfile(taskId)
		taskProfileInBinary = ''
		for dim in taskProfile.dimensions:
			taskProfileInBinary += str(round(taskProfile.dimensions[dim]))

		playerPreferenceIndex = int(playerPreferenceInBinary, 2)
		taskProfileIndex = int(taskProfileInBinary, 2)

		return self.taskMatrix[playerPreferenceIndex][taskProfileIndex]



# X = pd.read_csv('synergyTable.txt', sep=",") 
# >>> X.pivot_table(values='synergy', index='agent1', columns='agent2')
# agent2  (00)  (01)  (10)  (11)
# agent1
# (00)       1     0     0     0
# (01)       0     1     0     0
# (10)       0     0     1     0
# (11)       0     0     0     1
# >>> dX = X.pivot_table(values='synergy', index='agent1', columns='agent2')
# >>> dX
# agent2  (00)  (01)  (10)  (11)
# agent1
# (00)       0     0     0     0
# (01)       0     0     0     0
# (10)       0     0     0     0
# (11)       0     0     0     0
# >>> dX['(00)']
# agent1
# (00)    0
# (01)    0
# (10)    0
# (11)    0