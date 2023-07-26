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

	@abstractmethod
	def isTabular(self):
		pass

	# If returns true, must implement a groupPredict() method
	def isGroupPredict(self):  
		return False
	
	def groupPredict(self, groupIds):
		pass
	
	# instrumentation
	def getCompPercentage(self):
		return self.completionPerc


# ---------------------- Personality Diversity ---------------------------
class DiversityValueAlg(RegressionAlg):
	#  Consider the task preferences of students in addition to team diversity. People with the same personality can still have different preferences
	#  Diversity weight is the value determined by the teacher (0 = aligned, 1 = diverse)
	def __init__(self, playerModelBridge, diversityWeight):
		super().__init__(playerModelBridge)
		self.diversityWeight = diversityWeight

	def predict(self, profile, playerId):
		return 0
	
	def isTabular(self):
		return False
	
	def isGroupPredict(self):
		return True
	
	def getPersonalitiesListFromPlayerIds(self, groupIds):
		personalities = []  # list of PlayerPersonality objects

		for playerId in groupIds:
			personality = self.playerModelBridge.getPlayerPersonality(playerId)
			if personality:
				personalities.append(personality)

		return personalities
	
	def getTeamPersonalityDiveristy(self, personalities):
		if len(personalities) <= 0:
			return -1
		
		diversity = -1

		if isinstance(personalities[0], PersonalityMBTI):
			diversity = PersonalityMBTI.getTeamPersonalityDiversity(personalities)

		return diversity


	def groupPredict(self, groupIds):
		personalities = self.getPersonalitiesListFromPlayerIds(groupIds)  # list of PlayerPersonality objects
		diversity = self.getTeamPersonalityDiveristy(personalities)

		# inverse of distance squared
		# lower distance = higher quality
		distance = abs(diversity - self.diversityWeight)
		
		if distance == 0.0:
			return 1.0
		
		return 1.0 / (distance * distance)


# class DiversityLogarithmicCentroidDistance(RegressionAlg):
# 	def __init__(self, playerModelBridge):
# 		super().__init__(playerModelBridge)

# ---------------------- Regression Based Characteristic Functions ---------------------------
class RegCoalitionValueAlg(RegressionAlg):
	def __init__(self, playerModelBridge, qualityWeights):
		super().__init__(playerModelBridge)
		self.qualityWeights = PlayerCharacteristics(ability = 0.5, engagement = 0.5) if qualityWeights == None else qualityWeights

	def isTabular(self):
		return False


# ---------------------- KNNRegression ---------------------------
class KNNRegression(RegCoalitionValueAlg):

	def __init__(self, playerModelBridge, numberOfNNs, qualityWeights = None):
		super().__init__(playerModelBridge, qualityWeights)
		self.numberOfNNs = numberOfNNs


	def calcQuality(self, state):
		return self.qualityWeights.ability*state.characteristics.ability + self.qualityWeights.engagement*state.characteristics.engagement

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
		self.state = predictedState

		return self.calcQuality(predictedState)

# ---------------------- KNNRegressionSKLearn ---------------------------
class KNNRegressionSKLearn(RegCoalitionValueAlg):

	def __init__(self, playerModelBridge, numberOfNNs, qualityWeights = None):
		super().__init__(playerModelBridge, qualityWeights)
		self.numberOfNNs = numberOfNNs

	def calcQuality(self, state):
		return self.qualityWeights.ability*state.characteristics.ability + self.qualityWeights.engagement*state.characteristics.engagement

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
		self.state = predState
		return self.calcQuality(predState)

# ---------------------- LinearRegressionSKLearn ---------------------------
class LinearRegressionSKLearn(RegCoalitionValueAlg):

	def __init__(self, playerModelBridge, qualityWeights = None):
		super().__init__(playerModelBridge, qualityWeights)


	def calcQuality(self, state):
		return self.qualityWeights.ability*state.characteristics.ability + self.qualityWeights.engagement*state.characteristics.engagement


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
		self.state = predState
		return self.calcQuality(predState)

# ---------------------- SVMRegressionSKLearn ---------------------------
class SVMRegressionSKLearn(RegCoalitionValueAlg):

	def __init__(self, playerModelBridge, qualityWeights = None):
		super().__init__(playerModelBridge, qualityWeights)


	def calcQuality(self, state):
		return self.qualityWeights.ability*state.characteristics.ability + self.qualityWeights.engagement*state.characteristics.engagement


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
		self.state = predState
		return self.calcQuality(predState)

# ---------------------- DecisionTreesRegression ---------------------------
class DecisionTreesRegression(RegCoalitionValueAlg):

	def __init__(self, playerModelBridge, qualityWeights = None):
		super().__init__(playerModelBridge, qualityWeights)

	def predict(self, profile, playerId):
		pass


# ---------------------- NeuralNetworkRegression ---------------------------
class NeuralNetworkRegression(RegCoalitionValueAlg):

	def __init__(self, playerModelBridge, qualityWeights = None):
		super().__init__(playerModelBridge, qualityWeights)

	def predict(self, profile, playerId):
		pass


# ---------------------- Tabular Characteristic Functions -------------------------------------
class TabularCoalitionValueAlg(RegressionAlg):
	def __init__(self, playerModelBridge):
		super().__init__(playerModelBridge)
		self.playerPrefEstimates = {}


	def isTabular(self):
		return True

	def getPlayerPreferencesEstimations(self):
	 	for player in self.playerIds:
	 		self.playerPrefEstimates[player] = self.playerModelBridge.getPlayerPreferencesEst(player)

# ---------------------- Tabular Agent Synergy Method -------------------------------------
class TabularAgentSynergies(TabularCoalitionValueAlg):

	def __init__(self, playerModelBridge, taskModelBridge, syntergyTablePath):
		super().__init__(playerModelBridge)


		self.taskModelBridge = taskModelBridge
		tempTable = pd.read_csv(syntergyTablePath, sep=",", dtype={'agent_1': object, 'agent_2': object})
		synergyTable = tempTable.pivot_table(values='synergy', index='agent_1', columns='agent_2')

		self.synergyMatrix = synergyTable.to_numpy()
		self.synergyMatrix[numpy.isnan(self.synergyMatrix)] = 0
		self.synergyMatrix = self.symmetrize(self.synergyMatrix)

		self.playerIds = self.playerModelBridge.getAllPlayerIds()

		# tempTable = pd.read_csv('taskTable.txt', sep=',', dtype={'task': object, 'agent': object})
		# taskTable = tempTable.pivot_table(values='synergy', index='task', columns='agent')

		# self.taskMatrix = taskTable.to_numpy()
		# self.taskMatrix[numpy.isnan(self.taskMatrix)] = 0

	def symmetrize(self, table):
		return table + table.T - numpy.diag(table.diagonal())

	def predict(self, profile, playerId):
		firstPlayerPreferencesInBinary = ''
		for dim in profile.dimensions:
			firstPlayerPreferencesInBinary += str(round(profile.dimensions[dim]))

		if (self.playerPrefEstimates == {}):
			self.getPlayerPreferencesEstimations()

		secondPlayerPreferences = self.playerPrefEstimates[playerId]
		secondPlayerPreferenceInBinary = ''
		for dim in secondPlayerPreferences.dimensions:
			secondPlayerPreferenceInBinary += str(round(secondPlayerPreferences.dimensions[dim]))

		firstPlayerPreferencesIndex = int(firstPlayerPreferencesInBinary, 2)
		secondPlayerPreferencesIndex = int(secondPlayerPreferenceInBinary, 2)

		return self.synergyMatrix[firstPlayerPreferencesIndex][secondPlayerPreferencesIndex]


	# either this, or find here the best task
	def predictTasks(self, taskId, playerId):
		playerPreferences = self.playerPrefEstimates[playerId]
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


