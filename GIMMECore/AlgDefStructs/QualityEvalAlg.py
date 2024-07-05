import copy
import json
import numpy
import pandas as pd

from abc import ABC, abstractmethod
from ..PlayerStructs import *

class QualityEvalAlg(ABC):
	
	def __init__(self, playerModelBridge):
		self.playerModelBridge = playerModelBridge

	@abstractmethod
	def evaluate(self, profile, groupPlayerIds):
		pass
	


# ---------------------- Group-Based Quality Evaluation ---------------------------
class GroupQualityEvalAlg(QualityEvalAlg):
	
	def __init__(self, playerModelBridge):
		super().__init__(playerModelBridge)

# Personality Diversity
class DiversityQualityEvalAlg(GroupQualityEvalAlg):
	#  Consider the task preferences of students in addition to team diversity. People with the same personality can still have different preferences
	#  Diversity weight is the value determined by the teacher (0 = aligned, 1 = diverse)
	def __init__(self, playerModelBridge, diversityWeight):
		super().__init__(playerModelBridge)
		self.diversityWeight = diversityWeight

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


	def evaluate(self, _, groupPlayerIds):
		personalities = self.getPersonalitiesListFromPlayerIds(groupPlayerIds)  # list of PlayerPersonality objects
		diversity = self.getTeamPersonalityDiveristy(personalities)

		# inverse of distance squared
		# lower distance = higher quality
		distance = abs(diversity - self.diversityWeight)
		
		if distance == 0.0:
			return 1.0
		
		return 1.0 / (distance * distance)



# ---------------------- Regression-Based Characteristic Functions ---------------------------
class RegQualityEvalAlg(QualityEvalAlg):
	
	def __init__(self, playerModelBridge, qualityWeights):
		super().__init__(playerModelBridge)
		self.qualityWeights = PlayerCharacteristics(ability = 0.5, engagement = 0.5) if qualityWeights == None else qualityWeights

class KNNRegQualityEvalAlg(RegQualityEvalAlg):

	def __init__(self, playerModelBridge, numberOfNNs, qualityWeights = None):
		super().__init__(playerModelBridge, qualityWeights)
		self.numberOfNNs = numberOfNNs


	def calcQuality(self, state):
		return self.qualityWeights.ability*state.characteristics.ability + self.qualityWeights.engagement*state.characteristics.engagement

	def distSort(self, elem):
		return elem.dist

	def evaluate(self, profile, groupPlayerIds):
		totalQuality = 0
		groupSize = len(groupPlayerIds)
		for playerId in groupPlayerIds:
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

			self.state = predictedState
			totalQuality += self.calcQuality(predictedState) / groupSize

		return totalQuality




# ---------------------- Tabular Characteristic Functions -------------------------------------
class TabQualityEvalAlg(QualityEvalAlg):
	
	def __init__(self, playerModelBridge):
		super().__init__(playerModelBridge)
		self.playerPrefEstimates = {}

	def getPlayerPreferencesEstimations(self):
	 	for player in self.playerIds:
	 		self.playerPrefEstimates[player] = self.playerModelBridge.getPlayerPreferencesEst(player)


class SynergiesTabQualityEvalAlg(TabQualityEvalAlg):
	
	def __init__(self, playerModelBridge, taskModelBridge, syntergyTablePath):
		super().__init__(playerModelBridge)

		self.taskModelBridge = taskModelBridge
		tempTable = pd.read_csv(syntergyTablePath, sep=",", dtype={'agent_1': object, 'agent_2': object})
		synergyTable = tempTable.pivot_table(values='synergy', index='agent_1', columns='agent_2')

		self.synergyMatrix = synergyTable.to_numpy()
		self.synergyMatrix[numpy.isnan(self.synergyMatrix)] = 0
		self.synergyMatrix = self.symmetrize(self.synergyMatrix)

		self.playerIds = self.playerModelBridge.getAllPlayerIds()

	def symmetrize(self, table):
		return table + table.T - numpy.diag(table.diagonal())

	def evaluate(self, _, groupPlayerIds):
		
		totalQuality = 0
		groupSize = len(groupPlayerIds)
		numElemCombs = math.comb(groupSize, 2)
		for i in range(groupSize-1):
			firstPlayerId = groupPlayerIds[i]
			firstPlayerPreferences = self.playerModelBridge.getPlayerPreferencesEst(firstPlayerId)
			firstPlayerPreferencesInBinary = ''
			for dim in firstPlayerPreferences.dimensions:
				firstPlayerPreferencesInBinary += str(round(firstPlayerPreferences.dimensions[dim]))

			# assumes synergy matrix symetry
			for j in range(i+1, len(groupPlayerIds)):
				secondPlayerId = groupPlayerIds[j]
				secondPlayerPreferences = self.playerModelBridge.getPlayerPreferencesEst(secondPlayerId)
				secondPlayerPreferenceInBinary = ''
				for dim in secondPlayerPreferences.dimensions:
					secondPlayerPreferenceInBinary += str(round(secondPlayerPreferences.dimensions[dim]))

			firstPlayerPreferencesIndex = int(firstPlayerPreferencesInBinary, 2)
			secondPlayerPreferencesIndex = int(secondPlayerPreferenceInBinary, 2)

			totalQuality += self.synergyMatrix[firstPlayerPreferencesIndex][secondPlayerPreferencesIndex] / numElemCombs
		return totalQuality

