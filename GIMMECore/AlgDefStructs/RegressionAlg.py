from abc import ABC, abstractmethod
import copy
from PlayerStructs import *
import json

class RegressionAlg(ABC):

	@abstractmethod
	def predict(self, profile, playerModelBridge, playerId):
		pass


# ---------------------- KNNRegression stuff ---------------------------
class KNNRegression(RegressionAlg):

	def __init__(self, numberOfNNs):
		self.numberOfNNs = numberOfNNs
		# print(self.numberOfNNs)


	def interactionsProfileSort(self, elem):
		return elem.dist

	def predict(self, profile, playerModelBridge, playerId):

		pastModelIncs = playerModelBridge.getPlayerPastModelIncreases(playerId).getAllStates()
		# print(pastModelIncs)
		pastModelIncsCopy = copy.deepcopy(pastModelIncs)
		pastModelIncsSize = len(pastModelIncs)

		predictedState = PlayerState(profile, PlayerCharacteristics())
	
		for modelInc in pastModelIncsCopy:
			modelInc.dist = profile.sqrDistanceBetween(modelInc.profile)

		pastModelIncsCopy = sorted(pastModelIncsCopy, key=self.interactionsProfileSort)

		# print(self.numberOfNNs)
		# print(json.dumps(pastModelIncsCopy, default=lambda o: o.__dict__, sort_keys=True))
		numberOfIterations = min(self.numberOfNNs, len(pastModelIncsCopy))
		for i in range(numberOfIterations):
			currState = pastModelIncsCopy[i]
			# pastProfile = currState.profile
			pastCharacteristics = currState.characteristics
			# distance = profile.sqrDistanceBetween(pastProfile)

			predictedState.characteristics.ability += pastCharacteristics.ability / numberOfIterations * ((numberOfIterations - i)/numberOfIterations)
			predictedState.characteristics.engagement += pastCharacteristics.engagement/ numberOfIterations * ((numberOfIterations - i)/numberOfIterations)
		
		return predictedState


# ---------------------- NeuralNetworkRegression stuff ---------------------------
class NeuralNetworkRegression(RegressionAlg):

	def predict(self, profile, playerModelBridge, playerId):
		pass


# ---------------------- ReinforcementLearningRegression stuff ---------------------------
class ReinforcementLearningRegression(RegressionAlg):

	def predict(self, profile, playerModelBridge, playerId):
		pass

