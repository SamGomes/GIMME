from abc import ABC, abstractmethod
from PlayerStructs import *

class PlayerModelBridge(ABC):

	@abstractmethod
	def registerNewPlayer(self, playerId, currState, name, numPastModelIncreasesCells, maxAmountOfStoredProfilesPerCell, pastModelIncreasesGrid, currModelIncreases):
		pass


	@abstractmethod
	def saveplayerIncreases(self, playerId, stateIncreases):
		pass

	@abstractmethod
	def resetPlayer(self, playerId):
		pass


	@abstractmethod
	def getSelectedPlayerIds(self):
		pass

	@abstractmethod
	def getPlayerName(self, playerId):
		pass


	@abstractmethod
	def getPlayerCurrState(self,  playerId):
		pass

	@abstractmethod
	def getPlayerCurrProfile(self,  playerId):
		pass

	@abstractmethod
	def getPlayerCurrCharacteristics(self, playerId):
		pass
		

	@abstractmethod
	def getPlayerPastModelIncreases(self, playerId):
		pass

	@abstractmethod
	def getPlayerPersonality(self, playerId):
		pass


	@abstractmethod
	def setPlayerCharacteristics(self, playerId, characteristics):
		pass

	@abstractmethod
	def setPlayerCurrProfile(self, playerId, profile):
		pass
