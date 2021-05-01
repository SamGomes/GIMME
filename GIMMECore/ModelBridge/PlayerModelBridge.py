from abc import ABC, abstractmethod
from ..PlayerStructs import *

class PlayerModelBridge(ABC):
	@abstractmethod
	def resetPlayer(self, playerId):
		pass

	@abstractmethod
	def getAllPlayerIds(self):
		pass

	@abstractmethod
	def getPlayerName(self, playerId):
		pass


	@abstractmethod
	def getPlayerCurrState(self,  playerId):
		pass

	@abstractmethod
	def getPlayerCurrCharacteristics(self, playerId):
		pass
		

	@abstractmethod
	def getPlayerStateGrid(self, playerId):
		pass

	@abstractmethod
	def getPlayerPersonalityEst(self, playerId):
		pass

	@abstractmethod
	def setPlayerPersonalityEst(self, playerId, personality):
		pass

	@abstractmethod
	def setAndSavePlayerStateToGrid(self, playerId, increases, newState):
		pass

	
	@abstractmethod
	def setPlayerCharacteristics(self, playerId, characteristics):
		pass

	@abstractmethod		
	def setPlayerGroup(self, playerId, group):
		pass

	@abstractmethod		
	def setPlayerTasks(self, playerId, tasks):
		pass

	@abstractmethod		
	def setPlayerProfile(self, playerId, profile):
		pass