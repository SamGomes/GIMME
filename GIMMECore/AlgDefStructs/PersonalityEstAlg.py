from abc import ABC, abstractmethod
from ..PlayerStructs import *
import json

class PersonalityEstAlg(ABC):

	def __init__(self, playerModelBridge):
		self.playerModelBridge = playerModelBridge

	@abstractmethod
	def updateEstimates(self):
		pass



class ExploitationPersonalityEstAlg(PersonalityEstAlg):
	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		regAlg, 
		qualityWeights = None):

		super().__init__(playerModelBridge)
		
		self.playerModelBridge = playerModelBridge
		self.qualityWeights = PlayerCharacteristics(ability = 0.5, engagement = 0.5) if qualityWeights == None else qualityWeights 

		self.interactionsProfileTemplate = interactionsProfileTemplate
		self.regAlg = regAlg
		self.bestQualities = {}

	def calcQuality(self, state):
		return self.qualityWeights.ability*state.characteristics.ability + self.qualityWeights.engagement*state.characteristics.engagement
	
	
	def updateEstimates(self):
		playerIds = self.playerModelBridge.getAllPlayerIds()
		for playerId in playerIds:
			currPersonalityEst = self.playerModelBridge.getPlayerPersonalityEst(playerId)
			currPersonalityQuality = self.bestQualities.get(playerId, 0.0)
			lastDataPoint = self.playerModelBridge.getPlayerCurrState(playerId)
			quality = self.calcQuality(lastDataPoint)
			if quality > currPersonalityQuality:
				self.bestQualities[playerId] = currPersonalityQuality
				self.playerModelBridge.setPlayerPersonalityEst(playerId, lastDataPoint.profile)




class ExplorationPersonalityEstAlg(PersonalityEstAlg):
	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		regAlg, 
		numTestedPlayerProfiles = None, 
		qualityWeights = None):
		
		super().__init__(playerModelBridge)
		
		self.playerModelBridge = playerModelBridge
		self.qualityWeights = PlayerCharacteristics(ability = 0.5, engagement = 0.5) if qualityWeights == None else qualityWeights 

		self.numTestedPlayerProfiles = 100 if numTestedPlayerProfiles == None else numTestedPlayerProfiles
		self.interactionsProfileTemplate = interactionsProfileTemplate

		self.regAlg = regAlg

	def calcQuality(self, state):
		return self.qualityWeights.ability*state.characteristics.ability + self.qualityWeights.engagement*state.characteristics.engagement
	
	def updateEstimates(self):
		playerIds = self.playerModelBridge.getAllPlayerIds()
		for playerId in playerIds:
			
			currPersonalityEst = self.playerModelBridge.getPlayerPersonalityEst(playerId)
			newPersonalityEst = currPersonalityEst
			if(currPersonalityEst != None):
				bestQuality = self.calcQuality(self.regAlg.predict(currPersonalityEst, playerId))
			else:
				bestQuality = -1
			
			for i in range(self.numTestedPlayerProfiles):
				profile = self.interactionsProfileTemplate.generateCopy().randomize()
				currQuality = self.calcQuality(self.regAlg.predict(profile, playerId))
				if currQuality >= bestQuality:
					bestQuality = currQuality
					newPersonalityEst = profile

			self.playerModelBridge.setPlayerPersonalityEst(playerId, newPersonalityEst)
