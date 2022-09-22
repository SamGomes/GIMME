from abc import ABC, abstractmethod
from ..PlayerStructs import *
import json

class PreferencesEstAlg(ABC):

	def __init__(self, playerModelBridge):
		self.playerModelBridge = playerModelBridge

	@abstractmethod
	def updateEstimates(self):
		pass



class ExploitationPreferencesEstAlg(PreferencesEstAlg):
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
			currPreferencesEst = self.playerModelBridge.getPlayerPreferencesEst(playerId)
			currPreferencesQuality = self.bestQualities.get(playerId, 0.0)
			lastDataPoint = self.playerModelBridge.getPlayerCurrState(playerId)
			quality = self.calcQuality(lastDataPoint)
			if quality > currPreferencesQuality:
				self.bestQualities[playerId] = currPreferencesQuality
				self.playerModelBridge.setPlayerPreferencesEst(playerId, lastDataPoint.profile)




class ExplorationPreferencesEstAlg(PreferencesEstAlg):
	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		regAlg, 
		numTestedPlayerProfiles = None):
		
		super().__init__(playerModelBridge)
		
		self.playerModelBridge = playerModelBridge

		self.numTestedPlayerProfiles = 100 if numTestedPlayerProfiles == None else numTestedPlayerProfiles
		self.interactionsProfileTemplate = interactionsProfileTemplate

		self.regAlg = regAlg


	def updateEstimates(self):
		playerIds = self.playerModelBridge.getAllPlayerIds()
		updatedEstimates = {}
		for playerId in playerIds:
			
			currPreferencesEst = self.playerModelBridge.getPlayerPreferencesEst(playerId)
			newPreferencesEst = currPreferencesEst
			if(currPreferencesEst != None):
				bestQuality = self.regAlg.predict(currPreferencesEst, playerId)
			else:
				bestQuality = -1
			
			for i in range(self.numTestedPlayerProfiles):
				profile = self.interactionsProfileTemplate.generateCopy().randomize()
				currQuality = self.regAlg.predict(profile, playerId)
				if currQuality >= bestQuality:
					bestQuality = currQuality
					newPreferencesEst = profile

			self.playerModelBridge.setPlayerPreferencesEst(playerId, newPreferencesEst)
			updatedEstimates[str(playerId)] = newPreferencesEst


		
		return updatedEstimates
