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
		qualityEvalAlg, 
		qualityWeights = None):

		super().__init__(playerModelBridge)
		
		self.playerModelBridge = playerModelBridge
		self.qualityWeights = PlayerCharacteristics(ability = 0.5, engagement = 0.5) if qualityWeights == None else qualityWeights 

		self.interactionsProfileTemplate = interactionsProfileTemplate
		self.qualityEvalAlg = qualityEvalAlg
		self.bestQualities = {}

	def calcQuality(self, state):
		return self.qualityWeights.ability*state.characteristics.ability + self.qualityWeights.engagement*state.characteristics.engagement
	
	
	def updateEstimates(self):
		playerIds = self.playerModelBridge.get_all_player_ids()
		for playerId in playerIds:
			currPreferencesEst = self.playerModelBridge.get_player_preferences_est(playerId)
			currPreferencesQuality = self.bestQualities.get(playerId, 0.0)
			lastDataPoint = self.playerModelBridge.get_player_curr_state(playerId)
			quality = self.calcQuality(lastDataPoint)
			if quality > currPreferencesQuality:
				self.bestQualities[playerId] = currPreferencesQuality
				self.playerModelBridge.set_player_preferences_est(playerId, lastDataPoint.profile)




class ExplorationPreferencesEstAlg(PreferencesEstAlg):
	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		qualityEvalAlg, 
		numTestedPlayerProfiles = None):
		
		super().__init__(playerModelBridge)
		
		self.playerModelBridge = playerModelBridge

		self.numTestedPlayerProfiles = 100 if numTestedPlayerProfiles == None else numTestedPlayerProfiles
		self.interactionsProfileTemplate = interactionsProfileTemplate

		self.qualityEvalAlg = qualityEvalAlg


	def updateEstimates(self):
		playerIds = self.playerModelBridge.get_all_player_ids()
		updatedEstimates = {}
		for playerId in playerIds:
			
			currPreferencesEst = self.playerModelBridge.get_player_preferences_est(playerId)
			newPreferencesEst = currPreferencesEst
			if(currPreferencesEst != None):
				bestQuality = self.qualityEvalAlg.evaluate(currPreferencesEst, [playerId])
			else:
				bestQuality = -1
			
			for i in range(self.numTestedPlayerProfiles):
				profile = self.interactionsProfileTemplate.generate_copy().randomize()
				currQuality = self.qualityEvalAlg.evaluate(profile, [playerId])
				if currQuality >= bestQuality:
					bestQuality = currQuality
					newPreferencesEst = profile

			self.playerModelBridge.set_player_preferences_est(playerId, newPreferencesEst)
			updatedEstimates[str(playerId)] = newPreferencesEst


		
		return updatedEstimates
