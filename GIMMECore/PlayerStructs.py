import math
from sre_parse import State
import time
import copy
import json
from .InteractionsProfile import InteractionsProfile



class PlayerCharacteristics(object):
	def __init__(self, ability = None, engagement = None):
		self.ability  = 0 if ability==None else ability
		self.engagement = 0 if engagement==None else engagement

		# self._ability  = 0 if ability==None else min(ability, 20.0)
		# self._engagement = 0 if engagement==None else min(engagement, 1.0)

	def reset(self):
		self.ability = 0
		self.engagement = 0
		return self

	# #constraints enforced via properties
	# @property
	# def ability(self):
	# 	return self._ability

	# @property
	# def engagement(self):
	# 	return self._engagement

	# @ability.setter
	# def ability(self, newValue):
	# 	# print(newValue)
	# 	self._ability = min(newValue, 20.0)
	# 	# self._ability = newValue
	# 	return self._ability
	
	# @engagement.setter
	# def engagement(self, newValue):
	# 	self._engagement = min(newValue, 1.0)
	# 	# self._engagement = newValue
	# 	return self._engagement


class PlayerState(object):
	def __init__(self, stateType = None, profile = None, characteristics = None, dist = None, quality = None, group = None, tasks = None):
		self.creationTime = time.time()
		
		self.stateType = 1 if stateType == None else stateType
		self.profile = InteractionsProfile() if profile == None else profile
		self.characteristics = PlayerCharacteristics() if characteristics == None else characteristics
		self.dist = -1 if dist == None else dist
		self.quality = -1 if quality == None else quality

		self.group = [] if group == None else group
		self.tasks = [] if tasks == None else tasks


	def reset(self):
		self.characteristics.reset()
		self.profile.reset()
		self.creationTime = time.time()
		self.stateType = 1
		self.quality = -1
		self.dist = -1
		
		self.group = []
		self.tasks = []
		return self


class PlayerStatesDataFrame(object):
	def __init__(self, interactionsProfileTemplate, trimAlg, states = None):
		self.interactionsProfileTemplate = interactionsProfileTemplate
		self.trimAlg = trimAlg

		self.states = [] if states == None else states

		#auxiliary stuff
		self.flatProfiles = []
		self.flatAbilities = []
		self.flatEngagements = []
		if states != None:
			for state in self.states:
				self.flatProfiles.append(state.profile.flattened())
				self.flatAbilities.append(state.characteristics.ability)
				self.flatEngagements.append(state.characteristics.engagement)
		

	def reset(self):
		self.states = []
		
		#auxiliary stuff
		self.flatProfiles = []
		self.flatAbilities = []
		self.flatEngagements = []

		return self

	def pushToDataFrame(self, playerState):
		self.states.append(playerState)

		#update tuple representation		
		self.flatProfiles.append(playerState.profile.flattened())
		self.flatAbilities.append(playerState.characteristics.ability)
		self.flatEngagements.append(playerState.characteristics.engagement)

		# print(json.dumps(self.states, default=lambda o: o.__dict__, sort_keys=True, indent=2))
		# print("---------")
		# print(self.flatProfiles)
		# print(self.flatAbilities)
		# print(self.flatEngagements)

		trimmedListAndRemainder = self.trimAlg.trimmedList(self.states)
		trimmedList = trimmedListAndRemainder[0]
		remainderIndexes = trimmedListAndRemainder[1]

		# print(remainderIndexes)
		

		self.states = trimmedList

		
		#update tuple representation 
		for i in remainderIndexes:
			self.flatProfiles.pop(i)
			self.flatAbilities.pop(i)
			self.flatEngagements.pop(i)


	def getAllStates(self):
		return self.states


	def getAllStatesFlatten(self):
		return {'profiles': self.flatProfiles, 'abilities': self.flatAbilities, 'engagements': self.flatEngagements}

	def getNumStates(self):
		return len(self.states)