import math
#from sre_parse import State
import time
import copy
import json
from .InteractionsProfile import InteractionsProfile

from abc import ABC, abstractmethod


class PlayerCharacteristics(object):
	def __init__(self, ability=None, engagement=None, group_diversity=None):
		self.ability = 0 if ability == None else ability
		self.engagement = 0 if engagement == None else engagement
		self.group_diversity = 0 if group_diversity == None else group_diversity

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
	def __init__(self, stateType=None, profile=None, characteristics=None, dist=None, quality=None, group=None,
				 tasks=None):
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
	

class PlayerPersonality(ABC):
	def __init__(self):
		self.maxDifferenceValue = 1

	@abstractmethod
	def getPersonalityString(self):
		pass

	@abstractmethod
	def getPairPersonalityDiversity(self, other):
		pass

	@abstractmethod
	def getTeamPersonalityDiversity(players):
		pass


class PersonalityMBTI(PlayerPersonality):
	def __init__(self):
		super()
			
	def __init__(self, letter1, letter2, letter3, letter4):
		super()
		self.setLetters(letter1, letter2, letter3, letter4)

	def setLetters(self, letter1, letter2, letter3, letter4):
		self.letter1 = letter1.upper()
		self.letter2 = letter2.upper()
		self.letter3 = letter3.upper()
		self.letter4 = letter4.upper()

	def getPersonalityString(self):
		return self.letter1 + self.letter2 + self.letter3 + self.letter4

	def getLettersList(self):
		return [self.letter1, self.letter2, self.letter3, self.letter4]

	# Determine the difference between 2 personalities. The value ranges from 0 (no difference) to 1 (max difference). 
	def getPairPersonalityDiversity(self, other):
		if not isinstance(other, PersonalityMBTI):
			raise Exception("[ERROR] Comparison between different personality models not allowed.")

		difference = 0
		otherLetters = other.getLettersList()
		selfLetters = self.getLettersList()

		for i in range(0, len(selfLetters)):
			difference += 0 if selfLetters[i] == otherLetters[i] else self.maxDifferenceValue / 4

		return difference

	# Determine the group personality difference. Using a formula proposed by Pieterse, Kourie and Sonnekus.
	# players is a list of PlayerPersonalties
	def getTeamPersonalityDiversity(players):
		# Difference is 0 if there's only one player
		if len(players) == 1:
			return 0

		letters_list = [[], [], [], []]

		# Populate letters_list with all the players' letters
		for player in players:
			for letters, letter in zip(letters_list, player.getLettersList()):
				letters.append(letter)
			
		difference = 0.0

		for letters in letters_list:
			letters.sort()
			length = len(letters)


			if (letters[0] != letters[1]) or (letters[length-2] != letters[length-1]):
				# The first/last two letters are different -> means all but one letters are the same (difference = 1)
				difference += 1.0
				continue
			elif length <= 3:
				# The first/last two letters are the same and len =< 3, means all the letters are the same (difference = 0)
				continue

			for letter in letters:
				# If not all the letters are the same, then difference = 2
				if letter != letters[1]:
					difference += 2.0
					break

			# Otherwise, all the letters are the same (difference = 0)

		# Max value for difference is 8. Divide by 8 in order to normalize it.
		return difference / 8.0


class PlayerStatesDataFrame(object):
	def __init__(self, interactionsProfileTemplate, trimAlg, states=None):
		self.interactionsProfileTemplate = interactionsProfileTemplate
		self.trimAlg = trimAlg

		self.states = [] if states == None else states

		# auxiliary stuff
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

		# auxiliary stuff
		self.flatProfiles = []
		self.flatAbilities = []
		self.flatEngagements = []

		return self

	def pushToDataFrame(self, playerState):
		self.states.append(playerState)

		# update tuple representation
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

		# update tuple representation
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
