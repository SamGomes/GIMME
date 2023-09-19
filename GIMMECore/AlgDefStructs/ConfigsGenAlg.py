import gc
import random
import math
import copy
from sys import breakpointhook
import numpy
import re
from abc import ABC, abstractmethod

import GIMMESolver as gs
from ctypes import *

from ..InteractionsProfile import InteractionsProfile 
from ..PlayerStructs import *
from ..AlgDefStructs.RegressionAlg import *


class ConfigsGenAlg(ABC):

	def __init__(self, 
		playerModelBridge,
		interactionsProfileTemplate, 
		taskModelBridge = None, 
		preferredNumberOfPlayersPerGroup = None, 
		minNumberOfPlayersPerGroup = None, 
		maxNumberOfPlayersPerGroup = None,
		jointPlayerConstraints = "",
		separatedPlayerConstraints = ""):

		self.groupSizeFreqs = {}
		self.configSizeFreqs = {}

		self.jointPlayersConstraints = []
		self.separatedPlayersConstraints = []
		self.allConstraints = []
		
		minNumberOfPlayersPerGroup = 2 if minNumberOfPlayersPerGroup == None else minNumberOfPlayersPerGroup 
		maxNumberOfPlayersPerGroup = 5 if maxNumberOfPlayersPerGroup == None else maxNumberOfPlayersPerGroup

		if(minNumberOfPlayersPerGroup > maxNumberOfPlayersPerGroup):
			raise ValueError('The min number of players per group cannot be higher than the max!') 
		
		if preferredNumberOfPlayersPerGroup == None:
			self.maxNumberOfPlayersPerGroup = maxNumberOfPlayersPerGroup
			self.minNumberOfPlayersPerGroup = minNumberOfPlayersPerGroup
		else:
			self.maxNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup
			self.minNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup

		self.playerModelBridge = playerModelBridge
		self.taskModelBridge = taskModelBridge
		self.interactionsProfileTemplate = interactionsProfileTemplate
		
		jointPlayerConstraints = self.fromStringConstraintToList(jointPlayerConstraints)
		separatedPlayerConstraints = self.fromStringConstraintToList(separatedPlayerConstraints)
		
		for i in range(len(jointPlayerConstraints)):
			if jointPlayerConstraints[i] == ['']:
				continue
			self.addJointPlayersConstraints(jointPlayerConstraints[i])
			
		for i in range(len(separatedPlayerConstraints)):
			if separatedPlayerConstraints[i] == ['']:
				continue
			self.addSeparatedPlayersConstraints(separatedPlayerConstraints[i])

		self.completionPerc = 0.0


	def init(self):
		self.groupSizeFreqs = {}
		self.configSizeFreqs = {}
		return self

	def reset(self):
		return self.init()

	def randomConfigGenerator(self, playerIds, minNumGroups, maxNumGroups):
		
		returnedConfig = []
		playerJointRequirements = {}
		playerSeparatedRequirements = {}
		#listOfPlayersWithJointRequirements = []
		if self.jointPlayersConstraints != []:
			for id in playerIds:
				playerJointRequirements[str(id)] = []

			for constraint in self.jointPlayersConstraints:
				for i in range(len(constraint)):
					
					for j in range(len(constraint)):
						if constraint[i] == constraint[j]:
							continue

						playerJointRequirements[constraint[i]].append(constraint[j])


			for id in playerIds:
				for restrictedId in playerJointRequirements[id]:
					for restrictionOfRestrictedId in playerJointRequirements[restrictedId]:
						if restrictionOfRestrictedId not in playerJointRequirements[id] and restrictionOfRestrictedId != restrictedId:
							playerJointRequirements[id].append(restrictionOfRestrictedId)

		if self.separatedPlayersConstraints != []:
			for id in playerIds:
				playerSeparatedRequirements[str(id)] = []

			for constraint in self.separatedPlayersConstraints:
				for i in range(len(constraint)):
					
					for j in range(len(constraint)):
						if constraint[i] == constraint[j]:
							continue

						playerSeparatedRequirements[constraint[i]].append(constraint[j])

		if(len(playerIds) < self.minNumberOfPlayersPerGroup):
			print("number of players is lower than the minimum number of players per group!")
			return bestGroups
			
		# generate random config
		playersWithoutGroup = playerIds.copy()

		if(minNumGroups < maxNumGroups):
			numGroups = numpy.random.randint(minNumGroups, maxNumGroups)
		else: # players length is 1
			numGroups = maxNumGroups

		# generate min num players for each group
		playersWithoutGroupSize = len(playersWithoutGroup)
		# if (listOfPlayersWithJointRequirements != []):
		# 	playersWithoutGroup = listOfPlayersWithJointRequirements.copy()

		# playersWithoutGroupWithoutRestrictions = list(set(playersWithoutGroup) - set(listOfPlayersWithJointRequirements))
		for g in range(numGroups):
			currGroup = []
			currGroupSeparationRestrictions = []

			if(playersWithoutGroupSize < 1):
				break

			# add min number of players to the group
			for p in range(self.minNumberOfPlayersPerGroup):
				currPlayerIndex = random.randint(0, len(playersWithoutGroup) - 1)
				
				currPlayerID = playersWithoutGroup[currPlayerIndex]
				currGroup.append(currPlayerID)
				del playersWithoutGroup[currPlayerIndex]

			if ((playerSeparatedRequirements != {} or playerJointRequirements != {}) and len(playersWithoutGroup) > 0 ):
				self.verifyCoalitionValidity(currGroup, playerJointRequirements, playerSeparatedRequirements, playersWithoutGroup)
			returnedConfig.append(currGroup)
		
		# append the rest
		playersWithoutGroupSize = len(playersWithoutGroup)
		while playersWithoutGroupSize > 0:
			currPlayerIndex = 0;
			if (playersWithoutGroupSize > 1):
				currPlayerIndex = random.randint(0, playersWithoutGroupSize - 1)
			else:
				currPlayerIndex = 0
			currPlayerID = playersWithoutGroup[currPlayerIndex]

			groupsSize = len(returnedConfig)

			availableGroups = returnedConfig.copy()
			while (len(currGroup) > (self.maxNumberOfPlayersPerGroup - 1)):
				if(len(availableGroups) < 1):
					currGroup = random.choice(returnedConfig)
					break
				currGroup = random.choice(availableGroups)
				availableGroups.remove(currGroup)

			currGroup.append(currPlayerID)

			del playersWithoutGroup[currPlayerIndex]
			playersWithoutGroupSize = len(playersWithoutGroup)
		
		return returnedConfig

	def verifyCoalitionValidity(self, config, playerJointRequirements, playerSeparatedRequirements, playersWithoutGroup):
		for i in range(len(config)):
			if playerJointRequirements[config[i]] != []:
				playersNotInCoalition = []
				for player in playerJointRequirements[config[i]]:
					if player not in config:
						playersNotInCoalition.append(player)
				
				if playersNotInCoalition != []:
					
					for j in range(len(config)):
						if i != j and playersNotInCoalition[0] in playersWithoutGroup and config[j] not in playerJointRequirements[config[i]]:
							playersWithoutGroup.append(config[j])
							config[j] = playersNotInCoalition[0]
							playersWithoutGroup.remove(playersNotInCoalition[0])
							del playersNotInCoalition[0]

							if len(playersNotInCoalition) == 0:
								break
			
			if playerSeparatedRequirements[config[i]] != []:
				for player in playerSeparatedRequirements[config[i]]:
					if player in config:
						currPlayerIndex = random.randint(0, len(playersWithoutGroup) - 1)
						while playersWithoutGroup[currPlayerIndex] in playerSeparatedRequirements[config[i]]:
							currPlayerIndex = random.randint(0, len(playersWithoutGroup) - 1)

						config.remove(player)
						config.append(playersWithoutGroup[currPlayerIndex])
						del playersWithoutGroup[currPlayerIndex]

						playersWithoutGroup.append(player)
					
		return config


	def fromStringConstraintToList(self, constraints):
		constraints = constraints.split(';')

		for i in range(len(constraints)):
			constraints[i] = re.sub('[^A-Za-z0-9,_]+', '', constraints[i]).split(',')
		
		return constraints

	def addJointPlayersConstraints(self, players):
		self.jointPlayersConstraints.append(players)
		self.allConstraints.append({"players": players, "type": "JOIN"})

	def addSeparatedPlayersConstraints(self, players):
		self.separatedPlayersConstraints.append(players)
		self.allConstraints.append({"players": players, "type": "SEPARATE"})


	def resetPlayersConstraints(self):
		self.jointPlayersConstraints = []
		self.separatedPlayersConstraints = []
		self.allConstraints = []

	def getPlayerConstraints(self):
		return self.allConstraints

	@abstractmethod
	def organize(self):
		pass

	def updateMetrics(self, groups):

		# kind of sub-optimal, but guarantees encapsulation
		if(self.configSizeFreqs.get(len(groups))):
			self.configSizeFreqs[len(groups)]+=1
		else:
			self.configSizeFreqs[len(groups)]=1

		for group in groups:
			if(self.configSizeFreqs.get(len(group))):
				self.configSizeFreqs[len(group)]+=1
			else:
				self.configSizeFreqs[len(group)]=1


	def getCompPercentage(self):
		return self.completionPerc



class RandomConfigsGen(ConfigsGenAlg):

	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		preferredNumberOfPlayersPerGroup = None, 
		minNumberOfPlayersPerGroup = None, 
		maxNumberOfPlayersPerGroup = None,
		jointPlayerConstraints = "",
		separatedPlayerConstraints = ""):
		super().__init__(
			playerModelBridge = playerModelBridge,
			interactionsProfileTemplate = interactionsProfileTemplate, 
			preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
			minNumberOfPlayersPerGroup = minNumberOfPlayersPerGroup, 
			maxNumberOfPlayersPerGroup = maxNumberOfPlayersPerGroup,
			jointPlayerConstraints = jointPlayerConstraints,
			separatedPlayerConstraints = separatedPlayerConstraints)

	def organize(self):
		playerIds = self.playerModelBridge.getAllPlayerIds() 
		minNumGroups = math.ceil(len(playerIds) / self.maxNumberOfPlayersPerGroup)
		maxNumGroups = math.floor(len(playerIds) / self.minNumberOfPlayersPerGroup)
		
		newConfigProfiles = []
		newAvgCharacteristics = []
		newGroups = self.randomConfigGenerator(playerIds, minNumGroups, maxNumGroups)
		newConfigSize = len(newGroups)		
		# generate profiles
		for groupI in range(newConfigSize):
			group = newGroups[groupI]
			groupSize = len(group)


			profile = self.interactionsProfileTemplate.generateCopy().randomize()			
			# generate random profile
			# for currPlayer in group:
			# 	random = self.interactionsProfileTemplate.generateCopy().randomize()
			# 	for dim in profile.dimensions:
			# 		profile.dimensions[dim] += random.dimensions[dim] / groupSize
			# profile.normalize()
			newConfigProfiles.append(profile)


			currAvgCharacteristics = PlayerCharacteristics().reset()
			for currPlayer in group:
				currState = self.playerModelBridge.getPlayerCurrState(currPlayer)
				currAvgCharacteristics.ability += currState.characteristics.ability / groupSize
				currAvgCharacteristics.engagement += currState.characteristics.engagement / groupSize			
			# currAvgCharacteristics.profile = profile

			diversityValueAlg = DiversityValueAlg(self.playerModelBridge, 0)
			personalities = diversityValueAlg.getPersonalitiesListFromPlayerIds(group)
			currAvgCharacteristics.group_diversity = diversityValueAlg.getTeamPersonalityDiveristy(personalities)

			newAvgCharacteristics.append(currAvgCharacteristics)

			self.completionPerc = groupI/newConfigSize

		self.updateMetrics(newGroups)
		#print(newGroups)
		return {"groups": newGroups, "profiles": newConfigProfiles, "avgCharacteristics": newAvgCharacteristics}




class AnnealedPRSConfigsGen(ConfigsGenAlg):

	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		regAlg, 
		persEstAlg, 
		temperatureDecay, 
		numberOfConfigChoices = None, 
		preferredNumberOfPlayersPerGroup = None, 
		minNumberOfPlayersPerGroup = None, 
		maxNumberOfPlayersPerGroup = None):

		super().__init__(
			playerModelBridge = playerModelBridge,
			interactionsProfileTemplate = interactionsProfileTemplate, 
			preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
			minNumberOfPlayersPerGroup = minNumberOfPlayersPerGroup, 
			maxNumberOfPlayersPerGroup = maxNumberOfPlayersPerGroup)

		self.regAlg = regAlg
		self.persEstAlg = persEstAlg
		self.numberOfConfigChoices = 100 if numberOfConfigChoices == None else numberOfConfigChoices

		self.temperature = 1.0
		self.temperatureDecay = temperatureDecay


	def init(self):
		super().init()
		self.temperature = 1.0

	def reset(self, temperature):
		super().reset()
		self.temperature = numpy.clip(temperature, 0, 1)

	def organize(self):
		playerIds = self.playerModelBridge.getAllPlayerIds() 
		minNumGroups = math.ceil(len(playerIds) / self.maxNumberOfPlayersPerGroup)
		maxNumGroups = math.floor(len(playerIds) / self.minNumberOfPlayersPerGroup)

		currMaxQuality = -float("inf")
		bestGroups = []
		bestConfigProfiles = []
		bestAvgCharacteristics = []


		# estimate preferences
		self.playerPrefEstimates = self.persEstAlg.updateEstimates()

		if (self.regAlg.isTabular()):
			self.regAlg.playerPrefEstimates = self.playerPrefEstimates

		playersCurrState = {}
		for player in self.playerIds:
			playersCurrState[player] = self.playerModelBridge.getPlayerCurrState(player)

		# generate several random groups, calculate their fitness and select the best one
		for i in range(self.numberOfConfigChoices):
			
			# generate several random groups
			newGroups = self.randomConfigGenerator(playerIds, minNumGroups, maxNumGroups)
			newConfigSize = len(newGroups)
			currQuality = 0.0
			newConfigProfiles = []
			newAvgCharacteristics = []

			# generate profiles
			for groupI in range(newConfigSize):
				group = newGroups[groupI]
				groupSize = len(group)

				# generate group profile as random or average of the preferences estimates
				profile = self.interactionsProfileTemplate.generateCopy().reset()

				if(random.uniform(0.0, 1.0) > self.temperature):
					for currPlayer in group:
						preferences = self.playerPrefEstimates[currPlayer]
						for dim in profile.dimensions:
							profile.dimensions[dim] += preferences.dimensions[dim] / groupSize
					# profile.normalize()
				else:
					profile = self.interactionsProfileTemplate.generateCopy().randomize()

				newConfigProfiles.append(profile)
				
				# calculate fitness and average state
				currAvgCharacteristics = PlayerCharacteristics()
				currAvgCharacteristics.reset()
				for i in range(len(group)):

					currState = playersCurrState[group[i]]
					currState.profile = profile

					currAvgCharacteristics.ability += currState.characteristics.ability / groupSize
					currAvgCharacteristics.engagement += currState.characteristics.engagement / groupSize
				
					if (self.regAlg.isTabular()):
						firstPlayerPreferences = self.playerPrefEstimates[group[i]]
						for j in range(i+1, groupSize):
							currQuality += self.regAlg.predict(firstPlayerPreferences, group[j]) / math.comb(groupSize, 2)

					elif (not self.regAlg.isGroupPredict()):
						currQuality += self.regAlg.predict(profile, group[i])


				if (self.regAlg.isGroupPredict()):
					currQuality += self.regAlg.groupPredict(group)
		
				diversityValueAlg = DiversityValueAlg(self.playerModelBridge, 0)
				personalities = diversityValueAlg.getPersonalitiesListFromPlayerIds(group)
				currAvgCharacteristics.group_diversity = diversityValueAlg.getTeamPersonalityDiveristy(personalities)

				newAvgCharacteristics.append(currAvgCharacteristics)
			
			if (currQuality > currMaxQuality):
				bestGroups = newGroups
				bestConfigProfiles = newConfigProfiles
				bestAvgCharacteristics = newAvgCharacteristics
				currMaxQuality = currQuality


			self.completionPerc = i/self.numberOfConfigChoices

		if(self.temperature > 0.0):
			self.temperature -= self.temperatureDecay
		else:
			self.temperature = 1.0

		self.updateMetrics(bestGroups)

		return {"groups": bestGroups, "profiles": bestConfigProfiles, "avgCharacteristics": bestAvgCharacteristics}




class PureRandomSearchConfigsGen(ConfigsGenAlg):

	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		regAlg, 
		persEstAlg, 
		numberOfConfigChoices = None, 
		preferredNumberOfPlayersPerGroup = None, 
		minNumberOfPlayersPerGroup = None, 
		maxNumberOfPlayersPerGroup = None,
		jointPlayerConstraints = "",
		separatedPlayerConstraints = ""):
		
		super().__init__(
			playerModelBridge = playerModelBridge,
			interactionsProfileTemplate = interactionsProfileTemplate, 
			preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
			minNumberOfPlayersPerGroup = minNumberOfPlayersPerGroup, 
			maxNumberOfPlayersPerGroup = maxNumberOfPlayersPerGroup,
			jointPlayerConstraints = jointPlayerConstraints,
			separatedPlayerConstraints = separatedPlayerConstraints)

		self.regAlg = regAlg
		self.persEstAlg = persEstAlg
		self.numberOfConfigChoices = 100 if numberOfConfigChoices == None else numberOfConfigChoices
	
	def organize(self):
		playerIds = self.playerModelBridge.getAllPlayerIds() 
		minNumGroups = math.ceil(len(playerIds) / self.maxNumberOfPlayersPerGroup)
		maxNumGroups = math.floor(len(playerIds) / self.minNumberOfPlayersPerGroup)

		currMaxQuality = -float("inf")
		bestGroups = []
		bestConfigProfiles = []
		bestAvgCharacteristics = []


		# estimate preferences
		self.playerPrefEstimates = self.persEstAlg.updateEstimates()

		if (self.regAlg.isTabular()):
			self.regAlg.playerPrefEstimates = self.playerPrefEstimates

		# generate several random groups, calculate their fitness and select the best one
		for i in range(self.numberOfConfigChoices):
			
			# generate several random groups
			newGroups = self.randomConfigGenerator(playerIds, minNumGroups, maxNumGroups)
			newConfigSize = len(newGroups)
			currQuality = 0.0
			newConfigProfiles = []
			newAvgCharacteristics = []

			# generate profiles
			for groupI in range(newConfigSize):
				group = newGroups[groupI]
				groupSize = len(group)

				# generate profile as average of the preferences estimates
				profile = self.interactionsProfileTemplate.generateCopy().reset()
				
				for currPlayer in group:
					preferences = self.playerPrefEstimates[currPlayer]
					for dim in profile.dimensions:
						profile.dimensions[dim] += (preferences.dimensions[dim] / groupSize)

				# print("profile in-configGen: "+str(profile.dimensions)+";groupSize: "+str(groupSize))
				# profile.normalize()
				newConfigProfiles.append(profile)

				# calculate fitness and average state
				currAvgCharacteristics = PlayerCharacteristics()
				currAvgCharacteristics.reset()
				for i in range(len(group)):

					currState = self.playerModelBridge.getPlayerCurrState(group[i])
					currState.profile = profile

					currAvgCharacteristics.ability += currState.characteristics.ability / groupSize
					currAvgCharacteristics.engagement += currState.characteristics.engagement / groupSize
				
					if (self.regAlg.isTabular()):
						firstPlayerPreferences = self.playerPrefEstimates[group[i]]
						for j in range(i+1, groupSize):
							currQuality += self.regAlg.predict(firstPlayerPreferences, group[j]) / math.comb(groupSize, 2)

					elif (not self.regAlg.isGroupPredict()):
						currQuality += self.regAlg.predict(profile, group[i])

				if (self.regAlg.isGroupPredict()):
					currQuality += self.regAlg.groupPredict(group)
			
				diversityValueAlg = DiversityValueAlg(self.playerModelBridge, 0)
				personalities = diversityValueAlg.getPersonalitiesListFromPlayerIds(group)
				currAvgCharacteristics.group_diversity = diversityValueAlg.getTeamPersonalityDiveristy(personalities)

				newAvgCharacteristics.append(currAvgCharacteristics)
			
			if (currQuality > currMaxQuality):
				bestGroups = newGroups
				bestConfigProfiles = newConfigProfiles
				bestAvgCharacteristics = newAvgCharacteristics
				currMaxQuality = currQuality


			self.completionPerc = i/self.numberOfConfigChoices

		self.updateMetrics(bestGroups)

		return {"groups": bestGroups, "profiles": bestConfigProfiles, "avgCharacteristics": bestAvgCharacteristics}




class AccuratePRSConfigsGen(ConfigsGenAlg):

	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		simulationFunc, 
		numberOfConfigChoices = None, 
		preferredNumberOfPlayersPerGroup = None, 
		minNumberOfPlayersPerGroup = None, 
		maxNumberOfPlayersPerGroup = None, 
		qualityWeights = None):

		super().__init__(
			playerModelBridge = playerModelBridge,
			interactionsProfileTemplate = interactionsProfileTemplate, 
			preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
			minNumberOfPlayersPerGroup = minNumberOfPlayersPerGroup, 
			maxNumberOfPlayersPerGroup = maxNumberOfPlayersPerGroup)

		self.simulationFunc = simulationFunc
		self.currIteration = 0

		self.numberOfConfigChoices = 100 if numberOfConfigChoices == None else numberOfConfigChoices 
		self.qualityWeights = PlayerCharacteristics(ability = 0.5, engagement = 0.5) if qualityWeights == None else qualityWeights

	def init(self):
		super().init()
		self.currIteration = 0

	def updateCurrIteration(self, newCurrIteration):
		self.currIteration = newCurrIteration

	def calcQuality(self, state):
		return self.qualityWeights.ability*state.characteristics.ability + self.qualityWeights.engagement*state.characteristics.engagement
	
	def organize(self):
		playerIds = self.playerModelBridge.getAllPlayerIds() 
		minNumGroups = math.ceil(len(playerIds) / self.maxNumberOfPlayersPerGroup)
		maxNumGroups = math.floor(len(playerIds) / self.minNumberOfPlayersPerGroup)

		currMaxQuality = -float("inf")

		bestGroups = []
		bestConfigProfiles = []
		bestAvgCharacteristics = []

		# generate several random groups, calculate their fitness and select the best one
		for i in range(self.numberOfConfigChoices):
			
			# generate several random groups
			newGroups = self.randomConfigGenerator(playerIds, minNumGroups, maxNumGroups)
			newConfigSize = len(newGroups)
			currQuality = 0.0
			newConfigProfiles = []
			newAvgCharacteristics = []

			# generate profiles
			for groupI in range(newConfigSize):
				group = newGroups[groupI]
				groupSize = len(group)

				# generate profile as average of the preferences estimates
				profile = self.interactionsProfileTemplate.generateCopy().reset()
				for currPlayer in group:
					preferences = self.playerModelBridge.getPlayerRealPreferences(currPlayer)
					for dim in profile.dimensions:
						profile.dimensions[dim] += preferences.dimensions[dim] / groupSize
				# profile.normalize()
				newConfigProfiles.append(profile)


				# calculate fitness and average state
				currAvgCharacteristics = PlayerCharacteristics()
				currAvgCharacteristics.reset()
				for currPlayer in group:

					currState = self.playerModelBridge.getPlayerCurrState(currPlayer)
					currState.profile = profile

					currAvgCharacteristics.ability += currState.characteristics.ability / groupSize
					currAvgCharacteristics.engagement += currState.characteristics.engagement / groupSize
				
					# does not matter if its executed as bootstrap or not
					newState = self.simulationFunc(
						isBootstrap = False, 
						playerBridge = self.playerModelBridge, 
						state = currState, 
						playerId = currPlayer, 
						currIteration = self.currIteration)
					increases = PlayerState()
					increases.characteristics = PlayerCharacteristics(ability=(newState.characteristics.ability - currState.characteristics.ability), engagement=newState.characteristics.engagement)
					currQuality += self.calcQuality(increases)

				diversityValueAlg = DiversityValueAlg(self.playerModelBridge, 0)
				personalities = diversityValueAlg.getPersonalitiesListFromPlayerIds(group)
				currAvgCharacteristics.group_diversity = diversityValueAlg.getTeamPersonalityDiveristy(personalities)

				newAvgCharacteristics.append(currAvgCharacteristics)
			
			if (currQuality > currMaxQuality):
				bestGroups = newGroups
				bestConfigProfiles = newConfigProfiles
				bestAvgCharacteristics = newAvgCharacteristics
				currMaxQuality = currQuality

			self.completionPerc = i/self.numberOfConfigChoices


		self.updateMetrics(bestGroups)

		return {"groups": bestGroups, "profiles": bestConfigProfiles, "avgCharacteristics": bestAvgCharacteristics}



from deap import base, creator, tools, algorithms
from collections import *

class EvolutionaryConfigsGenDEAP(ConfigsGenAlg):
	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		regAlg = None, 
		persEstAlg = None,
		preferredNumberOfPlayersPerGroup = None, 
		minNumberOfPlayersPerGroup = None, 
		maxNumberOfPlayersPerGroup = None, 

		initialPopulationSize = None, 
		numberOfEvolutionsPerIteration = None,  
		probOfCross = None, 

		probOfMutation = None,
		probOfMutationConfig = None, 
		probOfMutationGIPs = None, 

		numChildrenPerIteration = None,
		numSurvivors = None,

		cxOp = None,
		
		jointPlayerConstraints = "",
		separatedPlayerConstraints = ""):

		super().__init__(
			playerModelBridge = playerModelBridge,
			interactionsProfileTemplate = interactionsProfileTemplate, 
			preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
			minNumberOfPlayersPerGroup = minNumberOfPlayersPerGroup, 
			maxNumberOfPlayersPerGroup = maxNumberOfPlayersPerGroup,
			jointPlayerConstraints = jointPlayerConstraints,
			separatedPlayerConstraints = separatedPlayerConstraints)

		self.regAlg = regAlg
		self.persEstAlg = persEstAlg

		self.initialPopulationSize = 100 if initialPopulationSize == None else initialPopulationSize 


		self.numberOfEvolutionsPerIteration = 500 if numberOfEvolutionsPerIteration == None else numberOfEvolutionsPerIteration
		
		self.probOfMutation = 0.2 if probOfMutation == None else probOfMutation
		self.probOfCross = 0.7 if probOfCross == None else probOfCross
		
		self.probOfMutationConfig = 0.2 if probOfMutationConfig == None else probOfMutationConfig
		self.probOfMutationGIPs = 0.2 if probOfMutationGIPs == None else probOfMutationGIPs
		

		self.numChildrenPerIteration = 5 if numChildrenPerIteration == None else numChildrenPerIteration 
		self.numSurvivors = 5 if numSurvivors == None else numSurvivors 

		if(regAlg==None):
			regAlg = KNNRegression(playerModelBridge, 5)

		self.playerIds = self.playerModelBridge.getAllPlayerIds() 
		self.minNumGroups = math.ceil(len(self.playerIds) / self.maxNumberOfPlayersPerGroup)
		self.maxNumGroups = math.floor(len(self.playerIds) / self.minNumberOfPlayersPerGroup)

		self.searchID = str(id(self)) 

		fitnessFuncId = "FitnessMax"+self.searchID
		individualId = "Individual"+self.searchID

		creator.create(fitnessFuncId, base.Fitness, weights=(1.0,))
		creator.create(individualId, list, fitness=getattr(creator, fitnessFuncId))

		# # conv test
		# creator.create(fitnessFuncId, base.Fitness, weights=(-1.0,))
		# creator.create(individualId, list, fitness=getattr(creator, fitnessFuncId))

		
		self.toolbox = base.Toolbox()

		self.toolbox.register("indices", self.randomIndividualGenerator, self.playerIds, self.minNumGroups, self.maxNumGroups)

		self.toolbox.register("individual", tools.initIterate, getattr(creator, individualId), self.toolbox.indices)
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

		self.cxOp = "order" if cxOp == None else cxOp
		if self.cxOp == "order":
			self.toolbox.register("mate", self.cxGIMME_Order)
		else:
			self.toolbox.register("mate", self.cxGIMME_Simple)

		self.toolbox.register("mutate", self.mutGIMME, pGIPs=self.probOfMutationGIPs, pConfig=self.probOfMutationConfig)

		# self.toolbox.register("select", tools.selRoulette)
		# self.toolbox.register("select", tools.selBest, k=self.numFitSurvivors)
		self.toolbox.register("select", self.selGIMME)

		# self.toolbox.register("evaluate", self.calcFitness_convergenceTest)
		self.toolbox.register("evaluate", self.calcFitness)

		self.resetGenAlg()


	def resetGenAlg(self):

		if hasattr(self, "pop"):
			del self.pop
		if hasattr(self, "hof"):
			del self.hof

		self.pop = self.toolbox.population(n = self.initialPopulationSize)
		self.hof = tools.HallOfFame(1)



	def randomIndividualGenerator(self, playerIds, minNumGroups, maxNumGroups):
		groups = self.randomConfigGenerator(playerIds, minNumGroups, maxNumGroups)
		profs = [self.randomProfileGenerator() for i in range(len(groups))]
		return [groups, profs]

	def randomProfileGenerator(self):
		return self.interactionsProfileTemplate.randomized()


	def cxGIMME_Order(self, ind1, ind2):

		# configs
		config1 = ind1[0]
		config2 = ind2[0]

		newConfig1 = []
		newConfig2 = []

		l1 = len(config1)
		l2 = len(config2)

		if (l1 > l2):
			maxLenConfig = config1
			maxLen = l1
			minLenConfig = config2
			minLen = l2
		else:
			maxLenConfig = config2
			maxLen = l2
			minLenConfig = config1
			minLen = l1

		cxpoints = [] 
		
		clist1 = []
		clist2 = []

		remainder1 = []
		remainder2 = []
		for i in range(minLen):
			parent1 = minLenConfig[i]
			parent2 = maxLenConfig[i]

			cxpoint = random.randint(0,len(minLenConfig[i]))
			cxpoints.append(cxpoint) 

			clist1.extend(parent1)
			clist2.extend(parent2)

			remainder1.extend(parent1[cxpoint:])
			remainder2.extend(parent2[cxpoint:])


		d1 = {k:v for v,k in enumerate(clist1)}
		d2 = {k:v for v,k in enumerate(clist2)}

		remainder1.sort(key=d2.get)
		remainder2.sort(key=d1.get)

		for i in range(minLen):
			parent1 = minLenConfig[i]
			parent2 = maxLenConfig[i]

			cxpoint = cxpoints[i] 

			#C1 Implementation
			# maintain left part
			child1, child2 = parent1[:cxpoint], parent2[:cxpoint]


			# reorder right part
			missingLen1 = len(parent1) - len(child1)
			child1.extend(remainder1[:missingLen1])
			remainder1 = remainder1[missingLen1:]

			missingLen2 = len(parent2) - len(child2)
			child2.extend(remainder2[:missingLen2])
			remainder2 = remainder2[missingLen2:]

			newConfig1.append(child1)
			newConfig2.append(child2)


		#the inds become children
		ind1[0] = newConfig1
		ind2[0] = newConfig2

		# breakpoint()

		# profiles are crossed with one point (no need for that when profiles are 1D)
		# breakpoint()
		# if self.interactionsProfileTemplate.dimensionality > 1:
		for i in range(minLen):
			prof1 = ind1[1][i].flattened()
			prof2 = ind2[1][i].flattened()

			newProfiles = tools.cxUniform(prof1, prof2, 0.5)
			# newProfiles = tools.cxOnePoint(prof1, prof2)
			
			#the inds become children
			ind1[1][i] = self.interactionsProfileTemplate.unflattened(newProfiles[0])
			ind2[1][i] = self.interactionsProfileTemplate.unflattened(newProfiles[1])

			# breakpoint()

		del ind1.fitness.values
		del ind2.fitness.values

		return (ind1, ind2)


	def cxGIMME_Simple(self, ind1, ind2):

		# configs
		config1 = ind1[0]
		config2 = ind2[0]

		newConfig1 = []
		newConfig2 = []

		l1 = len(config1)
		l2 = len(config2)

		if (l1 > l2):
			maxLenConfig = config1
			maxLen = l1
			minLenConfig = config2
			minLen = l2
		else:
			maxLenConfig = config2
			maxLen = l2
			minLenConfig = config1
			minLen = l1

		cxpoints = [] 
		
		clist1 = []
		clist2 = []

		remainder1 = []
		remainder2 = []
		for i in range(minLen):
		
			parent1 = [None, None]
			parent2 = [None, None]

			parent1[0] = minLenConfig[i]
			parent1[1] = ind1[1][i].flattened()
			parent2[0] = maxLenConfig[i]
			parent2[1] = ind2[1][i].flattened()

			clist1.append(parent1)
			clist2.append(parent2)


		# breakpoint()

		for ind,clist in zip([ind1,ind2], [clist1,clist2]):


			# print("-----------[Before]-----------")
			# print(json.dumps(ind[1], default=lambda o: o.__dict__))

			randI1 = random.randint(0, len(clist1) - 1)
			randI2 = random.randint(0, len(clist1) - 1)
			

			newProfilesConfig = tools.cxOnePoint(ind1 = clist[randI1][0], ind2 = clist[randI2][0])
			newProfilesGIP = tools.cxOnePoint(ind1 = clist[randI1][1], ind2 = clist[randI2][1])
			
			ind[0][randI1] = newProfilesConfig[0]
			ind[1][randI1] = self.interactionsProfileTemplate.unflattened(newProfilesGIP[0])

			ind[0][randI2] = newProfilesConfig[1]
			ind[1][randI2] = self.interactionsProfileTemplate.unflattened(newProfilesGIP[1])

			# print("-----------[After]-----------")
			# print(json.dumps(ind[1], default=lambda o: o.__dict__))

		# breakpoint()



		del ind1.fitness.values
		del ind2.fitness.values

		return (ind1, ind2)


	def mutGIMME(self, individual, pGIPs, pConfig):
		
		# mutate config
		if random.uniform(0, 1) <= pConfig:
			
			numberOfMutations = 1
			for i in range(numberOfMutations):
				# breakpoint()
				indCpy = copy.copy(individual)
				
				randI1 = random.randint(0, len(indCpy[0]) - 1)
				innerRandI1 = random.randint(0, len(indCpy[0][randI1]) - 1)

				randI2 = innerRandI2 = -1
				while(randI2 < 0 or randI1 == randI2):
					randI2 = random.randint(0, len(indCpy[0]) - 1)
				while(innerRandI2 < 0 or innerRandI1 == innerRandI2):
					innerRandI2 = random.randint(0, len(indCpy[0][randI2]) - 1)


				elem1 = indCpy[0][randI1][innerRandI1]
				elem2 = indCpy[0][randI2][innerRandI2]


				indCpy[0][randI1][innerRandI1] = elem2
				indCpy[0][randI2][innerRandI2] = elem1

				individual[0] = indCpy[0]
				# breakpoint()
			

		#mutate GIPs
		numberOfMutations = 1
		for i in range(numberOfMutations):
			profs = individual[1]
			keys = list(profs[0].dimensions.keys())
			for i in range(len(profs)):
				if random.uniform(0, 1) <= pGIPs:
					# profs[i].randomize()
					for key in keys:
						if random.uniform(0, 1) <= 0.5:
							profs[i].dimensions[key] += random.uniform(0, min(0.2, 1.0 - profs[i].dimensions[key])) 
						else:
							profs[i].dimensions[key] -= random.uniform(0, min(0.2, profs[i].dimensions[key])) 

			individual[1] = profs
		
		del individual.fitness.values
		return individual,


	def reset(self):
		super().reset()
		self.resetGenAlg()

	def calcFitness_convergenceTest(self, individual):
		config = individual[0]
		profiles = individual[1]

		totalFitness = 0.0

		targetConfig = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23]]

		lenConfig = len(config)
		for groupI in range(lenConfig):
			
			group = config[groupI]
			profile = profiles[groupI]
			
			for playerI in range(len(group)):
				totalFitness += profile.sqrDistanceBetween(InteractionsProfile(dimensions = {'dim_0': 0.98, 'dim_1': 0.005}))
				totalFitness += abs(config[groupI][playerI] - targetConfig[groupI][playerI])
		
		#print(totalFitness)
		totalFitness = totalFitness + 1.0 #helps selection (otherwise Pchoice would always be 0)
		individual.fitness.values = totalFitness,
		return totalFitness, #must return a tuple


	def selGIMME(self, individuals, k, fit_attr="fitness"):
		return tools.selBest(individuals, k, fit_attr)

	def calcFitness(self, individual):
		config = individual[0]
		profiles = individual[1]

		totalFitness = 0.0

		lenConfig = len(config)

		allConstrainsSatisfied = True
		for groupI in range(lenConfig):
			
			group = config[groupI]
			profile = profiles[groupI]


			for constraint in self.jointPlayersConstraints:
				hasToBeInGroup = False
				isNotInGroup = False
				for player in constraint:
					if player in group and isNotInGroup == False:
						hasToBeInGroup = True
					
					elif player not in group and hasToBeInGroup == False:
						isNotInGroup = True

					else:
						allConstrainsSatisfied = False
						break
				
				if allConstrainsSatisfied == False:
					break

			for constraint in self.separatedPlayersConstraints:
				hasToNotBeInGroup = False
				for player in constraint:
					if player in group:
						if hasToNotBeInGroup:
							allConstrainsSatisfied = False
							break
						hasToNotBeInGroup = True
				
				if allConstrainsSatisfied == False:
					break

			for i in range(len(group)):
				if (self.regAlg.isTabular()):
						firstPlayerPreferences = self.playerPrefEstimates[group[i]]
						for j in range(i+1, len(group)):
							totalFitness += self.regAlg.predict(firstPlayerPreferences, group[j]) / math.comb(len(group), 2)

				elif (not self.regAlg.isGroupPredict()):
					totalFitness += self.regAlg.predict(profile, group[i])

			if (self.regAlg.isGroupPredict()):
				totalFitness += self.regAlg.groupPredict(group)
		
		totalFitness = totalFitness + 1.0 #helps selection (otherwise Pchoice would always be 0)
		if allConstrainsSatisfied:
			totalFitness += 1000
		individual.fitness.values = totalFitness,

		return totalFitness, #must return a tuple



	def organize(self):
		self.resetGenAlg()
		if (self.regAlg.isTabular()):
			self.playerPrefEstimates = self.persEstAlg.updateEstimates()
			self.regAlg.playerPrefEstimates = self.playerPrefEstimates

		algorithms.eaMuCommaLambda(
			population = self.pop, 
			toolbox = self.toolbox, 

			lambda_ = self.numChildrenPerIteration, 
			mu = self.numSurvivors, 
			
			cxpb = self.probOfCross, 
			mutpb = self.probOfMutation, 
			
			ngen = self.numberOfEvolutionsPerIteration, 
			
			halloffame = self.hof, 
			verbose = False
		)


		self.completionPerc = len(tools.Logbook())/ self.numberOfEvolutionsPerIteration

		bestGroups = self.hof[0][0]
		bestConfigProfiles = self.hof[0][1]

		# print(bestGroups)
		# print(bestConfigProfiles[0].dimensions)
		# breakpoint()

		avgCharacteristicsArray = []
		for group in bestGroups:
			groupSize = len(group)
			avgCharacteristics = PlayerCharacteristics()
			for currPlayer in group:
				currState = self.playerModelBridge.getPlayerCurrState(currPlayer)
				avgCharacteristics.ability += currState.characteristics.ability / groupSize
				avgCharacteristics.engagement += currState.characteristics.engagement / groupSize
						
				diversityValueAlg = DiversityValueAlg(self.playerModelBridge, 0)
				personalities = diversityValueAlg.getPersonalitiesListFromPlayerIds(group)
				avgCharacteristics.group_diversity = diversityValueAlg.getTeamPersonalityDiveristy(personalities)

			avgCharacteristicsArray.append(avgCharacteristics)


		return {"groups": bestGroups, "profiles": bestConfigProfiles, "avgCharacteristics": avgCharacteristicsArray}

# deterministic algorithms
class ODPIP(ConfigsGenAlg):
	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		regAlg,
		persEstAlg,
		taskModelBridge = None,
		preferredNumberOfPlayersPerGroup = None, 
		minNumberOfPlayersPerGroup = None, 
		maxNumberOfPlayersPerGroup = None,
		jointPlayerConstraints = "",
		separatedPlayerConstraints = ""):

		super().__init__(playerModelBridge, 
		interactionsProfileTemplate, 
		taskModelBridge,
		preferredNumberOfPlayersPerGroup, 
		minNumberOfPlayersPerGroup, 
		maxNumberOfPlayersPerGroup,
		jointPlayerConstraints = jointPlayerConstraints,
		separatedPlayerConstraints = separatedPlayerConstraints)

		self.regAlg = regAlg
		self.persEstAlg = persEstAlg

		self.coalitionsProfiles = []
		self.coalitionsAvgCharacteristics = []
		self.coalitionsValues = []

		self.playerIds = []	
		

		self.playerPrefEstimates = {}

	

	def getCoalitionInByteFormatValue(self, coalitionInByteFormat):
		coalitionInBitFormat = self.convertCoalitionFromByteToBitFormat(coalitionInByteFormat, len(coalitionInByteFormat))
		return self.f[coalitionInBitFormat]


	def getCoalitionStructureInByteFormatValue(self, coalitionStructure):
		valueOfCS = 0
		for i in range(len(coalitionStructure)):
			valueOfCS += self.getCoalitionInByteFormatValue(coalitionStructure[i])
		
		return valueOfCS


	def convertCoalitionFromByteToBitFormat(self, coalitionInByteFormat, coalitionSize):
		coalitionInBitFormat = 0

		for i in range(coalitionSize):
			coalitionInBitFormat += 1 << (coalitionInByteFormat[i] - 1)

		return coalitionInBitFormat

	# convert group in bit format to group with the player ids
	def getGroupFromBitFormat(self, coalition):
		group = []
		tempCoalition = coalition
		playerNumber = 0
		while tempCoalition != 0:
			if tempCoalition & 1:
				group.append(playerNumber + 1)

			playerNumber += 1
			tempCoalition >>= 1

		
		return group

	def convertFromByteToIds(self, coalition):
		group = []

		for agent in coalition:
			group.append(self.playerIds[agent - 1])

		return group

	def convertFromIdsToBytes(self, coalition):
		group = []

		for agent in coalition:
			for i in range(len(self.playerIds)):
				if self.playerIds[i] == agent:
					group.append(i + 1)

		return group


	def getSizeOfCombinationInBitFormat(self, combinationInBitFormat):
		return bin(combinationInBitFormat).count("1")


	def convertSetOfCombinationsFromBitFormat(self, setOfCombinationsInBitFormat):
		setOfCombinationsInByteFormat = numpy.empty(len(setOfCombinationsInBitFormat), dtype=list)
		for i in range(len(setOfCombinationsInBitFormat)):

			setOfCombinationsInByteFormat[i] = self.getGroupFromBitFormat(setOfCombinationsInBitFormat[i])

		return setOfCombinationsInByteFormat

	def computeAllCoalitionsValues(self):
		numOfAgents = len(self.playerIds)
		numOfCoalitions = 1 << (numOfAgents)

		playersCurrState = {}
		for player in self.playerIds:
			playersCurrState[player] = self.playerModelBridge.getPlayerCurrState(player)


		# (the +- 1 accounts for non divisor cases that need one more/less member)
		adjustedMinSize = self.minNumberOfPlayersPerGroup
		adjustedMaxSize = self.maxNumberOfPlayersPerGroup
		if(adjustedMinSize == adjustedMaxSize and numOfAgents % adjustedMaxSize != 0):
			adjustedMinSize = adjustedMinSize - 1
			adjustedMaxSize = adjustedMaxSize + 1
				
		# initialize all coalitions
		for coalition in range(numOfCoalitions-1, 0, -1):
			group = self.getGroupFromBitFormat(coalition)
			groupInIds = self.convertFromByteToIds(group)

			currQuality = 0.0
			groupSize = len(group)

			# calculate the profile and characteristics only for groups in the range defined 
			if groupSize >= adjustedMinSize and groupSize <= adjustedMaxSize:
				# generate profile as average of the preferences estimates
				profile = self.interactionsProfileTemplate.generateCopy().reset()

				# if (self.regAlg.isTabular()):
				# 	profile = self.findBestProfileForGroup(groupInIds)

				# else:
				for currPlayer in groupInIds:
					preferences = self.playerPrefEstimates[currPlayer]
					for dim in profile.dimensions:
						profile.dimensions[dim] += (preferences.dimensions[dim] / groupSize)

				# calculate fitness and average state
				currAvgCharacteristics = PlayerCharacteristics()
				currAvgCharacteristics.reset()
				for i in range(groupSize):

					currState = playersCurrState[groupInIds[i]]
					currState.profile = profile

					currAvgCharacteristics.ability += currState.characteristics.ability / groupSize
					currAvgCharacteristics.engagement += currState.characteristics.engagement / groupSize


					if (self.regAlg.isTabular()):
						firstPlayerPreferences = self.playerPrefEstimates[groupInIds[i]]
						for j in range(i+1, groupSize):
							currQuality += self.regAlg.predict(firstPlayerPreferences, groupInIds[j]) / math.comb(groupSize, 2)

					elif (not self.regAlg.isGroupPredict()):
						currQuality += self.regAlg.predict(profile, groupInIds[i])

				if (self.regAlg.isGroupPredict()):
					currQuality += self.regAlg.groupPredict(groupInIds)
				
				diversityValueAlg = DiversityValueAlg(self.playerModelBridge, 0)
				personalities = diversityValueAlg.getPersonalitiesListFromPlayerIds(groupInIds)
				currAvgCharacteristics.group_diversity = diversityValueAlg.getTeamPersonalityDiveristy(personalities)

				self.coalitionsAvgCharacteristics[coalition] = currAvgCharacteristics
				self.coalitionsProfiles[coalition] = profile
			
			self.coalitionsValues[coalition] = currQuality

	def computeCoalitionsRestrictions(self):
		jointPlayersConstraintInBitFormat = self.jointPlayersConstraints[:]
		separatedPlayersConstraintInBitFormat = self.separatedPlayersConstraints[:]

		for i in range(len(jointPlayersConstraintInBitFormat)):
			jointPlayersConstraintInBitFormat[i] = self.convertFromIdsToBytes(jointPlayersConstraintInBitFormat[i])
			jointPlayersConstraintInBitFormat[i] = self.convertCoalitionFromByteToBitFormat(jointPlayersConstraintInBitFormat[i], len(jointPlayersConstraintInBitFormat[i]))

		for i in range(len(separatedPlayersConstraintInBitFormat)):
			separatedPlayersConstraintInBitFormat[i] = self.convertFromIdsToBytes(separatedPlayersConstraintInBitFormat[i])
			separatedPlayersConstraintInBitFormat[i] = self.convertCoalitionFromByteToBitFormat(separatedPlayersConstraintInBitFormat[i], len(separatedPlayersConstraintInBitFormat[i]))

		return jointPlayersConstraintInBitFormat, separatedPlayersConstraintInBitFormat


	def results(self, cSInByteFormat):
		bestGroups = []
		bestGroupsInBitFormat = []
		bestConfigProfiles = []
		avgCharacteristicsArray = []
	
		#this check was removed because of cases where the preferred group size would not match the integer divisor of the numPlayers
		#for coalition in cSInByteFormat:
			#if not (self.minNumberOfPlayersPerGroup <= len(coalition) <= self.maxNumberOfPlayersPerGroup):
				#return {"groups": [], "profiles": [], "avgCharacteristics": []}
		
		for coalition in cSInByteFormat:
			bestGroups.append(self.convertFromByteToIds(coalition))
			bestGroupsInBitFormat.append(self.convertCoalitionFromByteToBitFormat(coalition, len(coalition)))

		for group in bestGroupsInBitFormat:
			bestConfigProfiles.append(self.coalitionsProfiles[group])
			avgCharacteristicsArray.append(self.coalitionsAvgCharacteristics[group])

		return {"groups": bestGroups, "profiles": bestConfigProfiles, "avgCharacteristics": avgCharacteristicsArray}

	# function to compute best profile for group according to each players preferences about the task
	def findBestProfileForGroup(self, groupInIds):
		groupSize = len(groupInIds)
		bestProfile = self.interactionsProfileTemplate.generateCopy().reset()

		tasks = self.taskModelBridge.getAllTasksIds()
		
		for playerId in groupInIds:
			bestQuality = -1
			bestTaskId = -1
			
			for taskId in tasks:
				currQuality = self.regAlg.predictTasks(taskId, playerId)

				if currQuality > bestQuality:
					bestQuality = currQuality
					bestTaskId = taskId

			taskProfile = self.taskModelBridge.getTaskInteractionsProfile(bestTaskId)
			for dim in bestProfile.dimensions:
				bestProfile += taskProfile.dimensions[dim] / groupSize

		return bestProfile

	def organize(self):
		self.playerIds = self.playerModelBridge.getAllPlayerIds()
		for i in range(len(self.playerIds)):
			self.playerIds[i] = str(self.playerIds[i])
		self.numPlayers = len(self.playerIds)

		self.coalitionsProfiles = numpy.empty(1 << self.numPlayers, dtype=InteractionsProfile)
		self.coalitionsAvgCharacteristics = numpy.empty(1 << self.numPlayers, dtype=PlayerCharacteristics)
		self.coalitionsValues = numpy.empty(1 << self.numPlayers)

		# estimate preferences
		self.playerPrefEstimates = self.persEstAlg.updateEstimates()

		if (self.regAlg.isTabular()):
			self.regAlg.playerPrefEstimates = self.playerPrefEstimates
			
		# initialization(compute the value for every coalition between min and max number of players)
		self.computeAllCoalitionsValues()
		requiredJointPlayersInBitFormat, restrictedPlayersToJoinInBitFormat = self.computeCoalitionsRestrictions()


		bestCSFound_bitFormat = gs.odpip(self.numPlayers, self.minNumberOfPlayersPerGroup, self.maxNumberOfPlayersPerGroup, self.coalitionsValues.tolist(), requiredJointPlayersInBitFormat, restrictedPlayersToJoinInBitFormat)
		bestCSFound_byteFormat = self.convertSetOfCombinationsFromBitFormat(bestCSFound_bitFormat)

		del bestCSFound_bitFormat

		gc.collect()
		# i = 0
		# for coalition in bestCSFound_byteFormat:
		# 	print("{", end="")
		# 	for player in coalition:
		# 		preferences = self.playerModelBridge.getPlayerRealPreferences(player - 1)
		# 		print(str(player) + ",", end="")

		# 	print(self.coalitionsValues[bestCSFound_bitFormat[i]], end="")
		# 	print("}")

		
		return self.results(bestCSFound_byteFormat)


class CLink(ConfigsGenAlg):
	def __init__(self, 
		playerModelBridge, 
		interactionsProfileTemplate, 
		regAlg,
		persEstAlg,
		taskModelBridge = None,
		preferredNumberOfPlayersPerGroup = None, 
		minNumberOfPlayersPerGroup = None, 
		maxNumberOfPlayersPerGroup = None):

		super().__init__(playerModelBridge, 
		interactionsProfileTemplate, 
		taskModelBridge,
		preferredNumberOfPlayersPerGroup, 
		minNumberOfPlayersPerGroup, 
		maxNumberOfPlayersPerGroup)

		self.regAlg = regAlg
		self.persEstAlg = persEstAlg

		self.coalitionsProfiles = []
		self.coalitionsAvgCharacteristics = []
		self.coalitionsValues = []

		self.playerIds = []	

		self.playerPrefEstimates = {}


	def getCoalitionInByteFormatValue(self, coalitionInByteFormat):
		coalitionInBitFormat = self.convertCoalitionFromByteToBitFormat(coalitionInByteFormat, len(coalitionInByteFormat))
		return self.f[coalitionInBitFormat]


	def getCoalitionStructureInByteFormatValue(self, coalitionStructure):
		valueOfCS = 0
		for i in range(len(coalitionStructure)):
			valueOfCS += self.getCoalitionInByteFormatValue(coalitionStructure[i])
		
		return valueOfCS


	def convertCoalitionFromByteToBitFormat(self, coalitionInByteFormat, coalitionSize):
		coalitionInBitFormat = 0

		for i in range(coalitionSize):
			coalitionInBitFormat += 1 << (coalitionInByteFormat[i] - 1)

		return coalitionInBitFormat

	# convert group in bit format to group with the player ids
	def getGroupFromBitFormat(self, coalition):
		group = []
		tempCoalition = coalition
		playerNumber = 0
		while tempCoalition != 0:
			if tempCoalition & 1:
				group.append(playerNumber + 1)

			playerNumber += 1
			tempCoalition >>= 1

		
		return group

	def convertFromByteToIds(self, coalition):
		group = []

		for agent in coalition:
			group.append(self.playerIds[agent - 1])

		return group

	def getSizeOfCombinationInBitFormat(self, combinationInBitFormat):
		return bin(combinationInBitFormat).count("1")


	def convertSetOfCombinationsFromBitFormat(self, setOfCombinationsInBitFormat):
		setOfCombinationsInByteFormat = numpy.empty(len(setOfCombinationsInBitFormat), dtype=list)
		for i in range(len(setOfCombinationsInBitFormat)):

			setOfCombinationsInByteFormat[i] = self.getGroupFromBitFormat(setOfCombinationsInBitFormat[i])

		return setOfCombinationsInByteFormat

	def computeAllCoalitionsValues(self):
		numOfAgents = len(self.playerIds)
		numOfCoalitions = 1 << (numOfAgents)

		playersCurrState = {}
		for player in self.playerIds:
			playersCurrState[player] = self.playerModelBridge.getPlayerCurrState(player)

		# initialize all coalitions
		for coalition in range(numOfCoalitions-1, 0, -1):
			group = self.getGroupFromBitFormat(coalition)
			groupInIds = self.convertFromByteToIds(group)

			currQuality = 0.0
			groupSize = len(group)

			# calculate the profile and characteristics only for groups in the range defined
			if groupSize <= self.maxNumberOfPlayersPerGroup:	
				# generate profile as average of the preferences estimates
				profile = self.interactionsProfileTemplate.generateCopy().reset()

				# if (self.regAlg.isTabular()):
				# 	profile = self.findBestProfileForGroup(groupInIds)

				# else:
				for currPlayer in groupInIds:
					preferences = self.playerPrefEstimates[currPlayer]
					for dim in profile.dimensions:
						profile.dimensions[dim] += (preferences.dimensions[dim] / groupSize)

					
				# calculate fitness and average state
				currAvgCharacteristics = PlayerCharacteristics()
				currAvgCharacteristics.reset()
				for i in range(groupSize):

					currState = playersCurrState[groupInIds[i]]
					currState.profile = profile

					currAvgCharacteristics.ability += currState.characteristics.ability / groupSize
					currAvgCharacteristics.engagement += currState.characteristics.engagement / groupSize
				
					if (self.regAlg.isTabular()):
						firstPlayerPreferences = self.playerPrefEstimates[groupInIds[i]]
						for j in range(i+1, groupSize):
							currQuality += self.regAlg.predict(firstPlayerPreferences, groupInIds[j]) / math.comb(groupSize, 2)

					elif (not self.regAlg.isGroupPredict()):
						currQuality += self.regAlg.predict(profile, groupInIds[i])

				if (self.regAlg.isGroupPredict()):
					currQuality += self.regAlg.groupPredict(groupInIds)
			
				diversityValueAlg = DiversityValueAlg(self.playerModelBridge, 0)
				personalities = diversityValueAlg.getPersonalitiesListFromPlayerIds(groupInIds)
				currAvgCharacteristics.group_diversity = diversityValueAlg.getTeamPersonalityDiveristy(personalities)
						
				self.coalitionsAvgCharacteristics[coalition] = currAvgCharacteristics
				self.coalitionsProfiles[coalition] = profile
			
			self.coalitionsValues[coalition] = currQuality


	def results(self, cSInByteFormat):
		bestGroups = []
		bestGroupsInBitFormat = []
		bestConfigProfiles = []
		avgCharacteristicsArray = []
		for coalition in cSInByteFormat:
			bestGroups.append(self.convertFromByteToIds(coalition))
			bestGroupsInBitFormat.append(self.convertCoalitionFromByteToBitFormat(coalition, len(coalition)))

		for group in bestGroupsInBitFormat:
			bestConfigProfiles.append(self.coalitionsProfiles[group])
			avgCharacteristicsArray.append(self.coalitionsAvgCharacteristics[group])

		return {"groups": bestGroups, "profiles": bestConfigProfiles, "avgCharacteristics": avgCharacteristicsArray}

	# function to compute best profile for group according to each players preferences about the task
	def findBestProfileForGroup(self, groupInIds):
		groupSize = len(groupInIds)
		bestProfile = self.interactionsProfileTemplate.generateCopy().reset()

		tasks = self.taskModelBridge.getAllTasksIds()
		
		for playerId in groupInIds:
			bestQuality = -1
			bestTaskId = -1
			
			for taskId in tasks:
				currQuality = self.regAlg.predictTasks(taskId, playerId)

				if currQuality > bestQuality:
					bestQuality = currQuality
					bestTaskId = taskId

			taskProfile = self.taskModelBridge.getTaskInteractionsProfile(bestTaskId)
			for dim in bestProfile.dimensions:
				bestProfile += taskProfile.dimensions[dim] / groupSize

		return bestProfile

	def organize(self):
		self.playerIds = self.playerModelBridge.getAllPlayerIds()
		for i in range(len(self.playerIds)):
			self.playerIds[i] = str(self.playerIds[i])
		self.numPlayers = len(self.playerIds)

		self.coalitionsProfiles = numpy.empty(1 << self.numPlayers, dtype=InteractionsProfile)
		self.coalitionsAvgCharacteristics = numpy.empty(1 << self.numPlayers, dtype=PlayerCharacteristics)
		self.coalitionsValues = numpy.empty(1 << self.numPlayers)

		# estimate preferences
		self.playerPrefEstimates = self.persEstAlg.updateEstimates()

		if (self.regAlg.isTabular()):
			self.regAlg.playerPrefEstimates = self.playerPrefEstimates

		# initialization(compute the value for every coalition between min and max number of players)
		self.computeAllCoalitionsValues()

		bestCSFound_bitFormat = (gs.clink(self.numPlayers, self.minNumberOfPlayersPerGroup, self.maxNumberOfPlayersPerGroup, self.coalitionsValues.tolist()))
		bestCSFound_byteFormat = self.convertSetOfCombinationsFromBitFormat(bestCSFound_bitFormat)

		del bestCSFound_bitFormat
		
		gc.collect()


		return self.results(bestCSFound_byteFormat)
