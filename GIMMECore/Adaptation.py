import math
from .AlgDefStructs.RegressionAlg import *
from .AlgDefStructs.ConfigsGenAlg import *
from .AlgDefStructs.PreferencesEstAlg import *

from .ModelBridge.PlayerModelBridge import PlayerModelBridge 
from .ModelBridge.TaskModelBridge import TaskModelBridge 


class Adaptation(object):

	def init(self, \
		playerModelBridge, \
		taskModelBridge, \
		name, \
		configsGenAlg):

		self.initialized = True
		self.playerIds = []
		self.taskIds = []
		self.name = name

		# self.numTasksPerGroup = numTasksPerGroup;
		self.configsGenAlg = configsGenAlg
		self.playerModelBridge = playerModelBridge
		self.taskModelBridge = taskModelBridge

		self.configsGenAlg.init()	


	# instrumentation
	def getConfigsGenAlgCompPerc(self):
		return self.configsGenAlg.getCompPercentage()

	def iterate(self):
		if not self.initialized:
			raise AssertionError('Adaptation not Initialized! Core not executed.') 
			return
		
		self.playerIds = self.playerModelBridge.getAllPlayerIds()
		self.taskIds = self.taskModelBridge.getAllTaskIds()

		if len(self.playerIds) < self.configsGenAlg.minNumberOfPlayersPerGroup:
			raise ValueError('Not enough players to form a group.') 
			return
		

		# print(json.dumps(self.playerModelBridge.getPlayerStatesDataFrame(0).states, default=lambda o: [o.__dict__["quality"],o.__dict__["stateType"],o.__dict__["creationTime"]], sort_keys=True))
		# print("\n\n")
		adaptedConfig = self.configsGenAlg.organize()

		adaptedGroups = adaptedConfig["groups"]
		adaptedProfiles = adaptedConfig["profiles"]
		adaptedAvgCharacteristics = adaptedConfig["avgCharacteristics"]
		adaptedConfig["tasks"] = []

		for groupIndex in range(len(adaptedGroups)):
			currGroup = adaptedGroups[groupIndex]
			groupProfile = adaptedProfiles[groupIndex]
			avgState = adaptedAvgCharacteristics[groupIndex]

			adaptedTaskId = self.selectTask(self.taskIds, groupProfile, avgState)
			for playerId in currGroup:

				currState = self.playerModelBridge.getPlayerCurrState(playerId)
				currState.profile = groupProfile	
				self.playerModelBridge.setPlayerTasks(playerId, [adaptedTaskId])
				self.playerModelBridge.setPlayerCharacteristics(playerId, currState.characteristics)
				self.playerModelBridge.setPlayerProfile(playerId, currState.profile)
				self.playerModelBridge.setPlayerGroup(playerId, currGroup)
				
			adaptedConfig["tasks"].append(adaptedTaskId)

			
		# totalFitness = 0.0
		# for groupI in range(len(adaptedGroups)):
		# 	group = adaptedGroups[groupI]
		# 	profile = adaptedProfiles[groupI]
		# 	for playerId in group:
		# 		predictedIncreases = self.configsGenAlg.regAlg.predict(profile, playerId)
		# 		totalFitness += (0.5* predictedIncreases.characteristics.ability + \
		# 						0.5* predictedIncreases.characteristics.engagement)
		
		# totalFitness = totalFitness + 1.0 #helps selection (otherwise Pchoice would always be 0)
		# print(totalFitness, end="\n")




		return adaptedConfig

	def selectTask(self,
		possibleTaskIds,
		bestConfigProfile,
		avgState):
		lowestCost = math.inf
		bestTaskId = -1 #if no tasks are available 

		for i in range(len(possibleTaskIds)):
			currTaskId = possibleTaskIds[i]

			cost = abs(bestConfigProfile.sqrDistanceBetween(self.taskModelBridge.getTaskInteractionsProfile(currTaskId)) * self.taskModelBridge.getTaskProfileWeight(currTaskId))
			cost += abs(avgState.ability - self.taskModelBridge.getMinTaskRequiredAbility(currTaskId) * self.taskModelBridge.getTaskDifficultyWeight(currTaskId))

			if cost < lowestCost:
				lowestCost = cost
				bestTaskId = currTaskId
				
		return bestTaskId





	# Bootstrap
	def simulateReaction(self, playerId):
		currState = self.playerModelBridge.getPlayerCurrState(playerId)
		newState = self.calcReaction(state = currState, playerId = playerId)

		increases = PlayerState(stateType = newState.stateType)
		increases.profile = currState.profile
		increases.characteristics = PlayerCharacteristics(ability=(newState.characteristics.ability - currState.characteristics.ability), engagement=newState.characteristics.engagement)
		self.playerModelBridge.setAndSavePlayerStateToDataFrame(playerId, increases, newState)	
		return increases

	def calcReaction(self, state, playerId):
		preferences = self.playerModelBridge.getPlayerRealPreferences(playerId)
		numDims = len(preferences.dimensions)
		newState = PlayerState(
			stateType = 0, 
			characteristics = PlayerCharacteristics(
				ability=state.characteristics.ability, 
				engagement=state.characteristics.engagement
				), 
			profile=state.profile)
		newState.characteristics.engagement = 1 - (preferences.distanceBetween(state.profile) / math.sqrt(numDims))  #between 0 and 1
		if newState.characteristics.engagement>1:
			raise ValueError('Something went wrong. Engagement is > 1.') 
		abilityIncreaseSim = (newState.characteristics.engagement*self.playerModelBridge.getBaseLearningRate(playerId))
		newState.characteristics.ability = newState.characteristics.ability + abilityIncreaseSim
		return newState

	def bootstrap(self, numBootstrapIterations):
		if(numBootstrapIterations <= 0):
			raise ValueError('Number of bootstrap iterations must be higher than 0 for this method to be called.') 
			return

		numPlayers = len(self.playerModelBridge.getAllPlayerIds())
		i = 0
		while(i < numBootstrapIterations):
			print("Performming step ("+str(i)+" of "+str(numBootstrapIterations)+") of the bootstrap phase of \""+str(self.name)+"\"...                                                             ", end="\r")
			self.iterate()
			for x in range(numPlayers):
				increases = self.simulateReaction(playerId=x)	
			i+=1


		self.configsGenAlg.reset()
