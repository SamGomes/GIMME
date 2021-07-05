import random
import json
import time
import matplotlib.pyplot as plt
import math
import os
import sys
import datetime
import numpy


import matplotlib.pyplot as plt
from numpy import array
import matplotlib.collections as collections

sys.path.insert(1,'/home/samgomes/Documents/doutoramento/reps/GIMME/GIMME')
sys.path.insert(1,'/GIMME')
sys.path.insert(1,'../')
from GIMMECore import *
from ModelMocks import *
from LogManager import *


numRuns = 10

maxNumTrainingIterations = 20
numRealIterations = 20

preferredNumberOfPlayersPerGroup = 4
numberOfConfigChoices = 100

numTestedPlayerProfilesInEst = 500

playerWindow = 10
numPlayers = 23

numTasks = 1


# ----------------------- [Init GA] --------------------------------
initialPopulationSize = 200 
numberOfEvolutionsPerIteration = 30
 
probOfCross = 0.7
probOfMutation = 1.0

probOfMutationConfig = 0.35 
probOfMutationGIPs = 0.35 

numFitSurvivors = 15




startTime = str(datetime.datetime.now())
newpath = "./simulationResults/latestResults/"
if not os.path.exists(newpath):
    os.makedirs(newpath)


# ----------------------- [Init Models] --------------------------------
print("Initing mocked models...")

players = [0 for x in range(numPlayers)]
playersGrid = [0 for x in range(numPlayers)]
tasks = [0 for x in range(numTasks)]

# ----------------------- [Init Model Bridges] --------------------------------
print("Initing model bridges...")

playerBridgeGrid = CustomPlayerModelBridge(playersGrid)
playerBridge = CustomPlayerModelBridge(players)

taskBridge = CustomTaskModelBridge(tasks)


# ----------------------- [Init Adaptations] --------------------------------
adaptationSH = Adaptation()
adaptationEvl = Adaptation()

adaptationEvl1D = Adaptation()
adaptationEvl3D = Adaptation()
adaptationEvl4D = Adaptation()
adaptationEvl5D = Adaptation()
adaptationEvl6D = Adaptation()

adaptationRandom = Adaptation()
adaptationAccurate = Adaptation()


realPersonalities = []
questionnairePersonalities = []

# ----------------------- [Init Log Manager] --------------------------------
print("Initing .csv log manager...")
# logManager = MongoDBLogManager("mongodb+srv://studyAC1:studyAC1@cluster0-\
# nfksn.mongodb.net/test?retryWrites=true&w=majority")
logManager = CSVLogManager(newpath)



# ----------------------- [Init Algorithms] --------------------------------
print("Initing algorithms...")

regAlg = KNNRegression(playerModelBridge = playerBridge, numberOfNNs = 5)

# - - - - - 
intProfTemplate2D = InteractionsProfile({"dim_0": 0, "dim_1": 0})

evolutionaryConfigsAlg = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	qualityWeights = PlayerCharacteristics(ability=0.5, engagement=0.5),
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numFitSurvivors = numFitSurvivors
)
adaptationEvl.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge,
	configsGenAlg = evolutionaryConfigsAlg, 
	name="GIMME_Evl"
)

simpleConfigsAlg = StochasticHillclimberConfigsGen(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
	regAlg = regAlg, 
	persEstAlg = ExplorationPersonalityEstAlg(
		playerModelBridge = playerBridge, 
		interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
		regAlg = regAlg,
		numTestedPlayerProfiles = numTestedPlayerProfilesInEst, 
		qualityWeights = PlayerCharacteristics(ability=0.5, engagement=0.5)), 
	numberOfConfigChoices = numberOfConfigChoices, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	qualityWeights = PlayerCharacteristics(ability=0.5, engagement=0.5)
)
adaptationSH.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge,
	configsGenAlg = simpleConfigsAlg, 
	name="GIMME_SH"
)


randomConfigsAlg = RandomConfigsGen(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup
)
adaptationRandom.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge,
	configsGenAlg = randomConfigsAlg, 
	name="Random"
)




# - - - - -
intProfTemplate1D = InteractionsProfile({"dim_0": 0})
evolutionaryConfigsAlg1D = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate1D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	qualityWeights = PlayerCharacteristics(ability=0.5, engagement=0.5),
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numFitSurvivors = numFitSurvivors
)
adaptationEvl1D.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge, 
	configsGenAlg = evolutionaryConfigsAlg1D, 
	name="GIMME_Evl1D"
)


# GIMME is the same as GIMME 2D 

intProfTemplate3D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0})
evolutionaryConfigsAlg3D = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate3D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	qualityWeights = PlayerCharacteristics(ability=0.5, engagement=0.5),
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numFitSurvivors = numFitSurvivors
)
adaptationEvl3D.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge, 
	configsGenAlg = evolutionaryConfigsAlg3D, 
	name="GIMME_Evl3D"
)


intProfTemplate4D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0})
evolutionaryConfigsAlg4D = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate4D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	qualityWeights = PlayerCharacteristics(ability=0.5, engagement=0.5),
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numFitSurvivors = numFitSurvivors
)
adaptationEvl4D.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge, 
	configsGenAlg = evolutionaryConfigsAlg4D, 
	name="GIMME_Evl4D"
)




intProfTemplate5D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0, "dim_4": 0})
evolutionaryConfigsAlg5D = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate5D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	qualityWeights = PlayerCharacteristics(ability=0.5, engagement=0.5),
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numFitSurvivors = numFitSurvivors
)
adaptationEvl5D.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge, 
	configsGenAlg = evolutionaryConfigsAlg5D, 
	name="GIMME_Evl5D"
)



intProfTemplate6D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0, "dim_4": 0, "dim_5": 0})
evolutionaryConfigsAlg6D = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate6D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	qualityWeights = PlayerCharacteristics(ability=0.5, engagement=0.5),
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numFitSurvivors = numFitSurvivors
)
adaptationEvl6D.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge, 
	configsGenAlg = evolutionaryConfigsAlg6D, 
	name="GIMME_Evl6D"
)



# ----------------------- [Simulation Methods] --------------------------------

def simulateReaction(isBootstrap, playerBridge, currIteration, playerId):
	currState = playerBridge.getPlayerCurrState(playerId)
	newState = calcReaction(
		isBootstrap = isBootstrap, 
		playerBridge = playerBridge, 
		state = currState, 
		playerId = playerId, 
		currIteration = currIteration)

	increases = PlayerState(stateType = newState.stateType)
	increases.profile = currState.profile
	increases.characteristics = PlayerCharacteristics(ability=(newState.characteristics.ability - currState.characteristics.ability), engagement=newState.characteristics.engagement)
	playerBridge.setAndSavePlayerStateToGrid(playerId, increases, newState)	
	return increases

def calcReaction(isBootstrap, playerBridge, state, playerId, currIteration):
	personality = playerBridge.getPlayerRealPersonality(playerId)
	numDims = len(personality.dimensions)
	newStateType = 0 if isBootstrap else 1
	newState = PlayerState(
		stateType = newStateType, 
		characteristics = PlayerCharacteristics(
			ability=state.characteristics.ability, 
			engagement=state.characteristics.engagement
			), 
		profile=state.profile)
	newState.characteristics.engagement = 1 - (personality.distanceBetween(state.profile) / math.sqrt(numDims))  #between 0 and 1
	if newState.characteristics.engagement>1:
		breakpoint()
	abilityIncreaseSim = (newState.characteristics.engagement*playerBridge.getBaseLearningRate(playerId))
	newState.characteristics.ability = newState.characteristics.ability + abilityIncreaseSim
	return newState

def executionPhase(numRuns, isBootstrap, playerBridge, maxNumIterations, startingI, currRun, adaptation):
	if(maxNumIterations <= 0):
		return

	i = startingI
	while(i < maxNumIterations + startingI):

		if adaptation.name == "accurate":
			adaptation.configsGenAlg.updateCurrIteration(i)
		
		simsID = str(os.getpid())
		print("Process ["+simsID+"] performing step (" +str(i - startingI)+ " of "+str(maxNumIterations)+") of run ("+str(currRun+1)+" of "+str(numRuns)+") of algorithm \""+str(adaptation.name)+"\"...                                                             ", end="\r")
		adaptation.iterate()

		for x in range(numPlayers):
			increases = simulateReaction(isBootstrap, playerBridge, i, x)
			logManager.writeToLog("GIMMESims", "resultsEvl", 
				{
					"simsID": str(simsID),
					"algorithm": adaptation.name,
					"run": str(currRun),
					"iteration": str(i),
					"playerID": str(x),
					"abilityInc": str(increases.characteristics.ability),
					"engagementInc": str(increases.characteristics.engagement),
					"profDiff": str(playerBridge.getPlayerRealPersonality(x).distanceBetween(playerBridge.getPlayerCurrProfile(x)))
				})		
		i+=1


def executeSimulations(numRuns, profileTemplate, maxNumTrainingIterations, firstTrainingI, numRealIterations, firstRealI,\
	playerBridge, taskBridge, adaptation, estimatorsAccuracy = None, considerExtremePersonalityValues = None):

	estimatorsAccuracy = 0.1 if estimatorsAccuracy == None else estimatorsAccuracy
	considerExtremePersonalityValues = False if considerExtremePersonalityValues == None else considerExtremePersonalityValues

	adaptationName = adaptation.name

	numInteractionDimensions = len(profileTemplate.dimensions.keys())

	# create players and tasks
	for x in range(numPlayers):
		playerBridge.registerNewPlayer(
			playerId = int(x), 
			name = "name", 
			currState = PlayerState(profile = profileTemplate.generateCopy().reset()), 
			pastModelIncreasesGrid = PlayerStatesDataFrame(
				interactionsProfileTemplate = profileTemplate.generateCopy().reset(), 
				gridTrimAlg = QualitySortPlayerDataTrimAlg(
				# gridTrimAlg = AgeSortPlayerDataTrimAlg(
					maxNumModelElements = playerWindow, 
					qualityWeights = PlayerCharacteristics(ability=0.5, engagement=0.5)
					)
				), 
			currModelIncreases = PlayerCharacteristics(), personalityEst = profileTemplate.generateCopy().reset(), realPersonality = profileTemplate.generateCopy().reset())
	
	for x in range(numTasks):
		taskBridge.registerNewTask(
			taskId = int(x), 
			description = "description", 
			minRequiredAbility = random.uniform(0, 1), 
			profile = profileTemplate.generateCopy(), 
			minDuration = datetime.timedelta(minutes=1), 
			difficultyWeight = 0.5, 
			profileWeight = 0.5)



	for r in range(numRuns):
		realPersonalities = []
		questionnairePersonalities = []

		EPdimensions = [{"dim_0":1,"dim_1":0,},{"dim_0":0,"dim_1":1}]		
		EPdimensionsAux = EPdimensions.copy()	
		
		playersDimsStr = "players: [\n"	


		for x in range(numPlayers):
			profile = profileTemplate.generateCopy().reset()
			if(considerExtremePersonalityValues):
				if(len(EPdimensionsAux) == 0):
					EPdimensionsAux = EPdimensions.copy()
				profile.dimensions = EPdimensionsAux.pop()
				playersDimsStr += "{"+str(profile.dimensions)+"},\n"
			else:
				for d in range(numInteractionDimensions):
					profile.dimensions["dim_"+str(d)] = random.uniform(0, 1)
			realPersonalities.append(profile)
			realPersonalities[x].normalize()


			profile = profileTemplate.generateCopy().reset()
			currRealPersonality = realPersonalities[x]
			for d in range(numInteractionDimensions):
				profile.dimensions["dim_"+str(d)] = numpy.clip(random.gauss(currRealPersonality.dimensions["dim_"+str(d)], estimatorsAccuracy), 0, 1)
			questionnairePersonalities.append(profile)
			questionnairePersonalities[x].normalize()
		

			# init players including predicted personality
			playerBridge.resetPlayer(x)

			playerBridge.setPlayerPersonalityEst(x, profileTemplate.generateCopy().init())
			# realPersonality = realPersonalities[x]
			# playerBridge.setPlayerRealPersonality(x, realPersonality)

			questionnairePersonality = questionnairePersonalities[x]
			playerBridge.setPlayerRealPersonality(x, questionnairePersonality)
			playerBridge.setBaseLearningRate(x, 0.5)

			playerBridge.getPlayerStatesDataFrame(x).gridTrimAlg.considerStateResidue(False)

		playersDimsStr += "],\n"
		# print(playersDimsStr)


		if(maxNumTrainingIterations > 0):		
			adaptation.bootstrap(maxNumTrainingIterations)

		# change for "real" personality from which the predictions supposidely are based on...
		for x in range(numPlayers):
			playerBridge.resetState(x)

			realPersonality = realPersonalities[x]
			playerBridge.setPlayerRealPersonality(x, realPersonality)
			playerBridge.setBaseLearningRate(x, random.gauss(0.5, 0.05))

			playerBridge.getPlayerStatesDataFrame(x).gridTrimAlg.considerStateResidue(True)
		
		if r > 0:
			adaptation.configsGenAlg.reset()

		executionPhase(numRuns, False, playerBridge, numRealIterations, firstRealI, r, adaptation)






# ----------------------- [Simulation] --------------------------------
if __name__ == '__main__':

	print("------------------------------------------")
	print("-----                                -----")
	print("-----       GIMME API EXAMPLE        -----")
	print("-----                                -----")
	print("-----      (SIMULATING A CLASS)      -----")
	print("-----                                -----")
	print("------------------------------------------")


	print("------------------------------------------")
	print("NOTE: This example tests several group organization algorithms types.")
	print("For more details, consult the source code.")
	print("------------------------------------------")


	# ----------------------- [Execute Algorithms] ----------------------------

	input("<<< All ready! Press any key to start. >>>")

	# import time
	# startTime = time.time()

	adaptationEvl.name = "GIMME_Evl"
	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations, 
		playerBridge, taskBridge, adaptationEvl)

	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations, 
		playerBridge, taskBridge, adaptationSH)

	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationRandom)


	# adaptationEvl.name = "GIMME_Evl_Bootstrap"
	# executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations, 
	# 				playerBridge, taskBridge, adaptationEvl, estimatorsAccuracy = 0.1)

	# adaptationEvl.name = "GIMME_Evl_Bootstrap_HighAcc"
	# executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations, 
	# 				playerBridge, taskBridge, adaptationEvl, estimatorsAccuracy = 0.05)

	# adaptationEvl.name = "GIMME_Evl_Bootstrap_LowAcc"
	# executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations, 
	# 				playerBridge, taskBridge, adaptationEvl, estimatorsAccuracy = 0.2)


	# executeSimulations(numRuns, intProfTemplate1D, 0, 0, numRealIterations, maxNumTrainingIterations, 
	# 	playerBridge, taskBridge, adaptationEvl1D)

	# executeSimulations(numRuns, intProfTemplate3D, 0, 0, numRealIterations, maxNumTrainingIterations, 
	# 	playerBridge, taskBridge, adaptationEvl3D)

	# executeSimulations(numRuns, intProfTemplate4D, 0, 0, numRealIterations, maxNumTrainingIterations, 
	# 	playerBridge, taskBridge, adaptationEvl4D)

	# executeSimulations(numRuns, intProfTemplate5D, 0, 0, numRealIterations, maxNumTrainingIterations, 
	# 	playerBridge, taskBridge, adaptationEvl5D)

	# executeSimulations(numRuns, intProfTemplate6D, 0, 0, numRealIterations, maxNumTrainingIterations, 
	# 	playerBridge, taskBridge, adaptationEvl6D)



	# adaptationEvl.name = "GIMME_Evl_EP"
	# executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations, 
	# 	playerBridge, taskBridge, adaptationEvl, considerExtremePersonalityValues = True)

	# executionTime = (time.time() - startTime)
	# print('Execution time in seconds: ' + str(executionTime)+"                                                                                    ", end="\r")

	print("Done!                        ")


		