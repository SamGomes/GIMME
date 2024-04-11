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


#hack for fetching the ModelMocks package on the previous directory
from pathlib import Path
sys.path.insert(1,str(Path(sys.path[0]).parent))

from GIMMECore import *
from ModelMocks import *
from LogManager import *


numRuns = 2

maxNumTrainingIterations = 20
numRealIterations = 20

preferredNumberOfPlayersPerGroup = 4


playerWindow = 10
numPlayers = 20

numTasks = 1


# ----------------------- [Init PRS] --------------------------------
numberOfConfigChoices = 100
numTestedPlayerProfilesInEst = 500


# ----------------------- [Init GA] --------------------------------
initialPopulationSize = 100 
numberOfEvolutionsPerIteration = 50
 
probOfCross = 0.65
probOfMutation = 0.15
# probReproduction = 1 - (probOfCross + probOfMutation) = 0.15

probOfMutationConfig = 0.8
probOfMutationGIPs = 0.4

numSurvivors = 10
numChildrenPerIteration = 100




simsID = str(os.getpid())

startTime = str(datetime.datetime.now())
newpath = sys.path[0]+"/analyzer/results/"
if not os.path.exists(newpath):
    os.makedirs(newpath)


# ----------------------- [Init Models] --------------------------------
print("Initing mocked models...")

listPlayers = []
for numPlayersToTest in range(4, 25, 4):
	listPlayers.append([0 for x in range(numPlayersToTest)])

players = [0 for x in range(numPlayers)]
tasks = [0 for x in range(numTasks)]


# ----------------------- [Init Model Bridges] --------------------------------
print("Initing model bridges...")

playerBridge = CustomPlayerModelBridge(players)
taskBridge = CustomTaskModelBridge(tasks)


# ----------------------- [Init Adaptations] --------------------------------
adaptationPRS = Adaptation()
listAdaptationPRS = []
adaptationGA_scx = Adaptation()
adaptationGA = Adaptation()
listAdaptationGA = []
adaptationODPIP = Adaptation()
listAdaptationODPIP = []
adaptationTabularODPIP = Adaptation()
listAdaptationTabularODPIP = []
adaptationCLink = Adaptation()
listAdaptationCLink = []
adaptationTabularCLink = Adaptation()
listAdaptationTabularCLink = []

adaptationEvl1D = Adaptation()
adaptationEvl3D = Adaptation()
adaptationEvl4D = Adaptation()
adaptationEvl5D = Adaptation()
adaptationEvl6D = Adaptation()

adaptationRandom = Adaptation()
listAdaptationRandom = []
adaptationAccurate = Adaptation()


allRealPreferences = []
allQuestionnairePreferences = []

# ----------------------- [Init Log Manager] --------------------------------
print("Initing .csv log manager...")
# logManager = MongoDBLogManager("mongodb+srv://studyAC1:studyAC1@cluster0-\
# nfksn.mongodb.net/test?retryWrites=true&w=majority")
logManager = CSVLogManager(newpath, simsID)



# ----------------------- [Init Algorithms] --------------------------------
print("Initing algorithms...")

syntergyTablePath = sys.path[0]+"/synergyTable.txt"

regAlg = KNNRegression(playerModelBridge = playerBridge, numberOfNNs = 5)
tabularRegAlg = TabularAgentSynergies(playerModelBridge= playerBridge, 
										taskModelBridge=taskBridge, 
										syntergyTablePath=syntergyTablePath)

# - - - - - 
intProfTemplate2D = InteractionsProfile({"dim_0": 0, "dim_1": 0})


evolutionaryConfigsAlg = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numChildrenPerIteration = numChildrenPerIteration,
	numSurvivors = numSurvivors,

	cxOp = "order",
	# jointPlayerConstraints="[15,1];[3,4]", 
	# separatedPlayerConstraints="[0,1]"
)
adaptationGA.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge,
	configsGenAlg = evolutionaryConfigsAlg, 
	name="GIMME_GA"
)


ODPIPconfigsAlg = ODPIPConfigsGen(
	playerModelBridge = playerBridge,
	interactionsProfileTemplate = intProfTemplate2D.generateCopy(),
	regAlg = regAlg,
	persEstAlg = ExplorationPreferencesEstAlg(
		playerModelBridge = playerBridge, 
		interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
		regAlg = regAlg,
		numTestedPlayerProfiles = numTestedPlayerProfilesInEst),
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup
)
adaptationODPIP.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge,
	configsGenAlg = ODPIPconfigsAlg, 
	name="GIMME_ODPIP"
)


tabularODPIPconfigsAlg = ODPIPConfigsGen(
	playerModelBridge = playerBridge,
	interactionsProfileTemplate = intProfTemplate2D.generateCopy(),
	regAlg = tabularRegAlg,
	persEstAlg = ExplorationPreferencesEstAlg(
		playerModelBridge = playerBridge, 
		interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
		regAlg = regAlg,
		numTestedPlayerProfiles = numTestedPlayerProfilesInEst),
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup
	# preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup,
	#jointPlayerConstraints="[15,1];[2,7];[3,4];[12,13];[14,15];[0,12];[9,15]",
	# separatedPlayerConstraints="[0,1]"
)
adaptationTabularODPIP.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge,
	configsGenAlg = tabularODPIPconfigsAlg, 
	name="GIMME_ODPIP"
)


CLinkconfigsAlg = CLinkConfigsGen(
	playerModelBridge = playerBridge,
	interactionsProfileTemplate = intProfTemplate2D.generateCopy(),
	regAlg = regAlg,
	persEstAlg = ExplorationPreferencesEstAlg(
		playerModelBridge = playerBridge, 
		interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
		regAlg = regAlg,
		numTestedPlayerProfiles = numTestedPlayerProfilesInEst),
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup
)
adaptationCLink.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge,
	configsGenAlg = CLinkconfigsAlg, 
	name="GIMME_CLink"
)

tabularCLinkconfigsAlg = CLinkConfigsGen(
	playerModelBridge = playerBridge,
	interactionsProfileTemplate = intProfTemplate2D.generateCopy(),
	regAlg = tabularRegAlg,
	persEstAlg = ExplorationPreferencesEstAlg(
		playerModelBridge = playerBridge, 
		interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
		regAlg = regAlg,
		numTestedPlayerProfiles = numTestedPlayerProfilesInEst),
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup
)
adaptationTabularCLink.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge,
	configsGenAlg = tabularCLinkconfigsAlg, 
	name="GIMME_CLink_Tabular"
)

prsConfigsAlg = PureRandomSearchConfigsGen(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
	regAlg = regAlg, 
	persEstAlg = ExplorationPreferencesEstAlg(
		playerModelBridge = playerBridge, 
		interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
		regAlg = regAlg,
		numTestedPlayerProfiles = numTestedPlayerProfilesInEst), 
	numberOfConfigChoices = numberOfConfigChoices, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup,
	# jointPlayerConstraints="[15,1]", 
	# separatedPlayerConstraints="[0,1]"
)
adaptationPRS.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge,
	configsGenAlg = prsConfigsAlg, 
	name="GIMME_PRS"
)


randomConfigsAlg = RandomConfigsGen(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate2D.generateCopy(), 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup,
	# jointPlayerConstraints="[15,1]",
	# separatedPlayerConstraints="[0,1]"
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
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numChildrenPerIteration = numChildrenPerIteration,
	numSurvivors = numSurvivors
)
adaptationEvl1D.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge, 
	configsGenAlg = evolutionaryConfigsAlg1D, 
	name="GIMME_GA1D"
)


# GIMME is the same as GIMME 2D 
intProfTemplate3D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0})
evolutionaryConfigsAlg3D = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate3D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numChildrenPerIteration = numChildrenPerIteration,
	numSurvivors = numSurvivors
)
adaptationEvl3D.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge, 
	configsGenAlg = evolutionaryConfigsAlg3D, 
	name="GIMME_GA3D"
)


intProfTemplate4D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0})
evolutionaryConfigsAlg4D = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate4D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numChildrenPerIteration = numChildrenPerIteration,
	numSurvivors = numSurvivors
)
adaptationEvl4D.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge, 
	configsGenAlg = evolutionaryConfigsAlg4D, 
	name="GIMME_GA4D"
)




intProfTemplate5D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0, "dim_4": 0})
evolutionaryConfigsAlg5D = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate5D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numChildrenPerIteration = numChildrenPerIteration,
	numSurvivors = numSurvivors
)
adaptationEvl5D.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge, 
	configsGenAlg = evolutionaryConfigsAlg5D, 
	name="GIMME_GA5D"
)



intProfTemplate6D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0, "dim_4": 0, "dim_5": 0})
evolutionaryConfigsAlg6D = EvolutionaryConfigsGenDEAP(
	playerModelBridge = playerBridge, 
	interactionsProfileTemplate = intProfTemplate6D.generateCopy(), 
	regAlg = regAlg, 
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup, 
	initialPopulationSize = initialPopulationSize, 
	numberOfEvolutionsPerIteration = numberOfEvolutionsPerIteration, 
	
	probOfCross = probOfCross, 
	probOfMutation = probOfMutation,

	probOfMutationConfig = probOfMutationConfig, 
	probOfMutationGIPs = probOfMutationGIPs, 
	
	numChildrenPerIteration = numChildrenPerIteration,
	numSurvivors = numSurvivors
)
adaptationEvl6D.init(
	playerModelBridge = playerBridge, 
	taskModelBridge = taskBridge, 
	configsGenAlg = evolutionaryConfigsAlg6D, 
	name="GIMME_GA6D"
)



# ----------------------- [Simulation Methods] --------------------------------

def simulateReaction(playerBridge, currIteration, playerId):
	currState = playerBridge.getPlayerCurrState(playerId)
	newState = calcReaction(
		playerBridge = playerBridge, 
		state = currState, 
		playerId = playerId, 
		currIteration = currIteration)

	increases = PlayerState(stateType = newState.stateType)
	increases.profile = currState.profile
	increases.characteristics = PlayerCharacteristics(ability=(newState.characteristics.ability - currState.characteristics.ability), engagement=newState.characteristics.engagement)
	playerBridge.setAndSavePlayerStateToDataFrame(playerId, increases, newState)	
	return increases

def calcReaction(playerBridge, state, playerId, currIteration):
	preferences = playerBridge.getPlayerRealPreferences(playerId)
	numDims = len(preferences.dimensions)
	newState = PlayerState(
		stateType = 1, 
		characteristics = PlayerCharacteristics(
			ability=state.characteristics.ability, 
			engagement=state.characteristics.engagement
			), 
		profile=state.profile)
	newState.characteristics.engagement = 1 - (preferences.distanceBetween(state.profile) / math.sqrt(numDims))  #between 0 and 1
	if newState.characteristics.engagement>1:
		breakpoint()
	abilityIncreaseSim = (newState.characteristics.engagement*playerBridge.getBaseLearningRate(playerId))
	newState.characteristics.ability = newState.characteristics.ability + abilityIncreaseSim
	return newState

def executionPhase(numRuns, playerBridge, maxNumIterations, startingI, currRun, adaptation):
	if(maxNumIterations <= 0):
		return

	numPlayersToTest = len(playerBridge.getAllPlayerIds())
	
	i = startingI
	while(i < maxNumIterations + startingI):

		if adaptation.name == "accurate":
			adaptation.configsGenAlg.updateCurrIteration(i)
		
		print("Process ["+simsID+"] performing step (" +str(i - startingI)+ " of "+str(maxNumIterations)+") of run ("+str(currRun+1)+" of "+str(numRuns)+") of algorithm \""+str(adaptation.name)+"\"...                                                             ", end="\r")
		

		start = time.time()
		groups = adaptation.iterate()
		end = time.time()
		deltaTime = (end - start)


		for x in range(numPlayersToTest):
			increases = simulateReaction(playerBridge, i, x)
			logManager.writeToLog("", "results", 
				{
					"simsID": str(simsID),
					"algorithm": adaptation.name,
					"run": str(currRun),
					"iteration": str(i),
					"playerID": str(x),
					"abilityInc": str(increases.characteristics.ability),
					"engagementInc": str(increases.characteristics.engagement),
					"profDiff": str(playerBridge.getPlayerRealPreferences(x).distanceBetween(playerBridge.getPlayerCurrProfile(x))),
					"iterationElapsedTime": str(deltaTime)
				})		
		i+=1


def executeSimulations(numRuns, profileTemplate, maxNumTrainingIterations, firstTrainingI, numRealIterations, firstRealI,\
	playerBridge, taskBridge, adaptation, estimatorsAccuracy = None, considerExtremePreferencesValues = None):

	estimatorsAccuracy = 0.1 if estimatorsAccuracy == None else estimatorsAccuracy
	considerExtremePreferencesValues = False if considerExtremePreferencesValues == None else considerExtremePreferencesValues

	adaptationName = adaptation.name

	numInteractionDimensions = len(profileTemplate.dimensions.keys())

	numPlayersToTest = len(playerBridge.getAllPlayerIds())


	# re-init stuff
	players = [0 for x in range(numPlayers)]
	tasks = [0 for x in range(numTasks)]

	# create players and tasks
	for x in range(numPlayersToTest):
		playerBridge.registerNewPlayer(
			playerId = str(x), 
			name = "name", 
			currState = PlayerState(profile = profileTemplate.generateCopy().reset()), 
			pastModelIncreasesDataFrame = PlayerStatesDataFrame(
				interactionsProfileTemplate = profileTemplate.generateCopy().reset(), 
				trimAlg = ProximitySortPlayerDataTrimAlg(
					maxNumModelElements = playerWindow, 
					epsilon = 0.005
					)
				), 
			currModelIncreases = PlayerCharacteristics(), 
			preferencesEst = profileTemplate.generateCopy().reset(), 
			realPreferences = profileTemplate.generateCopy().reset())

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
		allRealPreferences = []
		allQuestionnairePreferences = []

		EPdimensions = [{"dim_0":0,"dim_1":0}, {"dim_0":1,"dim_1":0}, {"dim_0":0,"dim_1":1}, {"dim_0":1,"dim_1":1}, {"dim_0":1,"dim_1":1}]	
		
		playersDimsStr = "players: [\n"	


		for x in range(numPlayersToTest):
			profile = profileTemplate.generateCopy().reset()
			if(considerExtremePreferencesValues):
				profile.dimensions = EPdimensions[x % len(EPdimensions)]
				playersDimsStr += "{"+str(profile.dimensions)+"},\n"
			else:
				for d in range(numInteractionDimensions):
					profile.dimensions["dim_"+str(d)] = random.uniform(0, 1)


			allRealPreferences.append(profile)
			# allRealPreferences[x].normalize()



			profile = profileTemplate.generateCopy().reset()
			currRealPreferences = allRealPreferences[x]
			for d in range(numInteractionDimensions):
				profile.dimensions["dim_"+str(d)] = numpy.clip(random.gauss(currRealPreferences.dimensions["dim_"+str(d)], estimatorsAccuracy), 0, 1)
			allQuestionnairePreferences.append(profile)
			# allQuestionnairePreferences[x].normalize()
		

			# init players including predicted preferences
			playerBridge.resetPlayer(x)

			playerBridge.setPlayerPreferencesEst(x, profileTemplate.generateCopy().init())
			# realPreferences = allRealPreferences[x]
			# playerBridge.setPlayerRealPreferences(x, realPreferences)

			questionnairePreferences = allQuestionnairePreferences[x]
			playerBridge.setPlayerRealPreferences(x, questionnairePreferences)
			playerBridge.setBaseLearningRate(x, 0.5)

			playerBridge.getPlayerStatesDataFrame(x).trimAlg.considerStateResidue(False)

		playersDimsStr += "],\n"

		# print(playersDimsStr)

		if(maxNumTrainingIterations > 0):		
			adaptation.bootstrap(maxNumTrainingIterations)

		# change for "real" preferences from which the predictions supposidely are based on...
		for x in range(numPlayersToTest):
			playerBridge.resetState(x)

			realPreferences = allRealPreferences[x]
			playerBridge.setPlayerRealPreferences(x, realPreferences)
			playerBridge.setBaseLearningRate(x, random.gauss(0.5, 0.05))

			playerBridge.getPlayerStatesDataFrame(x).trimAlg.considerStateResidue(True)
		
		if r > 0:
			adaptation.configsGenAlg.reset()

		executionPhase(numRuns, playerBridge, numRealIterations, firstRealI, r, adaptation)







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
	print("All results are saved to \'"+newpath+"\'.")
	print("------------------------------------------")


	# ----------------------- [Execute Algorithms] ----------------------------

	# inputtedText = input("<<< All ready! Press Enter to start (Q, then Enter exits the application). >>>")
	# if (inputtedText== "Q"):
		# exit()

	

	# Explore Base GIMME
	# adaptationPRS.name = "Random"
	# executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
	#  	playerBridge, taskBridge, adaptationRandom)
 #
	# adaptationPRS.name = "GIMME_PRS"
	# executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
	#  	playerBridge, taskBridge, adaptationPRS)
 #
	# adaptationGA.name = "GIMME_GA"
	# executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
	# 	playerBridge, taskBridge, adaptationGA)
 # 
	adaptationODPIP.name = "GIMME_ODPIP"
	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationODPIP)
 # 
	# adaptationTabularODPIP.name = "GIMME_ODPIP_Tabular"
	# executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
	# 	playerBridge, taskBridge, adaptationTabularODPIP)
 # 
	# adaptationCLink.name = "GIMME_CLink"
	# executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
	#    	playerBridge, taskBridge, adaptationCLink)
 # 
	# adaptationTabularCLink.name = "GIMME_CLink_Tabular"
	# executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
	# 	playerBridge, taskBridge, adaptationTabularCLink)




	# Explore GIMME-Bootstrap
	adaptationODPIP.name = "GIMME_ODPIP_Bootstrap"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationODPIP, estimatorsAccuracy = 0.1)

	quit()
	
	adaptationODPIP.name = "GIMME_ODPIP_Bootstrap_HighAcc"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationODPIP, estimatorsAccuracy = 0.05)

	adaptationODPIP.name = "GIMME_ODPIP_Bootstrap_LowAcc"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationODPIP, estimatorsAccuracy = 0.2)


	adaptationCLink.name = "GIMME_CLink_Bootstrap"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationCLink, estimatorsAccuracy = 0.1)

	adaptationCLink.name = "GIMME_CLink_Bootstrap_HighAcc"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationCLink, estimatorsAccuracy = 0.05)

	adaptationCLink.name = "GIMME_CLink_Bootstrap_LowAcc"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationCLink, estimatorsAccuracy = 0.2)


	adaptationGA.name = "GIMME_GA_Bootstrap"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationGA, estimatorsAccuracy = 0.1)

	adaptationGA.name = "GIMME_GA_Bootstrap_HighAcc"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationGA, estimatorsAccuracy = 0.05)

	adaptationGA.name = "GIMME_GA_Bootstrap_LowAcc"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationGA, estimatorsAccuracy = 0.2)


	adaptationPRS.name = "GIMME_PRS_Bootstrap"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationPRS, estimatorsAccuracy = 0.1)

	adaptationPRS.name = "GIMME_PRS_Bootstrap_HighAcc"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationPRS, estimatorsAccuracy = 0.05)

	adaptationPRS.name = "GIMME_PRS_Bootstrap_LowAcc"
	executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
					playerBridge, taskBridge, adaptationPRS, estimatorsAccuracy = 0.2)



	# Explore GIP number of dimensions
	executeSimulations(numRuns, intProfTemplate1D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationEvl1D)

	executeSimulations(numRuns, intProfTemplate3D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationEvl3D)

	executeSimulations(numRuns, intProfTemplate4D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationEvl4D)

	executeSimulations(numRuns, intProfTemplate5D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationEvl5D)

	executeSimulations(numRuns, intProfTemplate6D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationEvl6D)



	# Explore GIMME with extreme profiles
	adaptationCLink.name = "GIMME_CLink_EP"
	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationCLink, considerExtremePreferencesValues = True)


	adaptationODPIP.name = "GIMME_ODPIP_EP"
	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationODPIP, considerExtremePreferencesValues = True)


	adaptationTabularODPIP.name = "GIMME_Tabular_ODPIP_EP"
	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationTabularODPIP, considerExtremePreferencesValues = True)

	adaptationGA.name = "GIMME_GA_EP"
	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationGA, considerExtremePreferencesValues = True)


	adaptationPRS.name = "GIMME_PRS_EP"
	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationPRS, considerExtremePreferencesValues = True)

	adaptationPRS.name = "Random_EP"
	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
		playerBridge, taskBridge, adaptationRandom, considerExtremePreferencesValues = True)

	# adaptationTabularODPIP.configsGenAlg.resetPlayersConstraints()

	print("Done!                        ")


		
