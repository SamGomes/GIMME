import random
import json
import os
import sys
import textwrap

from datetime import datetime, timedelta

import itertools
import threading
import time
import sys

sys.path.insert(1,sys.path[0].rsplit('/',2)[0])

#hack for fetching the ModelMocks package on the previous directory
sys.path.insert(1,sys.path[0].rsplit('/',1)[0])

from GIMMECore import *
from ModelMocks import *


print("------------------------------------------")
print("-----                                -----")
print("-----     SIMPLE GIMME API TEST      -----")
print("-----                                -----")
print("------------------------------------------")

numPlayers = int(input("How many students would you like? "))
preferredNumberOfPlayersPerGroup = int(input("How many students per group would you prefer? "))
numTasks = int(input("How many tasks would you like? "))

adaptationGIMME = Adaptation() 

players = [0 for x in range(numPlayers)]
tasks = [0 for x in range(numTasks)]

#minNumberPlayersPerGroup = 2
#maxNumberPlayersPerGroup = 5

playerBridge = CustomPlayerModelBridge(players)
taskBridge = CustomTaskModelBridge(tasks)

profileTemplate = InteractionsProfile({"Focus": 0, "Challenge": 0})


print("Setting up the players...")

for x in range(numPlayers):
	gridTrimAlg = QualitySortPlayerDataTrimAlg(maxNumModelElements = 30, qualityWeights = PlayerCharacteristics(ability=0.5, engagement=0.5))
	playerBridge.registerNewPlayer(
		playerId = int(x), 
		name = "Player "+str(x+1), 
		currState = PlayerState(profile = profileTemplate.generate_copy().reset()),
		pastModelIncreasesDataFrame = PlayerStatesDataFrame(
			interactions_profile_template= profileTemplate.generate_copy().reset(),
			trim_alg= gridTrimAlg),
		currModelIncreases = PlayerCharacteristics(), 
		preferencesEst = profileTemplate.generate_copy().reset(),
		realPreferences = profileTemplate.generate_copy().reset())
	playerBridge.resetState(x)
	playerBridge.get_player_states_data_frame(x).trim_alg.considerStateResidue(True)

	# init players including predicted preferences
	playerBridge.reset_player(x)

	playerBridge.set_player_preferences_est(x, profileTemplate.generate_copy().init())
	# realPreferences = realPersonalities[x]
	# playerBridge.setPlayerRealPreferences(x, realPreferences)

	playerBridge.setPlayerRealPreferences(x, profileTemplate.randomized())
	playerBridge.setBaseLearningRate(x, 0.5)

	playerBridge.get_player_states_data_frame(x).trim_alg.considerStateResidue(False)

print("Players created.")
print(json.dumps(playerBridge.players, default=lambda o: o.__dict__, sort_keys=True, indent=2))

print("\nSetting up the tasks...")

for x in range(numTasks):
	diffW = random.uniform(0, 1)
	profW = 1 - diffW
	taskBridge.registerNewTask(
		taskId = int(x), 
		description = "Task "+str(x+1), 
		minRequiredAbility = random.uniform(0, 1), 
		profile = profileTemplate.randomized(), 
		minDuration = str(timedelta(minutes=1)), 
		difficultyWeight = diffW, 
		profileWeight = profW)

print("Tasks created:")
print(json.dumps(taskBridge.tasks, default=lambda o: o.__dict__, sort_keys=True, indent=2))

print("\nSetting up the adaptation algorithms...")
def simulateReaction(isBootstrap, playerBridge, playerId):
	currState = playerBridge.get_player_curr_state(playerId)
	newState = calcReaction(
		isBootstrap = isBootstrap, 
		playerBridge = playerBridge, 
		state = currState, 
		playerId = playerId)

	increases = PlayerState(state_type= newState.stateType)
	increases.profile = currState.profile
	increases.characteristics = PlayerCharacteristics(ability=(newState.characteristics.ability - currState.characteristics.ability), engagement=newState.characteristics.engagement)
	playerBridge.set_and_save_player_state_to_data_frame(playerId, increases, newState)
	return increases

def calcReaction(isBootstrap, playerBridge, state, playerId):
	preferences = playerBridge.getPlayerRealPreferences(playerId)
	numDims = len(preferences.dimensions)
	newStateType = 0 if isBootstrap else 1
	newState = PlayerState(
		state_type= newStateType,
		characteristics = PlayerCharacteristics(
			ability=state.characteristics.ability, 
			engagement=state.characteristics.engagement
			), 
		profile=state.profile)
	newState.characteristics.engagement = 1 - (preferences.distance_between(state.profile) / math.sqrt(numDims))  #between 0 and 1
	if newState.characteristics.engagement>1:
		breakpoint()
	abilityIncreaseSim = (newState.characteristics.engagement*playerBridge.getBaseLearningRate(playerId))
	newState.characteristics.ability = newState.characteristics.ability + abilityIncreaseSim
	return newState



numberOfConfigChoices = 100
numTestedPlayerProfilesInEst = 500
regAlg = KNNRegression(playerBridge, 5)

ODPIPConfigsGenAlg = ODPIPConfigsGenAlg(
	playerModelBridge = playerBridge,
	interactionsProfileTemplate = profileTemplate.generate_copy(),
	regAlg = regAlg,
	persEstAlg = ExplorationPreferencesEstAlg(
		playerModelBridge = playerBridge, 
		interactionsProfileTemplate = profileTemplate.generate_copy(),
		regAlg = regAlg,
		numTestedPlayerProfiles = numTestedPlayerProfilesInEst),
	preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup,
	#minNumberOfPlayersPerGroup = minNumberPlayersPerGroup,
	#maxNumberOfPlayersPerGroup = maxNumberPlayersPerGroup
)
adaptationGIMME.init(
	player_model_bridge= playerBridge,
	task_model_bridge= taskBridge,
	configs_gen_alg= ODPIPConfigsGenAlg,
	name="Test Adaptation"
)
print("Adaptation initialized and ready!")
print("~~~~~~(Initialization Complete)~~~~~~\n\n\n")



ready = True
thinking = False

#here is a loading animation 
#(source: https://stackoverflow.com/questions/22029562/python-how-to-make-simple-animated-loading-while-process-is-running)
def animate():
	for c in itertools.cycle(['()       ','(.....)  ', '(..)(...)  ', '(...)(..)  ']):
		if not ready:
			break
		if not thinking:
			continue
		sys.stdout.write('\rcomputing new iteration' + c)
		sys.stdout.flush()
		time.sleep(0.3)

t = threading.Thread(target=animate)
t.start()

while(True):
	readyText = ""
	readyText = str(input("Ready to compute iteration? (y/n) "))
	while(readyText!="y" and readyText!="n"):
		readyText = str(input("Please answer y/n: "))
		continue
	ready = (readyText=="y")
	if(not ready):
		print("~~~~~~(The End)~~~~~~")
		break


	print("----------------------")
	thinking = True
	result = ""
	try:
		result = json.dumps(adaptationGIMME.iterate(), default=lambda o: o.__dict__, sort_keys=True)
	except Exception as e:
		print("An exception occurred. Possibly an impossible class configuration was input...")
		print("Exception: "+str(e))
		thinking = False
		ready=False
		print("~~~~~~(The End)~~~~~~")
		break
		
	print("\rIteration Summary:\n\n\n"+result)
	thinking = False
	print("----------------------\n\n\n")
	print("Player States:\n\n\n")
	for x in range(numPlayers):
		increases = simulateReaction(False, playerBridge, x)
		print(json.dumps(playerBridge.get_player_curr_state(x), default=lambda o: o.__dict__, sort_keys=True))


	print("~~~~~~~~~~~~~~~~~~~~~\n\n\n")
