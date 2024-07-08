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

sys.path.insert(1, sys.path[0].rsplit('/', 2)[0])

#hack for fetching the ModelMocks package on the previous directory
from pathlib import Path

sys.path.insert(1, str(Path(sys.path[0]).parent))

from GIMMECore import *
from ModelMocks import *
from LogManager import *

numRuns = 5
maxNumTrainingIterations = 10
numRealIterations = 10
preferredNumberOfPlayersPerGroup = 3

playerWindow = 10
numPlayers = 15
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
newpath = sys.path[0] + "/analyzer/results/"
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
logManager = CSVLogManager(newpath, simsID)
# logManager = SilentLogManager()


# ----------------------- [Init Algorithms] --------------------------------
print("Initing algorithms...")

syntergyTablePath = sys.path[0] + "/synergyTable.txt"

qualityEvalAlg = KNNRegQualityEvalAlg(player_model_bridge=playerBridge, k=5)
tabularQualityEvalAlg = SynergiesTabQualityEvalAlg(player_model_bridge=playerBridge,
                                                   task_model_bridge=taskBridge,
                                                   syntergy_table_path=syntergyTablePath)

# - - - - - 
intProfTemplate2D = InteractionsProfile({"dim_0": 0, "dim_1": 0})

evolutionaryConfigsAlg = EvolutionaryConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate2D.generate_copy(),
    quality_eval_alg=qualityEvalAlg,
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup,
    initial_population_size=initialPopulationSize,
    num_evolutions_per_iteration=numberOfEvolutionsPerIteration,

    prob_cross=probOfCross,
    prob_mut=probOfMutation,

    prob_mut_config=probOfMutationConfig,
    prob_mut_profiles=probOfMutationGIPs,

    num_children_per_iteration=numChildrenPerIteration,
    num_survivors=numSurvivors,

    cx_op="order",
    # jointPlayerConstraints="[15,1];[3,4]",
    # separatedPlayerConstraints="[0,1]"
)
adaptationGA.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=evolutionaryConfigsAlg,
    name="GIMME_GA"
)

ODPIPconfigsAlg = ODPIPConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate2D.generate_copy(),
    quality_eval_alg=qualityEvalAlg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=playerBridge,
        interactions_profile_template=intProfTemplate2D.generate_copy(),
        quality_eval_alg=qualityEvalAlg,
        num_tested_player_profiles=numTestedPlayerProfilesInEst),
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup
)
adaptationODPIP.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=ODPIPconfigsAlg,
    name="GIMME_ODPIP"
)

tabularODPIPconfigsAlg = ODPIPConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate2D.generate_copy(),
    quality_eval_alg=tabularQualityEvalAlg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=playerBridge,
        interactions_profile_template=intProfTemplate2D.generate_copy(),
        quality_eval_alg=qualityEvalAlg,
        num_tested_player_profiles=numTestedPlayerProfilesInEst),
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup
    # preferredNumberOfPlayersPerGroup = preferredNumberOfPlayersPerGroup,
    #jointPlayerConstraints="[15,1];[2,7];[3,4];[12,13];[14,15];[0,12];[9,15]",
    # separatedPlayerConstraints="[0,1]"
)
adaptationTabularODPIP.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=tabularODPIPconfigsAlg,
    name="GIMME_ODPIP"
)

CLinkconfigsAlg = CLinkConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate2D.generate_copy(),
    quality_eval_alg=qualityEvalAlg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=playerBridge,
        interactions_profile_template=intProfTemplate2D.generate_copy(),
        quality_eval_alg=qualityEvalAlg,
        num_tested_player_profiles=numTestedPlayerProfilesInEst),
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup
)
adaptationCLink.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=CLinkconfigsAlg,
    name="GIMME_CLink"
)

tabularCLinkconfigsAlg = CLinkConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate2D.generate_copy(),
    quality_eval_alg=tabularQualityEvalAlg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=playerBridge,
        interactions_profile_template=intProfTemplate2D.generate_copy(),
        quality_eval_alg=qualityEvalAlg,
        num_tested_player_profiles=numTestedPlayerProfilesInEst),
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup
)
adaptationTabularCLink.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=tabularCLinkconfigsAlg,
    name="GIMME_CLink_Tabular"
)

prsConfigsAlg = PureRandomSearchConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate2D.generate_copy(),
    quality_eval_alg=qualityEvalAlg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=playerBridge,
        interactions_profile_template=intProfTemplate2D.generate_copy(),
        quality_eval_alg=qualityEvalAlg,
        num_tested_player_profiles=numTestedPlayerProfilesInEst),
    number_of_config_choices=numberOfConfigChoices,
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup,
    # jointPlayerConstraints="[15,1]",
    # separatedPlayerConstraints="[0,1]"
)
adaptationPRS.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=prsConfigsAlg,
    name="GIMME_PRS"
)

randomConfigsAlg = RandomConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate2D.generate_copy(),
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup,
    # jointPlayerConstraints="[15,1]",
    # separatedPlayerConstraints="[0,1]"
)
adaptationRandom.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=randomConfigsAlg,
    name="Random"
)

# - - - - -
intProfTemplate1D = InteractionsProfile({"dim_0": 0})
evolutionaryConfigsAlg1D = EvolutionaryConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate1D.generate_copy(),
    quality_eval_alg=qualityEvalAlg,
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup,
    initial_population_size=initialPopulationSize,
    num_evolutions_per_iteration=numberOfEvolutionsPerIteration,

    prob_cross=probOfCross,
    prob_mut=probOfMutation,

    prob_mut_config=probOfMutationConfig,
    prob_mut_profiles=probOfMutationGIPs,

    num_children_per_iteration=numChildrenPerIteration,
    num_survivors=numSurvivors
)
adaptationEvl1D.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=evolutionaryConfigsAlg1D,
    name="GIMME_GA1D"
)

# GIMME is the same as GIMME 2D
intProfTemplate3D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0})
evolutionaryConfigsAlg3D = EvolutionaryConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate3D.generate_copy(),
    quality_eval_alg=qualityEvalAlg,
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup,
    initial_population_size=initialPopulationSize,
    num_evolutions_per_iteration=numberOfEvolutionsPerIteration,

    prob_cross=probOfCross,
    prob_mut=probOfMutation,

    prob_mut_config=probOfMutationConfig,
    prob_mut_profiles=probOfMutationGIPs,

    num_children_per_iteration=numChildrenPerIteration,
    num_survivors=numSurvivors
)
adaptationEvl3D.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=evolutionaryConfigsAlg3D,
    name="GIMME_GA3D"
)

intProfTemplate4D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0})
evolutionaryConfigsAlg4D = EvolutionaryConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate4D.generate_copy(),
    quality_eval_alg=qualityEvalAlg,
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup,
    initial_population_size=initialPopulationSize,
    num_evolutions_per_iteration=numberOfEvolutionsPerIteration,

    prob_cross=probOfCross,
    prob_mut=probOfMutation,

    prob_mut_config=probOfMutationConfig,
    prob_mut_profiles=probOfMutationGIPs,

    num_children_per_iteration=numChildrenPerIteration,
    num_survivors=numSurvivors
)
adaptationEvl4D.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=evolutionaryConfigsAlg4D,
    name="GIMME_GA4D"
)

intProfTemplate5D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0, "dim_4": 0})
evolutionaryConfigsAlg5D = EvolutionaryConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate5D.generate_copy(),
    quality_eval_alg=qualityEvalAlg,
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup,
    initial_population_size=initialPopulationSize,
    num_evolutions_per_iteration=numberOfEvolutionsPerIteration,

    prob_cross=probOfCross,
    prob_mut=probOfMutation,

    prob_mut_config=probOfMutationConfig,
    prob_mut_profiles=probOfMutationGIPs,

    num_children_per_iteration=numChildrenPerIteration,
    num_survivors=numSurvivors
)
adaptationEvl5D.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=evolutionaryConfigsAlg5D,
    name="GIMME_GA5D"
)

intProfTemplate6D = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0, "dim_4": 0, "dim_5": 0})
evolutionaryConfigsAlg6D = EvolutionaryConfigsGenAlg(
    player_model_bridge=playerBridge,
    interactions_profile_template=intProfTemplate6D.generate_copy(),
    quality_eval_alg=qualityEvalAlg,
    preferred_number_of_players_per_group=preferredNumberOfPlayersPerGroup,
    initial_population_size=initialPopulationSize,
    num_evolutions_per_iteration=numberOfEvolutionsPerIteration,

    prob_cross=probOfCross,
    prob_mut=probOfMutation,

    prob_mut_config=probOfMutationConfig,
    prob_mut_profiles=probOfMutationGIPs,

    num_children_per_iteration=numChildrenPerIteration,
    num_survivors=numSurvivors
)
adaptationEvl6D.init(
    player_model_bridge=playerBridge,
    task_model_bridge=taskBridge,
    configs_gen_alg=evolutionaryConfigsAlg6D,
    name="GIMME_GA6D"
)


# ----------------------- [Simulation Methods] --------------------------------

def simulateReaction(playerBridge, currIteration, playerId):
    currState = playerBridge.get_player_curr_state(playerId)
    newState = calcReaction(
        playerBridge=playerBridge,
        state=currState,
        playerId=playerId,
        currIteration=currIteration)

    increases = PlayerState(state_type=newState.stateType)
    increases.profile = currState.profile
    increases.characteristics = PlayerCharacteristics(
        ability=(newState.characteristics.ability - currState.characteristics.ability),
        engagement=newState.characteristics.engagement)
    playerBridge.set_and_save_player_state_to_data_frame(playerId, increases, newState)
    return increases


def calcReaction(playerBridge, state, playerId, currIteration):
    preferences = playerBridge.get_player_real_preferences(playerId)
    numDims = len(preferences.dimensions)
    newState = PlayerState(
        state_type=1,
        characteristics=PlayerCharacteristics(
            ability=state.characteristics.ability,
            engagement=state.characteristics.engagement
        ),
        profile=state.profile)
    newState.characteristics.engagement = 1 - (
                preferences.distance_between(state.profile) / math.sqrt(numDims))  #between 0 and 1
    if newState.characteristics.engagement > 1:
        breakpoint()
    abilityIncreaseSim = (newState.characteristics.engagement * playerBridge.getBaseLearningRate(playerId))
    newState.characteristics.ability = newState.characteristics.ability + abilityIncreaseSim
    return newState


def executionPhase(numRuns, playerBridge, maxNumIterations, startingI, currRun, adaptation):
    if (maxNumIterations <= 0):
        return

    numPlayersToTest = len(playerBridge.get_all_player_ids())

    i = startingI
    while (i < maxNumIterations + startingI):

        if adaptation.name == "accurate":
            adaptation.configs_gen_alg.updateCurrIteration(i)

        print("Process [" + simsID + "] performing step (" + str(i - startingI) + " of " + str(
            maxNumIterations) + ") of run (" + str(currRun + 1) + " of " + str(numRuns) + ") of algorithm \"" + str(
            adaptation.name) + "\"...                                                             ", end="\r")

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
                                      "profDiff": str(playerBridge.get_player_real_preferences(x).distance_between(
                                          playerBridge.get_player_curr_profile(x))),
                                      "iterationElapsedTime": str(deltaTime)
                                  })
        i += 1


def executeSimulations(numRuns, profileTemplate, maxNumTrainingIterations, firstTrainingI, numRealIterations,
                       firstRealI, \
                       playerBridge, taskBridge, adaptation, estimatorsAccuracy=None,
                       considerExtremePreferencesValues=None):
    estimatorsAccuracy = 0.1 if estimatorsAccuracy == None else estimatorsAccuracy
    considerExtremePreferencesValues = False if considerExtremePreferencesValues == None else considerExtremePreferencesValues

    adaptationName = adaptation.name

    numInteractionDimensions = len(profileTemplate.dimensions.keys())

    numPlayersToTest = len(playerBridge.get_all_player_ids())

    # re-init stuff
    players = [0 for x in range(numPlayers)]
    tasks = [0 for x in range(numTasks)]

    # create players and tasks
    for x in range(numPlayersToTest):
        playerBridge.register_new_player(
            player_id=str(x),
            name="name",
            curr_state=PlayerState(profile=profileTemplate.generate_copy().reset()),
            past_model_increases_data_frame=PlayerStatesDataFrame(
                interactions_profile_template=profileTemplate.generate_copy().reset(),
                trim_alg=ProximitySortPlayerDataTrimAlg(
                    max_num_model_elements=playerWindow,
                    epsilon=0.005
                )
            ),
            curr_model_increases=PlayerCharacteristics(),
            preferences_est=profileTemplate.generate_copy().reset(),
            real_preferences=profileTemplate.generate_copy().reset())

    for x in range(numTasks):
        taskBridge.registerNewTask(
            taskId=int(x),
            description="description",
            minRequiredAbility=random.uniform(0, 1),
            profile=profileTemplate.generate_copy(),
            minDuration=datetime.timedelta(minutes=1),
            difficultyWeight=0.5,
            profileWeight=0.5)

    for r in range(numRuns):
        allRealPreferences = []
        allQuestionnairePreferences = []

        EPdimensions = [{"dim_0": 0, "dim_1": 0}, {"dim_0": 1, "dim_1": 0}, {"dim_0": 0, "dim_1": 1},
                        {"dim_0": 1, "dim_1": 1}, {"dim_0": 1, "dim_1": 1}]

        playersDimsStr = "players: [\n"

        for x in range(numPlayersToTest):
            profile = profileTemplate.generate_copy().reset()
            if (considerExtremePreferencesValues):
                profile.dimensions = EPdimensions[x % len(EPdimensions)]
                playersDimsStr += "{" + str(profile.dimensions) + "},\n"
            else:
                for d in range(numInteractionDimensions):
                    profile.dimensions["dim_" + str(d)] = random.uniform(0, 1)

            allRealPreferences.append(profile)
            # allRealPreferences[x].normalize()

            profile = profileTemplate.generate_copy().reset()
            currRealPreferences = allRealPreferences[x]
            for d in range(numInteractionDimensions):
                profile.dimensions["dim_" + str(d)] = numpy.clip(
                    random.gauss(currRealPreferences.dimensions["dim_" + str(d)], estimatorsAccuracy), 0, 1)
            allQuestionnairePreferences.append(profile)
            # allQuestionnairePreferences[x].normalize()

            # init players including predicted preferences
            playerBridge.reset_player(x)

            playerBridge.set_player_preferences_est(x, profileTemplate.generate_copy().init())
            # realPreferences = allRealPreferences[x]
            # playerBridge.setPlayerRealPreferences(x, realPreferences)

            questionnairePreferences = allQuestionnairePreferences[x]
            playerBridge.set_player_real_preferences(x, questionnairePreferences)
            playerBridge.setBaseLearningRate(x, 0.5)

            playerBridge.get_player_states_data_frame(x).trim_alg.consider_state_residue(False)

        playersDimsStr += "],\n"

        # print(playersDimsStr)

        if (maxNumTrainingIterations > 0):
            adaptation.bootstrap(maxNumTrainingIterations)

        # change for "real" preferences from which the predictions supposidely are based on...
        for x in range(numPlayersToTest):
            playerBridge.reset_state(x)

            realPreferences = allRealPreferences[x]
            playerBridge.set_player_real_preferences(x, realPreferences)
            playerBridge.setBaseLearningRate(x, random.gauss(0.5, 0.05))

            playerBridge.get_player_states_data_frame(x).trim_alg.consider_state_residue(True)

        if r > 0:
            adaptation.configs_gen_alg.reset()

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
    print("All results are saved to \'" + newpath + "\'.")
    print("------------------------------------------")

    # ----------------------- [Execute Algorithms] ----------------------------

    # inputtedText = input("<<< All ready! Press Enter to start (Q, then Enter exits the application). >>>")
    # if (inputtedText== "Q"):
    # exit()

    # Explore Base GIMME
    adaptationPRS.name = "Random"
    executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationRandom)

    adaptationPRS.name = "GIMME_PRS"
    executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationPRS)

    adaptationGA.name = "GIMME_GA"
    executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationGA)

    adaptationODPIP.name = "GIMME_ODPIP"
    executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationODPIP)

    adaptationTabularODPIP.name = "GIMME_ODPIP_Tabular"
    executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationTabularODPIP)

    # 	adaptationCLink.name = "GIMME_CLink"
    # 	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
    # 	   	playerBridge, taskBridge, adaptationCLink)
    #
    # 	adaptationTabularCLink.name = "GIMME_CLink_Tabular"
    # 	executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
    # 		playerBridge, taskBridge, adaptationTabularCLink)

    # Explore GIMME-Bootstrap
    adaptationODPIP.name = "GIMME_ODPIP_Bootstrap"
    executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations,
                       maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationODPIP, estimatorsAccuracy=0.1)

    adaptationODPIP.name = "GIMME_ODPIP_Bootstrap_HighAcc"
    executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations,
                       maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationODPIP, estimatorsAccuracy=0.05)

    adaptationODPIP.name = "GIMME_ODPIP_Bootstrap_LowAcc"
    executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations,
                       maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationODPIP, estimatorsAccuracy=0.2)

    # adaptationCLink.name = "GIMME_CLink_Bootstrap"
    # executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
    # 				playerBridge, taskBridge, adaptationCLink, estimatorsAccuracy = 0.1)
    #
    # adaptationCLink.name = "GIMME_CLink_Bootstrap_HighAcc"
    # executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
    # 				playerBridge, taskBridge, adaptationCLink, estimatorsAccuracy = 0.05)
    #
    # adaptationCLink.name = "GIMME_CLink_Bootstrap_LowAcc"
    # executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations, maxNumTrainingIterations,
    # 				playerBridge, taskBridge, adaptationCLink, estimatorsAccuracy = 0.2)

    adaptationGA.name = "GIMME_GA_Bootstrap"
    executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations,
                       maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationGA, estimatorsAccuracy=0.1)

    adaptationGA.name = "GIMME_GA_Bootstrap_HighAcc"
    executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations,
                       maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationGA, estimatorsAccuracy=0.05)

    adaptationGA.name = "GIMME_GA_Bootstrap_LowAcc"
    executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations,
                       maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationGA, estimatorsAccuracy=0.2)

    adaptationPRS.name = "GIMME_PRS_Bootstrap"
    executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations,
                       maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationPRS, estimatorsAccuracy=0.1)

    adaptationPRS.name = "GIMME_PRS_Bootstrap_HighAcc"
    executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations,
                       maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationPRS, estimatorsAccuracy=0.05)

    adaptationPRS.name = "GIMME_PRS_Bootstrap_LowAcc"
    executeSimulations(numRuns, intProfTemplate2D, maxNumTrainingIterations, 0, numRealIterations,
                       maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationPRS, estimatorsAccuracy=0.2)

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
    # adaptationCLink.name = "GIMME_CLink_EP"
    # executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
    # 	playerBridge, taskBridge, adaptationCLink, considerExtremePreferencesValues = True)

    adaptationODPIP.name = "GIMME_ODPIP_EP"
    executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationODPIP, considerExtremePreferencesValues=True)

    adaptationTabularODPIP.name = "GIMME_Tabular_ODPIP_EP"
    executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationTabularODPIP, considerExtremePreferencesValues=True)

    adaptationGA.name = "GIMME_GA_EP"
    executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationGA, considerExtremePreferencesValues=True)

    adaptationPRS.name = "GIMME_PRS_EP"
    executeSimulations(numRuns, intProfTemplate2D, 0, 0, numRealIterations, maxNumTrainingIterations,
                       playerBridge, taskBridge, adaptationPRS, considerExtremePreferencesValues=True)

    print("Done!                        ")
