from datetime import timedelta
import sys
import json

sys.path.insert(1, sys.path[0].rsplit('/', 2)[0])

from GIMMECore import *

# hack for fetching the ModelMocks package on the previous directory 
from pathlib import Path

sys.path.insert(1, str(Path(sys.path[0]).parent))
from ModelMocks import *

print("------------------------------------------")
print("-----                                -----")
print("-----     SIMPLE GIMME API TEST      -----")
print("-----                                -----")
print("------------------------------------------")

num_players = 15
preferred_num_group_players = 4
num_tasks = 1

# ----------------------- [Init Model Bridges] --------------------------------
print("Initializing model bridges...")

player_bridge = CustomPlayerModelBridge()
task_bridge = CustomTaskModelBridge()

prof_template = InteractionsProfile({"Focus": 0, "Challenge": 0})

print("Setting up the players...")

for x in range(num_players):
    gridTrimAlg = QualitySortPlayerDataTrimAlg(
        max_num_model_elements=30,
        quality_weights=PlayerCharacteristics(ability=0.5, engagement=0.5))

    player_bridge.register_new_player(
        player_id=int(x),
        name="Player " + str(x + 1),
        curr_state=PlayerState(profile=prof_template.generate_copy().reset()),
        past_model_increases_data_frame=PlayerStatesDataFrame(
            interactions_profile_template=prof_template.generate_copy().reset(),
            trim_alg=gridTrimAlg),
        preferences_est=prof_template.generate_copy().reset(),
        real_preferences=prof_template.randomized(),
        base_learning_rate=0.5)
    player_bridge.get_player_states_data_frame(x).trim_alg.consider_state_residue(False)

print("Players created.")
print(json.dumps(player_bridge.players, default=lambda o: o.__dict__, sort_keys=True, indent=2))

print("\nSetting up the tasks...")

for x in range(num_tasks):
    diff_w = random.uniform(0, 1)
    prof_w = 1 - diff_w
    task_bridge.register_new_task(
        task_id=int(x),
        description="Task " + str(x + 1),
        min_required_ability=random.uniform(0, 1),
        profile=prof_template.randomized(),
        min_duration=str(timedelta(minutes=1)),
        difficulty_weight=diff_w,
        profile_weight=prof_w)

print("Tasks created:")
print(json.dumps(task_bridge.tasks, default=lambda o: o.__dict__, sort_keys=True, indent=2))

print("\nSetting up the adaptation algorithms...")

num_tested_profs_in_est = 500
quality_eval_alg = KNNRegQualityEvalAlg(player_bridge, 5)

configs_gen_alg = ODPIPConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=prof_template.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=player_bridge,
        interactions_profile_template=prof_template.generate_copy(),
        quality_eval_alg=quality_eval_alg,
        num_tested_player_profiles=num_tested_profs_in_est),
    preferred_num_players_per_group=preferred_num_group_players
)
course_adapt = Adaptation(name="Test Adaptation",
                          player_model_bridge=player_bridge,
                          task_model_bridge=task_bridge,
                          configs_gen_alg=configs_gen_alg)

course_adapt.bootstrap(5)
print("\nreturn: "+ str(
    round(quality_eval_alg.evaluate(
        InteractionsProfile({"Focus": 0.3, "Challenge": 0.4}), 
        [1, 2, 5, 8])
        , 2)))
