from datetime import datetime, timedelta

import itertools
import threading
import sys

sys.path.insert(1, sys.path[0].rsplit('/', 2)[0])

# hack for fetching the ModelMocks package on the previous directory
from pathlib import Path
sys.path.insert(1, str(Path(sys.path[0]).parent))

from GIMMECore import *
from ModelMocks import *

print("------------------------------------------")
print("-----                                -----")
print("-----     SIMPLE GIMME API TEST      -----")
print("-----                                -----")
print("------------------------------------------")

num_players = int(input("How many students would you like? "))
preferred_num_group_players = int(input("How many students per group would you prefer? "))
num_tasks = int(input("How many tasks would you like? "))

adaptation_gimme = Adaptation()

# ----------------------- [Init Model Bridges] --------------------------------
print("Initializing model bridges...")

player_bridge = CustomPlayerModelBridge()
task_bridge = CustomTaskModelBridge()

prof_template = InteractionsProfile({"Focus": 0, "Challenge": 0})

print("Setting up the players...")

for x in range(num_players):
    gridTrimAlg = QualitySortPlayerDataTrimAlg(max_num_model_elements=30,
                                               quality_weights=PlayerCharacteristics(ability=0.5, engagement=0.5))
    player_bridge.register_new_player(
        player_id=int(x),
        name="Player " + str(x + 1),
        curr_state=PlayerState(profile=prof_template.generate_copy().reset()),
        past_model_increases_data_frame=PlayerStatesDataFrame(
            interactions_profile_template=prof_template.generate_copy().reset(),
            trim_alg=gridTrimAlg),
        curr_model_increases=PlayerCharacteristics(),
        preferences_est=prof_template.generate_copy().reset(),
        real_preferences=prof_template.generate_copy().reset())
    player_bridge.reset_state(x)
    player_bridge.get_player_states_data_frame(x).trim_alg.consider_state_residue(True)

    # init players including predicted preferences
    player_bridge.reset_player(x)

    player_bridge.set_player_preferences_est(x, prof_template.generate_copy().init())
    player_bridge.set_player_real_preferences(x, prof_template.randomized())
    player_bridge.set_base_learning_rate(x, 0.5)

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


def simulate_reaction(is_bootstrap, player_id):
    curr_state = player_bridge.get_player_curr_state(player_id)
    new_state = calc_reaction(
        is_bootstrap=is_bootstrap,
        state=curr_state,
        player_id=player_id)

    increases = PlayerState(state_type=new_state.state_type)
    increases.profile = curr_state.profile
    increases.characteristics = PlayerCharacteristics(
        ability=(new_state.characteristics.ability - curr_state.characteristics.ability),
        engagement=new_state.characteristics.engagement)
    player_bridge.set_and_save_player_state_to_data_frame(player_id, increases, new_state)
    return increases


def calc_reaction(is_bootstrap, state, player_id):
    preferences = player_bridge.get_player_real_preferences(player_id)
    num_dims = len(preferences.dimensions)
    new_state_type = 0 if is_bootstrap else 1
    new_state = PlayerState(
        state_type=new_state_type,
        characteristics=PlayerCharacteristics(
            ability=state.characteristics.ability,
            engagement=state.characteristics.engagement
        ),
        profile=state.profile)
    new_state.characteristics.engagement = 1 - (
            preferences.distance_between(state.profile) / math.sqrt(num_dims))  #between 0 and 1
    if new_state.characteristics.engagement > 1:
        breakpoint()
    ability_inc_sim = (new_state.characteristics.engagement * player_bridge.get_base_learning_rate(player_id))
    new_state.characteristics.ability = new_state.characteristics.ability + ability_inc_sim
    return new_state


numberOfConfigChoices = 100
numTestedPlayerProfilesInEst = 500
quality_eval_alg = KNNRegQualityEvalAlg(player_bridge, 5)

ODPIPConfigsGenAlg = ODPIPConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=prof_template.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=player_bridge,
        interactions_profile_template=prof_template.generate_copy(),
        quality_eval_alg=quality_eval_alg,
        num_tested_player_profiles=numTestedPlayerProfilesInEst),
    preferred_number_of_players_per_group=preferred_num_group_players
)
adaptation_gimme.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=ODPIPConfigsGenAlg,
    name="Test Adaptation"
)
print("Adaptation initialized and ready!")
print("~~~~~~(Initialization Complete)~~~~~~\n\n\n")

ready = True
thinking = False


# here is a loading animation
# (source: https://stackoverflow.com/questions/
# 22029562/python-how-to-make-simple-animated-loading-while-process-is-running)
def animate():
    for c in itertools.cycle(['()       ', '(.....)  ', '(..)(...)  ', '(...)(..)  ']):
        if not ready:
            break
        if not thinking:
            continue
        sys.stdout.write('\rcomputing new iteration' + c)
        sys.stdout.flush()
        time.sleep(0.3)


t = threading.Thread(target=animate)
t.start()

while True:
    ready_text = str(input("Ready to compute iteration? (y/n) "))
    while ready_text != "y" and ready_text != "n":
        ready_text = str(input("Please answer y/n: "))
        continue
    ready = (ready_text == "y")
    if not ready:
        print("~~~~~~(The End)~~~~~~")
        break

    print("----------------------")
    thinking = True
    result = ""
    try:
        result = json.dumps(adaptation_gimme.iterate(), default=lambda o: o.__dict__, sort_keys=True)
    except Exception as e:
        print("An exception occurred. Possibly an impossible class configuration was input...")
        print("Exception: " + str(e))
        thinking = False
        ready = False
        print("~~~~~~(The End)~~~~~~")
        break

    print("\rIteration Summary:                       \n\n\n" + result)
    thinking = False
    print("----------------------\n\n\n")
    print("Player States:\n\n\n")
    for x in range(num_players):
        increases = simulate_reaction(False, x)
        print(json.dumps(player_bridge.get_player_curr_state(x), default=lambda o: o.__dict__, sort_keys=True))

    print("~~~~~~~~~~~~~~~~~~~~~\n\n\n")
