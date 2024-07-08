import sys
import datetime

sys.path.insert(1, sys.path[0].rsplit('/', 2)[0])

# hack for fetching the ModelMocks package on the previous directory
from pathlib import Path
sys.path.insert(1, str(Path(sys.path[0]).parent))

from GIMMECore import *
from ModelMocks import *
from LogManager import *

num_runs = 5
max_num_training_iterations = 10
num_real_iterations = 10
preferred_num_players_per_group = 3

player_window = 10
num_players = 15
num_tasks = 1

# ----------------------- [Init PRS] --------------------------------
num_config_choices = 100
num_tested_profiles_in_est = 500

# ----------------------- [Init GA] --------------------------------
initial_population_size = 100
num_evolutions_per_iteration = 50

prob_cross = 0.65
prob_mut = 0.15
# prob_reproduction = 1 - (prob_cross + prob_mut) = 0.15

prob_mut_config = 0.8
prob_mut_profiles = 0.4

num_survivors = 10
num_children_per_iteration = 100

sims_id = str(os.getpid())

start_time = str(datetime.datetime.now())
new_path = sys.path[0] + "/analyzer/results/"
if not os.path.exists(new_path):
    os.makedirs(new_path)


# ----------------------- [Init Model Bridges] --------------------------------
print("Initializing model bridges...")

player_bridge = CustomPlayerModelBridge()
task_bridge = CustomTaskModelBridge()


# create players and tasks
for x in range(num_players):
    player_bridge.register_new_player(
        player_id=str(x),
        name=None,
        curr_state=None,
        curr_model_increases=None,
        preferences_est=None,
        real_preferences=None,
        past_model_increases_data_frame=None)

for x in range(num_players):
    task_bridge.register_new_task(
        task_id=int(x),
        description=None,
        min_required_ability=None,
        profile=None,
        min_duration=None,
        difficulty_weight=None,
        profile_weight=None)

# ----------------------- [Init Adaptations] --------------------------------
adaptation_prs = Adaptation()
list_adaptation_prs = []
adaptation_evl_scx = Adaptation()
adaptation_evl = Adaptation()
list_adaptation_ga = []
adaptation_odpip = Adaptation()
list_adaptation_odpip = []
adaptation_tab_odpip = Adaptation()
list_adaptation_tabular_odpip = []
adaptation_clink = Adaptation()
list_adaptation_clink = []
adaptation_tab_clink = Adaptation()
list_adaptation_tabular_clink = []

adaptation_evl_1d = Adaptation()
adaptation_evl_3d = Adaptation()
adaptation_evl_4d = Adaptation()
adaptation_evl_5d = Adaptation()
adaptation_evl_6d = Adaptation()

adaptation_random = Adaptation()
list_adaptation_random = []

all_real_preferences = []
all_questionnaire_preferences = []

# ----------------------- [Init Log Manager] --------------------------------
print("Initing .csv log manager...")
log_manager = CSVLogManager(new_path, sims_id)
# log_manager = SilentLogManager()


# ----------------------- [Init Algorithms] --------------------------------
print("Initing algorithms...")

synergy_table_path = sys.path[0] + "/synergyTable.txt"

quality_eval_alg = KNNRegQualityEvalAlg(player_model_bridge=player_bridge, k=5)
tab_quality_eval_alg = SynergiesTabQualityEvalAlg(player_model_bridge=player_bridge,
                                                  synergy_table_path=synergy_table_path)

# - - - - - 
int_prof_2d = InteractionsProfile({"dim_0": 0, "dim_1": 0})

evl_configs_alg = EvolutionaryConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_2d.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    preferred_number_of_players_per_group=preferred_num_players_per_group,
    initial_population_size=initial_population_size,
    num_evolutions_per_iteration=num_evolutions_per_iteration,

    prob_cross=prob_cross,
    prob_mut=prob_mut,

    prob_mut_config=prob_mut_config,
    prob_mut_profiles=prob_mut_profiles,

    num_children_per_iteration=num_children_per_iteration,
    num_survivors=num_survivors,

    cx_op="order",
    # jointPlayerConstraints="[15,1];[3,4]",
    # separatedPlayerConstraints="[0,1]"
)
adaptation_evl.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=evl_configs_alg,
    name="GIMME_GA"
)

odpip_configs_alg = ODPIPConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_2d.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=player_bridge,
        interactions_profile_template=int_prof_2d.generate_copy(),
        quality_eval_alg=quality_eval_alg,
        num_tested_player_profiles=num_tested_profiles_in_est),
    preferred_number_of_players_per_group=preferred_num_players_per_group
)
adaptation_odpip.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=odpip_configs_alg,
    name="GIMME_ODPIP"
)

tab_odpip_configs_alg = ODPIPConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_2d.generate_copy(),
    quality_eval_alg=tab_quality_eval_alg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=player_bridge,
        interactions_profile_template=int_prof_2d.generate_copy(),
        quality_eval_alg=quality_eval_alg,
        num_tested_player_profiles=num_tested_profiles_in_est),
    preferred_number_of_players_per_group=preferred_num_players_per_group
    #jointPlayerConstraints="[15,1];[2,7];[3,4];[12,13];[14,15];[0,12];[9,15]",
    # separatedPlayerConstraints="[0,1]"
)
adaptation_tab_odpip.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=tab_odpip_configs_alg,
    name="GIMME_ODPIP"
)

clink_configs_alg = CLinkConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_2d.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=player_bridge,
        interactions_profile_template=int_prof_2d.generate_copy(),
        quality_eval_alg=quality_eval_alg,
        num_tested_player_profiles=num_tested_profiles_in_est),
    preferred_number_of_players_per_group=preferred_num_players_per_group
)
adaptation_clink.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=clink_configs_alg,
    name="GIMME_CLink"
)

tab_clink_configs_alg = CLinkConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_2d.generate_copy(),
    quality_eval_alg=tab_quality_eval_alg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=player_bridge,
        interactions_profile_template=int_prof_2d.generate_copy(),
        quality_eval_alg=quality_eval_alg,
        num_tested_player_profiles=num_tested_profiles_in_est),
    preferred_number_of_players_per_group=preferred_num_players_per_group
)
adaptation_tab_clink.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=tab_clink_configs_alg,
    name="GIMME_CLink_Tabular"
)

prs_configs_alg = PureRandomSearchConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_2d.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    pers_est_alg=ExplorationPreferencesEstAlg(
        player_model_bridge=player_bridge,
        interactions_profile_template=int_prof_2d.generate_copy(),
        quality_eval_alg=quality_eval_alg,
        num_tested_player_profiles=num_tested_profiles_in_est),
    number_of_config_choices=num_config_choices,
    preferred_number_of_players_per_group=preferred_num_players_per_group,
    # jointPlayerConstraints="[15,1]",
    # separatedPlayerConstraints="[0,1]"
)
adaptation_prs.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=prs_configs_alg,
    name="GIMME_PRS"
)

random_configs_alg = RandomConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_2d.generate_copy(),
    preferred_number_of_players_per_group=preferred_num_players_per_group,
    # jointPlayerConstraints="[15,1]",
    # separatedPlayerConstraints="[0,1]"
)
adaptation_random.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=random_configs_alg,
    name="Random"
)

# - - - - -
int_prof_1d = InteractionsProfile({"dim_0": 0})
evolutionaryConfigsAlg1D = EvolutionaryConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_1d.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    preferred_number_of_players_per_group=preferred_num_players_per_group,
    initial_population_size=initial_population_size,
    num_evolutions_per_iteration=num_evolutions_per_iteration,

    prob_cross=prob_cross,
    prob_mut=prob_mut,

    prob_mut_config=prob_mut_config,
    prob_mut_profiles=prob_mut_profiles,

    num_children_per_iteration=num_children_per_iteration,
    num_survivors=num_survivors
)
adaptation_evl_1d.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=evolutionaryConfigsAlg1D,
    name="GIMME_GA1D"
)

# GIMME is the same as GIMME 2D
int_prof_3d = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0})
evolutionaryConfigsAlg3D = EvolutionaryConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_3d.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    preferred_number_of_players_per_group=preferred_num_players_per_group,
    initial_population_size=initial_population_size,
    num_evolutions_per_iteration=num_evolutions_per_iteration,

    prob_cross=prob_cross,
    prob_mut=prob_mut,

    prob_mut_config=prob_mut_config,
    prob_mut_profiles=prob_mut_profiles,

    num_children_per_iteration=num_children_per_iteration,
    num_survivors=num_survivors
)
adaptation_evl_3d.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=evolutionaryConfigsAlg3D,
    name="GIMME_GA3D"
)

int_prof_4d = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0})
evolutionaryConfigsAlg4D = EvolutionaryConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_4d.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    preferred_number_of_players_per_group=preferred_num_players_per_group,
    initial_population_size=initial_population_size,
    num_evolutions_per_iteration=num_evolutions_per_iteration,

    prob_cross=prob_cross,
    prob_mut=prob_mut,

    prob_mut_config=prob_mut_config,
    prob_mut_profiles=prob_mut_profiles,

    num_children_per_iteration=num_children_per_iteration,
    num_survivors=num_survivors
)
adaptation_evl_4d.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=evolutionaryConfigsAlg4D,
    name="GIMME_GA4D"
)

int_prof_5d = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0, "dim_4": 0})
evolutionaryConfigsAlg5D = EvolutionaryConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_5d.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    preferred_number_of_players_per_group=preferred_num_players_per_group,
    initial_population_size=initial_population_size,
    num_evolutions_per_iteration=num_evolutions_per_iteration,

    prob_cross=prob_cross,
    prob_mut=prob_mut,

    prob_mut_config=prob_mut_config,
    prob_mut_profiles=prob_mut_profiles,

    num_children_per_iteration=num_children_per_iteration,
    num_survivors=num_survivors
)
adaptation_evl_5d.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=evolutionaryConfigsAlg5D,
    name="GIMME_GA5D"
)

int_prof_6d = InteractionsProfile({"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0, "dim_4": 0, "dim_5": 0})
evolutionaryConfigsAlg6D = EvolutionaryConfigsGenAlg(
    player_model_bridge=player_bridge,
    interactions_profile_template=int_prof_6d.generate_copy(),
    quality_eval_alg=quality_eval_alg,
    preferred_number_of_players_per_group=preferred_num_players_per_group,
    initial_population_size=initial_population_size,
    num_evolutions_per_iteration=num_evolutions_per_iteration,

    prob_cross=prob_cross,
    prob_mut=prob_mut,

    prob_mut_config=prob_mut_config,
    prob_mut_profiles=prob_mut_profiles,

    num_children_per_iteration=num_children_per_iteration,
    num_survivors=num_survivors
)
adaptation_evl_6d.init(
    player_model_bridge=player_bridge,
    task_model_bridge=task_bridge,
    configs_gen_alg=evolutionaryConfigsAlg6D,
    name="GIMME_GA6D"
)


# ----------------------- [Simulation Methods] --------------------------------

def simulate_reaction(player_bridge, player_id):
    curr_state = player_bridge.get_player_curr_state(player_id)
    new_state = calc_reaction(
        player_bridge=player_bridge,
        state=curr_state,
        player_id=player_id)

    increases = PlayerState(state_type=new_state.state_type)
    increases.profile = curr_state.profile
    increases.characteristics = PlayerCharacteristics(
        ability=(new_state.characteristics.ability - curr_state.characteristics.ability),
        engagement=new_state.characteristics.engagement)
    player_bridge.set_and_save_player_state_to_data_frame(player_id, increases, new_state)
    return increases


def calc_reaction(player_bridge, state, player_id):
    preferences = player_bridge.get_player_real_preferences(player_id)
    num_dims = len(preferences.dimensions)
    new_state = PlayerState(
        state_type=1,
        characteristics=PlayerCharacteristics(
            ability=state.characteristics.ability,
            engagement=state.characteristics.engagement
        ),
        profile=state.profile)
    new_state.characteristics.engagement = 1 - (
            preferences.distance_between(state.profile) / math.sqrt(num_dims))  #between 0 and 1
    if new_state.characteristics.engagement > 1:
        breakpoint()
    ability_increase_sim = (new_state.characteristics.engagement * player_bridge.get_base_learning_rate(player_id))
    new_state.characteristics.ability = new_state.characteristics.ability + ability_increase_sim
    return new_state


def execution_phase(num_runs, player_bridge, max_num_iterations, starting_i, curr_run, adaptation):
    if max_num_iterations <= 0:
        return

    num_tested_players = len(player_bridge.get_all_player_ids())

    i = starting_i
    while i < max_num_iterations + starting_i:

        if adaptation.name == "accurate":
            adaptation.configs_gen_alg.updateCurrIteration(i)

        print("Process [" + sims_id + "] performing step (" + str(i - starting_i) + " of " + str(
            max_num_iterations) + ") of run (" + str(curr_run + 1) + " of " + str(num_runs) + ") of algorithm \"" + str(
            adaptation.name) + "\"...                                                             ", end="\r")

        start = time.time()
        adaptation.iterate()
        end = time.time()
        deltaTime = (end - start)

        for x in range(num_tested_players):
            increases = simulate_reaction(player_bridge, x)
            log_manager.writeToLog("", "results",
                                   {
                                       "simsID": str(sims_id),
                                       "algorithm": adaptation.name,
                                       "run": str(curr_run),
                                       "iteration": str(i),
                                       "playerID": str(x),
                                       "abilityInc": str(increases.characteristics.ability),
                                       "engagementInc": str(increases.characteristics.engagement),
                                       "profDiff": str(player_bridge.get_player_real_preferences(x).distance_between(
                                           player_bridge.get_player_curr_profile(x))),
                                       "iterationElapsedTime": str(deltaTime)
                                   })
        i += 1


def execute_simulations(num_runs, prof_template, max_num_training_iterations, num_real_iterations,
                        first_real_i,
                        player_bridge, task_bridge, adaptation, est_error=None,
                        tests_extreme_values=None):
    est_error = 0.1 if est_error is None else est_error
    tests_extreme_values = False if tests_extreme_values is None else tests_extreme_values

    # adaptationName = adaptation.name
    num_int_dims = len(prof_template.dimensions.keys())
    num_tested_players = len(player_bridge.get_all_player_ids())

    # # re-init stuff
    # players = [0 for x in range(num_players)]
    # tasks = [0 for x in range(num_tasks)]

    # create players and tasks
    for x in range(num_tested_players):
        player_bridge.register_new_player(
            player_id=str(x),
            name="name",
            curr_state=PlayerState(profile=prof_template.generate_copy().reset()),
            past_model_increases_data_frame=PlayerStatesDataFrame(
                interactions_profile_template=prof_template.generate_copy().reset(),
                trim_alg=ProximitySortPlayerDataTrimAlg(
                    max_num_model_elements=player_window,
                    epsilon=0.005
                )
            ),
            curr_model_increases=PlayerCharacteristics(),
            preferences_est=prof_template.generate_copy().reset(),
            real_preferences=prof_template.generate_copy().reset())

    for x in range(num_tasks):
        task_bridge.register_new_task(
            task_id=int(x),
            description="description",
            min_required_ability=random.uniform(0, 1),
            profile=prof_template.generate_copy(),
            min_duration=datetime.timedelta(minutes=1),
            difficulty_weight=0.5,
            profile_weight=0.5)

    for r in range(num_runs):
        all_real_prefs = []
        all_questionnaire_prefs = []

        ep_dims = [{"dim_0": 0, "dim_1": 0}, {"dim_0": 1, "dim_1": 0}, {"dim_0": 0, "dim_1": 1},
                   {"dim_0": 1, "dim_1": 1}, {"dim_0": 1, "dim_1": 1}]

        players_dims_str = "players: [\n"

        for x in range(num_tested_players):
            profile = prof_template.generate_copy().reset()
            if tests_extreme_values:
                profile.dimensions = ep_dims[x % len(ep_dims)]
                players_dims_str += "{" + str(profile.dimensions) + "},\n"
            else:
                for d in range(num_int_dims):
                    profile.dimensions["dim_" + str(d)] = random.uniform(0, 1)

            all_real_prefs.append(profile)

            profile = prof_template.generate_copy().reset()
            curr_real_preferences = all_real_prefs[x]
            for d in range(num_int_dims):
                profile.dimensions["dim_" + str(d)] = numpy.clip(
                    random.gauss(curr_real_preferences.dimensions["dim_" + str(d)], est_error), 0, 1)
            all_questionnaire_prefs.append(profile)

            # init players including predicted preferences
            player_bridge.reset_player(x)

            player_bridge.set_player_preferences_est(x, prof_template.generate_copy().init())

            questionnaire_prefs = all_questionnaire_prefs[x]
            player_bridge.set_player_real_preferences(x, questionnaire_prefs)
            player_bridge.set_base_learning_rate(x, 0.5)

            player_bridge.get_player_states_data_frame(x).trim_alg.consider_state_residue(False)

        players_dims_str += "],\n"
        # print(players_dims_str)

        if max_num_training_iterations > 0:
            adaptation.bootstrap(max_num_training_iterations)

        # change for "real" preferences from which the predictions are meant to be based on...
        for x in range(num_tested_players):
            player_bridge.reset_state(x)

            real_pref = all_real_prefs[x]
            player_bridge.set_player_real_preferences(x, real_pref)
            player_bridge.set_base_learning_rate(x, random.gauss(0.5, 0.05))

            player_bridge.get_player_states_data_frame(x).trim_alg.consider_state_residue(True)

        if r > 0:
            adaptation.configs_gen_alg.reset()

        execution_phase(num_runs, player_bridge, num_real_iterations, first_real_i, r, adaptation)


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
    print("All results are saved to \'" + new_path + "\'.")
    print("------------------------------------------")

    # ----------------------- [Execute Algorithms] ----------------------------

    # inputtedText = input("<<< All ready! Press Enter to start (Q, then Enter exits the application). >>>")
    # if (inputtedText== "Q"):
    # exit()

    # - - - - - - - - - - - - - - Explore Base GIMME - - - - - - - - - - - - - -
    adaptation_prs.name = "Random"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_random)

    adaptation_prs.name = "GIMME_PRS"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_prs)

    adaptation_evl.name = "GIMME_GA"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_evl)

    adaptation_odpip.name = "GIMME_ODPIP"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_odpip)

    adaptation_tab_odpip.name = "GIMME_ODPIP_Tabular"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_tab_odpip)

    adaptation_clink.name = "GIMME_CLink"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_clink)

    adaptation_tab_clink.name = "GIMME_CLink_Tabular"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_tab_clink)

    # - - - - - - - - - - - - - - Explore GIMME-Bootstrap - - - - - - - - - - - - - -
    adaptation_odpip.name = "GIMME_ODPIP_Bootstrap"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_odpip, est_error=0.1)

    adaptation_odpip.name = "GIMME_ODPIP_Bootstrap_HighAcc"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_odpip, est_error=0.05)

    adaptation_odpip.name = "GIMME_ODPIP_Bootstrap_LowAcc"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_odpip, est_error=0.2)

    adaptation_clink.name = "GIMME_CLink_Bootstrap"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_odpip, est_error=0.1)

    adaptation_clink.name = "GIMME_CLink_Bootstrap_HighAcc"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_odpip, est_error=0.05)

    adaptation_clink.name = "GIMME_CLink_Bootstrap_LowAcc"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_odpip, est_error=0.2)

    adaptation_evl.name = "GIMME_GA_Bootstrap"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_evl, est_error=0.1)

    adaptation_evl.name = "GIMME_GA_Bootstrap_HighAcc"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_evl, est_error=0.05)

    adaptation_evl.name = "GIMME_GA_Bootstrap_LowAcc"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_evl, est_error=0.2)

    adaptation_prs.name = "GIMME_PRS_Bootstrap"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_prs, est_error=0.1)

    adaptation_prs.name = "GIMME_PRS_Bootstrap_HighAcc"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_prs, est_error=0.05)

    adaptation_prs.name = "GIMME_PRS_Bootstrap_LowAcc"
    execute_simulations(num_runs, int_prof_2d, max_num_training_iterations, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_prs, est_error=0.2)

    # - - - - - - - - - - - - - - Explore GIP number of dimensions - - - - - - - - - - - - - -
    execute_simulations(num_runs, int_prof_1d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_evl_1d)

    execute_simulations(num_runs, int_prof_3d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_evl_3d)

    execute_simulations(num_runs, int_prof_4d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_evl_4d)

    execute_simulations(num_runs, int_prof_5d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_evl_5d)

    execute_simulations(num_runs, int_prof_6d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_evl_6d)

    # - - - - - - - - - - - - - - Explore GIMME with extreme profiles - - - - - - - - - - - - - -
    adaptation_clink.name = "GIMME_CLink_EP"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_clink, tests_extreme_values=True)

    adaptation_odpip.name = "GIMME_ODPIP_EP"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_odpip, tests_extreme_values=True)

    adaptation_tab_odpip.name = "GIMME_Tabular_ODPIP_EP"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_tab_odpip, tests_extreme_values=True)

    adaptation_evl.name = "GIMME_GA_EP"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_evl, tests_extreme_values=True)

    adaptation_prs.name = "GIMME_PRS_EP"
    execute_simulations(num_runs, int_prof_2d, 0, num_real_iterations,
                        max_num_training_iterations,
                        player_bridge, task_bridge, adaptation_prs, tests_extreme_values=True)

    print("Done!                        ")
