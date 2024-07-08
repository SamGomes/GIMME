import math
from .AlgDefStructs.QualityEvalAlg import *
from .AlgDefStructs.ConfigsGenAlg import *
from .AlgDefStructs.PreferencesEstAlg import *

from .ModelBridge.PlayerModelBridge import PlayerModelBridge
from .ModelBridge.TaskModelBridge import TaskModelBridge


class Adaptation(object):

    def __init__(self):
        self.initialized = False
        self.player_ids = []
        self.task_ids = []
        self.name = "<adaptation with no name>"

        self.configs_gen_alg = None
        self.player_model_bridge = None
        self.task_model_bridge = None

    def init(self,
             player_model_bridge,
             task_model_bridge,
             name,
             configs_gen_alg):

        self.initialized = True
        self.player_ids = []
        self.task_ids = []
        self.name = name

        self.configs_gen_alg = configs_gen_alg
        self.player_model_bridge = player_model_bridge
        self.task_model_bridge = task_model_bridge

        self.configs_gen_alg.init()

    def iterate(self):
        if not self.initialized:
            raise AssertionError('Adaptation not Initialized! Core not executed.')

        self.player_ids = self.player_model_bridge.get_all_player_ids()
        self.task_ids = self.task_model_bridge.get_all_task_ids()

        if len(self.player_ids) < self.configs_gen_alg.min_num_players_per_group:
            raise ValueError('Not enough players to form a group.')

        adapted_config = self.configs_gen_alg.organize()

        adapted_groups = adapted_config["groups"]
        adapted_profiles = adapted_config["profiles"]
        adapted_avg_characteristics = adapted_config["avgCharacteristics"]
        adapted_config["tasks"] = []

        for group_index in range(len(adapted_groups)):
            curr_group = adapted_groups[group_index]
            group_profile = adapted_profiles[group_index]
            avg_state = adapted_avg_characteristics[group_index]

            adapted_task_id = self.select_task(self.task_ids, group_profile, avg_state)
            for player_id in curr_group:
                curr_state = self.player_model_bridge.get_player_curr_state(player_id)
                curr_state.profile = group_profile
                self.player_model_bridge.set_player_tasks(player_id, [adapted_task_id])
                self.player_model_bridge.set_player_characteristics(player_id, curr_state.characteristics)
                self.player_model_bridge.set_player_profile(player_id, curr_state.profile)
                self.player_model_bridge.set_player_group(player_id, curr_group)

            adapted_config["tasks"].append(adapted_task_id)

        return adapted_config

    def select_task(self,
                    possible_task_ids,
                    best_config_profile,
                    avg_state):
        lowest_cost = math.inf

        # if no tasks are available
        best_task_id = -1

        for i in range(len(possible_task_ids)):
            curr_task_id = possible_task_ids[i]

            cost = abs(best_config_profile.sqr_distance_between(self.task_model_bridge.get_task_interactions_profile(
                curr_task_id)) * self.task_model_bridge.get_task_profile_weight(curr_task_id))
            cost += abs(avg_state.ability - self.task_model_bridge.get_min_task_required_ability(
                curr_task_id) * self.task_model_bridge.get_task_difficulty_weight(curr_task_id))

            if cost < lowest_cost:
                lowest_cost = cost
                best_task_id = curr_task_id

        return best_task_id

    # Bootstrap
    def simulate_reaction(self, player_id):
        curr_state = self.player_model_bridge.get_player_curr_state(player_id)
        new_state = self.calc_reaction(state=curr_state, player_id=player_id)

        increases = PlayerState(state_type=new_state.state_type)
        increases.profile = curr_state.profile
        increases.characteristics = PlayerCharacteristics(
            ability=(new_state.characteristics.ability - curr_state.characteristics.ability),
            engagement=new_state.characteristics.engagement)
        self.player_model_bridge.set_and_save_player_state_to_data_frame(player_id, increases, new_state)
        return increases

    def calc_reaction(self, state, player_id):
        preferences = self.player_model_bridge.get_player_real_preferences(player_id)
        num_dims = len(preferences.dimensions)
        new_state = PlayerState(
            state_type=0,
            characteristics=PlayerCharacteristics(
                ability=state.characteristics.ability,
                engagement=state.characteristics.engagement
            ),
            profile=state.profile)
        new_state.characteristics.engagement = 1 - (
                preferences.distance_between(state.profile) / math.sqrt(num_dims))  #between 0 and 1
        if new_state.characteristics.engagement > 1:
            raise ValueError('Something went wrong. Engagement is > 1.')
        ability_increase_sim = (
                new_state.characteristics.engagement * self.player_model_bridge.getBaseLearningRate(player_id))
        new_state.characteristics.ability = new_state.characteristics.ability + ability_increase_sim
        return new_state

    def bootstrap(self, numBootstrapIterations):
        if numBootstrapIterations <= 0:
            raise ValueError('Number of bootstrap iterations must be higher than 0 for this method to be called.')

        numPlayers = len(self.player_model_bridge.get_all_player_ids())
        i = 0
        while i < numBootstrapIterations:
            print("Performing step (" + str(i) + " of " + str(
                numBootstrapIterations) + ") of the bootstrap phase of \"" + str(
                self.name) + "\"...                                                             ", end="\r")
            self.iterate()
            for x in range(numPlayers):
                self.simulate_reaction(player_id=x)
            i += 1

        self.configs_gen_alg.reset()
