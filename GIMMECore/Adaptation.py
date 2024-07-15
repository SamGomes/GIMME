from .AlgDefStructs.ConfigsGenAlg import *
from .AlgDefStructs.PreferencesEstAlg import *

# from .ModelBridge.PlayerModelBridge import PlayerModelBridge
# from .ModelBridge.TaskModelBridge import TaskModelBridge


class Adaptation(object):
    # private members
    __player_model_bridge = None
    __task_model_bridge = None
    __name = None
    __configs_gen_alg = None

    __player_ids = None
    __task_ids = None

    def __init__(self, name="<adaptation with no name>",
                 player_model_bridge=None,
                 task_model_bridge=None,
                 configs_gen_alg=None):

        self.__player_model_bridge = player_model_bridge
        self.__task_model_bridge = task_model_bridge
        self.__name = name
        self.__configs_gen_alg = configs_gen_alg

        if self.__player_model_bridge is None:
            self.__player_ids = []
        else:
            self.__player_ids = self.__player_model_bridge.get_all_player_ids()
        if self.__task_model_bridge is None:
            self.__task_ids = []
        else:
            self.__task_ids = self.__task_model_bridge.get_all_task_ids()

    def iterate(self):
        missing_keys = []
        attrs = self.__dict__
        for key in attrs.keys():
            if attrs[key] is None:
                missing_keys.append(key)
        if len(missing_keys) > 0:
            raise AssertionError(
                "Adaptation with name: '" + self.__name + "' is not ready (missing the following parameters: "
                + str(missing_keys) + "). Core not executed.")

        self.__player_ids = self.__player_model_bridge.get_all_player_ids()
        self.__task_ids = self.__task_model_bridge.get_all_task_ids()

        if len(self.__player_ids) < self.__configs_gen_alg._min_num_players_per_group:
            raise ValueError('Not enough players to form a group.')

        adapted_config = self.__configs_gen_alg.organize()

        adapted_groups = adapted_config["groups"]
        adapted_profiles = adapted_config["profiles"]
        adapted_avg_characteristics = adapted_config["avgCharacteristics"]
        adapted_config["tasks"] = []

        # print(adapted_config)

        for group_index in range(len(adapted_groups)):
            curr_group = adapted_groups[group_index]
            group_profile = adapted_profiles[group_index]
            avg_state = adapted_avg_characteristics[group_index]

            adapted_task_id = self.__select_task(self.__task_ids, group_profile, avg_state)
            for player_id in curr_group:
                curr_state = self.__player_model_bridge.get_player_curr_state(player_id)
                curr_state.profile = group_profile
                self.__player_model_bridge.set_player_tasks(player_id, [adapted_task_id])
                self.__player_model_bridge.set_player_characteristics(player_id, curr_state.characteristics)
                self.__player_model_bridge.set_player_profile(player_id, curr_state.profile)
                self.__player_model_bridge.set_player_group(player_id, curr_group)

            adapted_config["tasks"].append(adapted_task_id)

        return adapted_config

    # def reset_configs_gen_alg(self):
    #     self.__configs_gen_alg.reset()

    def set_name(self, name):
        self.__name = name

    def get_name(self):
        return self.__name

    def __select_task(self,
                      possible_task_ids,
                      best_config_profile,
                      avg_characteristics):
        lowest_cost = math.inf

        # if no tasks are available
        best_task_id = -1

        for i in range(len(possible_task_ids)):
            curr_task_id = possible_task_ids[i]

            cost = abs(best_config_profile.sqr_distance_between(self.__task_model_bridge.get_task_interactions_profile(
                curr_task_id)) * self.__task_model_bridge.get_task_profile_weight(curr_task_id))
            cost += abs(avg_characteristics.ability - self.__task_model_bridge.get_min_task_required_ability(
                curr_task_id) * self.__task_model_bridge.get_task_difficulty_weight(curr_task_id))

            if cost < lowest_cost:
                lowest_cost = cost
                best_task_id = curr_task_id

        return best_task_id

    # Bootstrap
    def __simulate_reaction(self, player_id):
        curr_state = self.__player_model_bridge.get_player_curr_state(player_id)
        new_state = self.__calc_reaction(state=curr_state, player_id=player_id)

        increases = PlayerState(type=new_state.type)
        increases.profile = curr_state.profile
        increases.characteristics = PlayerCharacteristics(
            ability=(new_state.characteristics.ability - curr_state.characteristics.ability),
            engagement=new_state.characteristics.engagement)
        self.__player_model_bridge.set_and_save_player_state_to_data_frame(player_id, increases, new_state)
        return increases

    def __calc_reaction(self, state, player_id):
        preferences = self.__player_model_bridge.get_player_real_preferences(player_id)
        num_dims = len(preferences.dimensions)
        new_state = PlayerState(
            type=0,
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
                new_state.characteristics.engagement * self.__player_model_bridge.get_base_learning_rate(player_id))
        new_state.characteristics.ability = new_state.characteristics.ability + ability_increase_sim
        return new_state

    def bootstrap(self, num_bootstrap_iterations):
        if num_bootstrap_iterations <= 0:
            raise ValueError('Number of bootstrap iterations must be higher than 0 for this method to be called.')

        num_players = len(self.__player_model_bridge.get_all_player_ids())
        i = 0
        while i < num_bootstrap_iterations:
            print("Performing step (" + str(i) + " of " + str(
                num_bootstrap_iterations) + ") of the bootstrap phase of \"" + str(
                self.__name) + "\"...                                                             ", end="\r")
            self.iterate()
            for x in range(num_players):
                self.__simulate_reaction(player_id=x)
            i += 1

        # self.__configs_gen_alg.reset()
